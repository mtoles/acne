import sys
import os
import enum
# Add parent directory to path to import pt_features and models
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from flask import Flask, render_template, request, jsonify
import pandas as pd
from pathlib import Path
import yaml
import requests
from concurrent.futures import ThreadPoolExecutor, as_completed

from pt_features import PtFeaturesMeta
from models import MrModel

app = Flask(__name__)

# Load MODEL_ID from config.yml
def load_model_id():
    try:
        config_path = Path(__file__).parent.parent / "config.yml"
        with open(config_path, "r") as file:
            config = yaml.safe_load(file)
            return config["model"]["id"]
    except (FileNotFoundError, KeyError, yaml.YAMLError) as e:
        print(f"Error loading config.yml: {e}")
        return None

MODEL_ID = load_model_id()

def check_vllm_server():
    """Check if vLLM server is running"""
    try:
        response = requests.get("http://localhost:8000/health", timeout=2)
        return response.status_code == 200
    except:
        return False

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/api/features', methods=['GET'])
def get_features():
    """Get list of features from pt_features registry that have a query method"""
    features = []
    for name, cls in PtFeaturesMeta.registry.items():
        if hasattr(cls, 'query'):
            try:
                # Try to get a sample query
                sample_keyword = cls.keywords[0] if hasattr(cls, 'keywords') and cls.keywords else "sample"
                query = cls.query(keyword=sample_keyword)
                
                # Handle options - convert OptionType to string for JSON serialization
                options = cls.options if hasattr(cls, 'options') else []
                if isinstance(options, enum.Enum):
                    options = options.value
                elif not isinstance(options, list):
                    options = str(options)
                
                features.append({
                    'name': name,
                    'query': query,
                    'options': options
                })
            except Exception as e:
                # If query method exists but fails, still include it
                options = cls.options if hasattr(cls, 'options') else []
                if isinstance(options, enum.Enum):
                    options = options.value
                elif not isinstance(options, list):
                    options = str(options)
                    
                features.append({
                    'name': name,
                    'query': '',
                    'options': options
                })
    
    return jsonify({'features': sorted(features, key=lambda x: x['name'])})

@app.route('/api/check_server', methods=['GET'])
def check_server():
    """Check if vLLM server is running"""
    is_running = check_vllm_server()
    return jsonify({'running': is_running, 'model_id': MODEL_ID})

@app.route('/api/run', methods=['POST'])
def run_query():
    """Run the comparison query"""
    data = request.json
    feature_name = data.get('feature')
    n = int(data.get('n', 100))
    custom_query = data.get('query', '').strip()
    options_str = data.get('options', '').strip()
    
    # Check if vLLM server is running
    if not check_vllm_server():
        return jsonify({
            'error': 'vLLM server is not running. Please start it using start_vllm.sh'
        }), 503
    
    # Parse options
    if options_str:
        custom_options = [opt.strip() for opt in options_str.split(',')]
    else:
        custom_options = None
    
    try:
        # Get the feature class
        target_cls = PtFeaturesMeta.registry.get(feature_name)
        if not target_cls:
            return jsonify({'error': f'Feature {feature_name} not found'}), 400
        
        # Initialize model
        model = MrModel(model_id=MODEL_ID)
        
        # Load labeled data
        labeled_data_dir = Path(__file__).parent.parent / "labeled_data" / "mgb" / feature_name
        chunk_file = labeled_data_dir / f"{feature_name}_chunks.xlsx"
        
        if not chunk_file.exists():
            return jsonify({'error': f'Labeled data not found for {feature_name}'}), 404
        
        # Load data
        annot_df = pd.read_excel(chunk_file)
        annot_df = annot_df[annot_df["val_unified"].notna()]
        annot_df["val_unified"] = annot_df["val_unified"].astype(str)
        
        # Limit to n examples
        if len(annot_df) > n:
            annot_df = annot_df.sample(n=n, random_state=42).sort_index()
        
        chunk_df = annot_df.copy()
        
        # Process chunks
        def process_single_chunk(args):
            idx, chunk, found_kw = args
            kwargs = {}
            if custom_query:
                kwargs['custom_query'] = custom_query
            if custom_options:
                kwargs['custom_options'] = custom_options
            
            pred_dict = target_cls.forward(
                model=model, 
                chunk=chunk, 
                keyword=found_kw, 
                inference_type="cot",
                **kwargs
            )
            return idx, pred_dict
        
        # Prepare arguments - use actual DataFrame index to avoid alignment issues
        chunk_args = [
            (idx, chunk, found_kw)
            for idx, chunk, found_kw in zip(
                chunk_df.index,
                chunk_df["chunk"],
                chunk_df["found_keywords"]
            )
        ]
        
        # Process with ThreadPoolExecutor
        predictions = {}
        with ThreadPoolExecutor(max_workers=100) as executor:
            future_to_index = {
                executor.submit(process_single_chunk, args): args[0] for args in chunk_args
            }
            
            for future in as_completed(future_to_index):
                idx, preds_for_chunk = future.result()
                for key, value in preds_for_chunk.items():
                    if key not in predictions:
                        predictions[key] = {}
                    predictions[key][idx] = value
        
        # Add predictions to dataframe - use loc to ensure proper alignment
        for key, pred_dict in predictions.items():
            for idx, value in pred_dict.items():
                chunk_df.loc[idx, key] = value
        
        # Get prediction column
        pred_columns = [col for col in chunk_df.columns if col == feature_name]
        if not pred_columns:
            pred_columns = [col for col in predictions.keys()]
        
        if not pred_columns:
            return jsonify({'error': 'No predictions generated'}), 500
        
        pred_col = pred_columns[0]
        ground_truth = chunk_df["val_unified"]
        preds = chunk_df[pred_col]
        
        # Calculate metrics
        correct_mask = preds == ground_truth
        correct = correct_mask.sum()
        total = len(preds)
        accuracy = correct / total if total > 0 else 0
        
        # Prepare results
        correct_results = []
        incorrect_results = []
        
        for idx, row in chunk_df.iterrows():
            result = {
                'chunk': row['chunk'],
                'gt_answer': row['val_unified'],
                'predicted_answer': row[pred_col],
                'keyword': row['found_keywords']
            }
            
            if row[pred_col] == row['val_unified']:
                correct_results.append(result)
            else:
                incorrect_results.append(result)
        
        return jsonify({
            'summary': {
                'accuracy': f"{accuracy * 100:.2f}%",
                'correct': int(correct),
                'total': int(total),
                'query': custom_query if custom_query else (target_cls.query(keyword="sample") if hasattr(target_cls, 'query') else ''),
                'options': custom_options if custom_options else (target_cls.options if hasattr(target_cls, 'options') else [])
            },
            'correct': correct_results,
            'incorrect': incorrect_results
        })
        
    except Exception as e:
        import traceback
        return jsonify({
            'error': str(e),
            'traceback': traceback.format_exc()
        }), 500

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)

