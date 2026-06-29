import sys
import os
import enum
# Add parent directory to path to import pt_features and models
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from flask import Flask, render_template, request, jsonify, session, redirect, url_for
import pandas as pd
from pathlib import Path
import yaml
import requests
from concurrent.futures import ThreadPoolExecutor, as_completed

from pt_features import (
    PtFeaturesMeta,
    PtNumericFeatureBase,
    find_baseline_prompt_from_prompt_iteration_labels,
)
from models import MrModel
from utils import OptionType, compute_numeric_pct_error, compute_numeric_abs_error
# Shared with auto_prompt so the UI uses byte-identical train/val/test splits (no drift).
from feature_data import load_acne_feature_data


# Residents-study "special" features shown at the TOP of the feature dropdown. Each reuses an
# existing feature class but populates the query box with the ORIGINAL short one-sentence prompt
# (not the full optimized prompt). They always run on mgb data (the app's only data source).
# (display_name, real_feature_class_name, override_query_or_None)
#   override None  -> use that class's baseline prompt from the prompt-tuning sheet
#   override set   -> use this literal query (the numeric duration feature has no short baseline)
SPECIAL_FEATURES_ORDER = [
    ("Residents Study 1 - Alcohol Amount", "alcohol_amount", None),
    ("Residents Study 2 - Cancer Stage at Diagnosis", "cancer_stage_at_diagnosis", None),
    (
        "Residents Study 3 - Antibiotic Duration Numeric",
        "antibiotic_duration_numeric",
        "How many days total did the patient take {keyword}? Answer with a single integer for "
        "the total number of days, 0 if there is no indication of antibiotic use, or F if taken "
        "but the duration is unknown.",
    ),
]

# display_name -> real feature class name
SPECIAL_FEATURE_TO_REAL = {disp: real for disp, real, _ in SPECIAL_FEATURES_ORDER}


def resolve_feature_name(feature_name):
    """Map a Residents-study display name to its underlying feature class name; pass others through."""
    if feature_name in SPECIAL_FEATURE_TO_REAL:
        return SPECIAL_FEATURE_TO_REAL[feature_name]
    return feature_name


def serialize_options(target_cls):
    """Convert a feature class's options to a JSON-serializable form (list, or "numeric"/"date")."""
    options = target_cls.options if hasattr(target_cls, "options") else []
    if isinstance(options, enum.Enum):
        return options.value
    if not isinstance(options, list):
        return str(options)
    return options


def build_special_feature_entries():
    """Build the /api/features entries for the Residents-study features, in defined order."""
    entries = []
    for display_name, real_name, override_query in SPECIAL_FEATURES_ORDER:
        target_cls = PtFeaturesMeta.registry[real_name]
        if override_query is not None:
            query = override_query
        else:
            # keyword="{keyword}" keeps the placeholder so it's substituted per-chunk at run time
            query = find_baseline_prompt_from_prompt_iteration_labels(real_name, keyword="{keyword}")
        entries.append({
            "name": display_name,
            "query": query,
            "options": serialize_options(target_cls),
        })
    return entries


def load_train_val_split(feature_name):
    """The UI is the procedurally-identical human version of auto_prompt's optimization loop, so
    it reuses auto_prompt's exact split via the shared feature_data.load_acne_feature_data.

    Returns (train_df, val_df) on the mgb data. The test third is intentionally dropped here --
    the human optimizer must never see it (it is auto_prompt's final holdout). downsample=None;
    any example-count cap is applied by the caller (see /api/run).
    """
    train_df, val_df, _test_df, _test_full_df, _target_cls = load_acne_feature_data(
        feature_name, data_source="mgb", downsample=None
    )
    return train_df, val_df


app = Flask(__name__)
app.secret_key = "acne-app-session-key-2026"

# Password required to access the app.
APP_PASSWORD = "acne2026"


@app.before_request
def require_login():
    """Gate every route (except the login page and static files) behind the password."""
    if request.endpoint in ("login", "static"):
        return None
    if session.get("authenticated"):
        return None
    return redirect(url_for("login"))


@app.route('/login', methods=['GET', 'POST'])
def login():
    error = None
    if request.method == 'POST':
        if request.form.get('password') == APP_PASSWORD:
            session['authenticated'] = True
            return redirect(url_for('index'))
        error = 'Incorrect password'
    return render_template('login.html', error=error)


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

# Single source of truth for the vLLM endpoints so the health check and the inference client
# can never drift to different ports (the bug: health-checked :8000 while MrModel defaulted to
# :8010 -> "Connection error" at run time). Honors the same env vars MrModel reads. By default the
# app round-robins across the local vLLM (:8000, the port start_vllm.sh / run_both.sh launch on)
# AND the remote "communication" vLLM, exactly like run_both.sh -- MrModel load-balances across the
# comma-separated list (see models.py _route/_pick). The remote endpoint is on the tailnet, so it
# must go through the Tailscale userspace proxy (VLLM_PROXY); MrModel applies that proxy ONLY to
# non-local URLs, so defaulting it here leaves the local endpoint connecting directly.
COMMUNICATION_VLLM_URL = os.environ.get(
    "COMMUNICATION_VLLM_URL", "http://100.119.93.116:9090/v1"
)
os.environ.setdefault("VLLM_PROXY", "http://localhost:1055")

def _resolve_vllm_base_urls():
    base_url = os.environ.get("VLLM_BASE_URL")
    if base_url is None:
        vllm_port = os.environ.get("VLLM_PORT", "8000")
        local_url = f"http://localhost:{vllm_port}/v1"
        base_url = f"{local_url},{COMMUNICATION_VLLM_URL}"
    return [u.strip() for u in base_url.split(",") if u.strip()]

VLLM_BASE_URLS = _resolve_vllm_base_urls()
# Full comma-separated list handed to MrModel (round-robined across every endpoint).
VLLM_BASE_URL = ",".join(VLLM_BASE_URLS)
# Health-check only the first (local) endpoint -- it's the one that must be up to serve the UI.
VLLM_HEALTH_URL = VLLM_BASE_URLS[0]

def check_vllm_server():
    """Check if the (local) vLLM server is running"""
    health_url = VLLM_HEALTH_URL.rstrip("/").removesuffix("/v1") + "/health"
    try:
        response = requests.get(health_url, timeout=2)
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
                # Get query template with {keyword} placeholder so the UI shows a template
                query = cls.query(keyword="{keyword}")
                
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
    
    # Residents-study special features pinned to the top, then the rest sorted by name.
    special_entries = build_special_feature_entries()
    return jsonify({'features': special_entries + sorted(features, key=lambda x: x['name'])})

@app.route('/api/check_server', methods=['GET'])
def check_server():
    """Check if vLLM server is running"""
    is_running = check_vllm_server()
    return jsonify({'running': is_running, 'model_id': MODEL_ID})

@app.route('/api/dataset_size', methods=['GET'])
def get_dataset_size():
    """Get the size of the train dataset for a given feature"""
    feature_name = request.args.get('feature')

    if not feature_name:
        return jsonify({'error': 'Feature name is required'}), 400

    feature_name = resolve_feature_name(feature_name)

    try:
        # Use the same train/val split the run uses (auto_prompt-identical). Default the UI's
        # example count to the full train size; val size is reported for display only.
        train_df, val_df = load_train_val_split(feature_name)
        if train_df is None:
            return jsonify({'error': f'Labeled data not found for {feature_name}'}), 404

        return jsonify({'size': len(train_df), 'val_size': len(val_df)})

    except Exception as e:
        import traceback
        return jsonify({
            'error': str(e),
            'traceback': traceback.format_exc()
        }), 500

@app.route('/api/run', methods=['POST'])
def run_query():
    """Run the comparison query"""
    data = request.json
    feature_name = resolve_feature_name(data.get('feature'))
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

        # Initialize model on the SAME endpoint the health check verified.
        model = MrModel(model_id=MODEL_ID, base_url=VLLM_BASE_URL)

        # auto_prompt-identical train/val split (see load_train_val_split). The human plays the
        # optimizer: they see ALL errors on TRAIN and edit the prompt; VAL is reported as a bare
        # accuracy number (no examples) -- the accept/reject signal. TEST is never shown here.
        train_df, val_df = load_train_val_split(feature_name)
        if train_df is None:
            return jsonify({'error': f'Labeled data not found for {feature_name}'}), 404

        if len(train_df) == 0:
            return jsonify({'error': f'There are no labeled examples for {feature_name}'}), 400

        # Optional speed cap applied equally to TRAIN and VAL (the analog of auto_prompt's
        # --downsample, which caps every split). Default n is the full train size, so by default
        # both splits run in full. Same seed -> deterministic subsample.
        if len(train_df) > n:
            train_df = train_df.sample(n=n, random_state=42).sort_index()
        if len(val_df) > n:
            val_df = val_df.sample(n=n, random_state=42).sort_index()

        is_numeric_feature = issubclass(target_cls, PtNumericFeatureBase)

        def normalize_numeric(val):
            val_str = str(val).strip()
            if val_str.upper() == "F":
                return "F"
            try:
                return str(int(float(val_str)))
            except (ValueError, TypeError):
                return val_str

        def run_inference(df):
            """Run the prompt over df's chunks; return (chunk_df_with_preds, pred_col)."""
            chunk_df = df.copy()

            def process_single_chunk(args):
                idx, chunk, found_kw = args
                kwargs = {}
                if custom_query and custom_query.strip():
                    # Substitute {keyword} placeholder with the actual found keyword
                    kwargs['custom_query'] = custom_query.replace("{keyword}", found_kw)
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

            # Use the actual DataFrame index to avoid alignment issues
            chunk_args = [
                (idx, chunk, found_kw)
                for idx, chunk, found_kw in zip(
                    chunk_df.index, chunk_df["chunk"], chunk_df["found_keywords"]
                )
            ]

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

            for key, pred_dict in predictions.items():
                for idx, value in pred_dict.items():
                    chunk_df.loc[idx, key] = value

            pred_columns = [col for col in chunk_df.columns if col == feature_name]
            if not pred_columns:
                pred_columns = [col for col in predictions.keys()]
            pred_col = pred_columns[0] if pred_columns else None
            return chunk_df, pred_col

        def score(chunk_df, pred_col):
            """Compute accuracy and (for numeric features) error metrics on a scored chunk_df.
            Mutates chunk_df's pred/gt columns to normalized numeric form when applicable."""
            if is_numeric_feature:
                chunk_df[pred_col] = chunk_df[pred_col].apply(normalize_numeric)
                chunk_df["val_unified"] = chunk_df["val_unified"].apply(normalize_numeric)
            ground_truth = chunk_df["val_unified"]
            preds = chunk_df[pred_col]
            total = len(preds)
            correct = int((preds == ground_truth).sum())
            accuracy = correct / total if total > 0 else 0
            avg_pct_error = None
            avg_abs_error = None
            if is_numeric_feature and total > 0:
                pct_errors = [compute_numeric_pct_error(p, g) for p, g in zip(preds, ground_truth)]
                avg_pct_error = sum(pct_errors) / len(pct_errors)
                abs_errors = [compute_numeric_abs_error(p, g) for p, g in zip(preds, ground_truth)]
                abs_errors_valid = [e for e in abs_errors if e is not None]
                avg_abs_error = sum(abs_errors_valid) / len(abs_errors_valid) if abs_errors_valid else None
            return {
                'accuracy': accuracy, 'correct': correct, 'total': total,
                'avg_pct_error': avg_pct_error, 'avg_abs_error': avg_abs_error,
            }

        # --- TRAIN: full error breakdown (with examples) ---
        train_chunk_df, train_pred_col = run_inference(train_df)
        if not train_pred_col:
            return jsonify({'error': 'No predictions generated'}), 500
        train_metrics = score(train_chunk_df, train_pred_col)

        correct_results = []
        incorrect_results = []
        for idx, row in train_chunk_df.iterrows():
            result = {
                'chunk': row['chunk'],
                'gt_answer': row['val_unified'],
                'predicted_answer': row[train_pred_col],
                'keyword': row['found_keywords']
            }
            if row[train_pred_col] == row['val_unified']:
                correct_results.append(result)
            else:
                incorrect_results.append(result)

        # --- VAL: hidden accuracy number only (no examples returned) ---
        val_metrics = None
        if len(val_df) > 0:
            val_chunk_df, val_pred_col = run_inference(val_df)
            if val_pred_col:
                val_metrics = score(val_chunk_df, val_pred_col)

        # Convert options to JSON-serializable format
        if custom_options:
            serialized_options = custom_options
        elif hasattr(target_cls, 'options'):
            options = target_cls.options
            if isinstance(options, OptionType):
                serialized_options = options.value
            elif isinstance(options, enum.Enum):
                serialized_options = options.value
            else:
                serialized_options = options
        else:
            serialized_options = []

        summary = {
            'accuracy': f"{train_metrics['accuracy'] * 100:.2f}%",
            'correct': train_metrics['correct'],
            'total': train_metrics['total'],
            'query': custom_query if custom_query else (target_cls.query(keyword="sample") if hasattr(target_cls, 'query') else ''),
            'options': serialized_options,
        }
        if train_metrics['avg_pct_error'] is not None:
            summary['avg_pct_error'] = f"{train_metrics['avg_pct_error']:.1f}%"
        if train_metrics['avg_abs_error'] is not None:
            summary['avg_abs_error'] = f"{train_metrics['avg_abs_error']:.1f} days"

        # Hidden validation metrics: number(s) only, never the underlying examples.
        if val_metrics is not None:
            summary['val_accuracy'] = f"{val_metrics['accuracy'] * 100:.2f}%"
            summary['val_total'] = val_metrics['total']
            if val_metrics['avg_pct_error'] is not None:
                summary['val_avg_pct_error'] = f"{val_metrics['avg_pct_error']:.1f}%"
            if val_metrics['avg_abs_error'] is not None:
                summary['val_avg_abs_error'] = f"{val_metrics['avg_abs_error']:.1f} days"

        return jsonify({
            'summary': summary,
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
    # use_reloader=False: the debug reloader watches every .py in the project and restarts the
    # server on any change, which KILLS in-flight /api/run requests (looks like "no throughput"
    # whenever pipeline .py files are being edited). Keep the debugger, drop the file-watch.
    # threaded=True so concurrent requests (and the /api/check_server poll) don't block a run.
    app.run(debug=True, use_reloader=False, threaded=True, host='0.0.0.0', port=5001)

