import pandas as pd
import os
import time
from pathlib import Path
from tqdm import tqdm

from make_db import store_dir as file_store_parent_dir
from pt_features import PtFeaturesMeta, PtFeatureBase, PtDateFeatureBase
from utils import chunk_text, has_keyword
from models import MrModel, DummyModel
from sqlalchemy import create_engine, text
from make_db import db_url

tqdm.pandas()  # Enable tqdm for pandas operations

# Initialize model
model = MrModel()

# # Test the model with a single example
# test_question = "Does this patient have a fever? A. Yes, B. No"
# test_chunk = "Patient presents with temperature of 101.2°F and chills."
# test_history = model.format_chunk_qs(test_question, [test_chunk], options=["A", "B"])
# result = model.predict_single_with_logit_trick(test_history, output_choices=set(["A", "B"]))
# print(f"\nTest prediction result: {result}")

# get all null values from the db that are LlmFeatureBase
classes = [
    cls for cls in PtFeaturesMeta.registry.values() if issubclass(cls, PtFeatureBase)
]
db = create_engine(db_url)


def get_samples(column_name, annot_report_numbers, downsample_size=100):
    print(f"\nGetting samples for column {column_name}:")

    with db.connect() as conn:
        # Quote each report number and join them
        quoted_numbers = [f"'{num}'" for num in annot_report_numbers]
        # Get sample of rows, limited to downsample_size if specified
        base_query = f"SELECT DISTINCT * FROM vis WHERE Report_Number IN ({','.join(quoted_numbers)})"
        if downsample_size is not None:
            base_query += f" LIMIT {downsample_size}"
        sample_query = text(base_query)
        sample_result = conn.execute(sample_query)
        df = pd.DataFrame(sample_result.fetchall(), columns=sample_result.keys())
        print(f"Found {len(df)} unique samples")
        return df


def process_file(file_path):
    print(f"\nProcessing {file_path}")
    feature_name = file_path.stem.replace("_validation", "")

    # Create feature directory structure within preds
    preds_dir = Path("preds")
    feature_dir = preds_dir / feature_name
    feature_dir.mkdir(parents=True, exist_ok=True)

    # Read the validation file
    annot_df = pd.read_excel(file_path)
    annot_df["Report_Number"] = annot_df["Report_Number"].astype(str)
    annot_df = annot_df.dropna(subset=["Report_Number"])
    annot_df["val_unified"] = annot_df["val_dorsa"].where(
        pd.notna(annot_df["val_dorsa"]), annot_df["val_james"]
    )
    annot_df = annot_df.dropna(subset=["val_unified"])
    if len(annot_df) == 0:
        print(f"skipping {file_path} because it has no samples with validation data")
        return
    annot_report_numbers = annot_df["Report_Number"].unique()[:10] # TESTING DOWNSAMPLE

    target_cls = PtFeaturesMeta.registry[feature_name]
    # if issubclass(target_cls, PtDateFeatureBase):
    #     print(f"skipping {feature_name} because it is a date feature")
    #     return

    # Get samples and process
    df = get_samples(feature_name, annot_report_numbers)
    # drop rows where Report_Number is null
    df = df.set_index("Report_Number")
    df["val_unified"] = annot_df.set_index("Report_Number")["val_unified"]

    # Process chunks
    df = df.assign(chunk=df["Report_Text"].progress_apply(chunk_text))
    chunk_df = df.explode("chunk")
    chunk_df["found_keywords"] = chunk_df["chunk"].progress_apply(
        has_keyword, keywords=target_cls.keywords
    )
    chunk_df["has_kw"] = chunk_df["found_keywords"].apply(lambda x: len(x) > 0)

    percent_has_kw = len(chunk_df[chunk_df["has_kw"]]) / len(chunk_df)
    # chunk_df = chunk_df[chunk_df["has_kw"]]

    # Apply model to chunks
    preds = []
    queries = []
    for chunk, found_keywords in tqdm(zip(chunk_df[chunk_df["has_kw"]]["chunk"], chunk_df[chunk_df["has_kw"]]["found_keywords"])):
        preds_for_chunk = {}
        queries_for_chunk = []
        for found_keyword in found_keywords:
            # All queries are now callable functions - pass found keywords instead of all keywords
            query = target_cls.query(chunk=chunk, keywords=found_keyword)
            queries_for_chunk.append(query)
            
            history = model.format_chunk_qs(
                q=query, chunk=chunk, options=target_cls.options
            )
            if issubclass(target_cls, PtFeatureBase):
                pred = model.predict_single_with_logit_trick(
                    history,
                    output_choices=set(target_cls.options) if target_cls.options else None,
                )
            else:
                pred = model.predict_single(
                    history,
                    max_tokens=target_cls.max_tokens
                )
            preds_for_chunk[found_keyword] = pred
        preds.append(preds_for_chunk)
        queries.append(queries_for_chunk)

    chunk_df["chunk_pred"] = "NO_KW"
    chunk_df["chunk_query"] = ""
    chunk_df.loc[chunk_df["has_kw"], "chunk_pred"] = preds
    chunk_df.loc[chunk_df["has_kw"], "chunk_query"] = pd.Series(queries, dtype=object, index=chunk_df.index[chunk_df["has_kw"]]) # avoid error from list of lists

    # Group predictions
    df["pooled_answers"] = chunk_df.groupby(chunk_df.index)["chunk_pred"].agg(list)
    df["report_no_pred"] = df["pooled_answers"].apply(target_cls.pooling_fn)

    # Calculate accuracy
    df["is_correct"] = df["report_no_pred"] == df["val_unified"]
    accuracy = df["is_correct"].mean() * 100
    correct_count = df["is_correct"].sum()
    total_count = len(df)

    # Save results
    results = {
        "accuracy": accuracy,
        "correct_count": correct_count,
        "total_count": total_count,
        "percent_has_kw": percent_has_kw * 100,
        "incorrect_examples": [],
    }

    # Collect incorrect examples
    for index, row in df[df["is_correct"] == False].iterrows():
        results["incorrect_examples"].append(
            {
                "Report_Number": index,
                "Report_Text": row["Report_Text"],
                "Prediction": row["report_no_pred"],
                "Ground_Truth": row["val_unified"],
            }
        )

    # Save summary to txt within feature directory
    summary_path = feature_dir / "summary.txt"
    with open(summary_path, "w") as f:
        f.write(f"#########################\n\n")
        f.write(f"Results for {feature_name}\n")
        f.write(f"Accuracy: {accuracy:.2f}% ({correct_count} / {total_count})\n")
        f.write(f"Percentage of chunks with keywords: {percent_has_kw*100:.2f}%\n\n")
        f.write("Incorrect Examples:\n")
        for ex in results["incorrect_examples"]:
            f.write(f"\nReport Number: {ex['Report_Number']}\n")
            f.write(f"Prediction: {ex['Prediction']}\n")
            f.write(f"Ground Truth: {ex['Ground_Truth']}\n")

    # Save individual chunk summaries
    incorrect_examples = df[df["is_correct"] == False]
    incorrect_chunks = chunk_df[
        chunk_df.index.isin(incorrect_examples.index) & chunk_df["has_kw"]
    ]

    for index, row in incorrect_examples.iterrows():
        # Create report number directory within feature directory
        report_dir = feature_dir / str(index)
        report_dir.mkdir(exist_ok=True)

        # Save report summary
        report_summary_path = report_dir / "report_summary.txt"
        with open(report_summary_path, "w") as f:
            f.write(f"Report Number: {index}\n")
            f.write(f"Report Text: {row['Report_Text']}\n")
            f.write(f"Prediction: {row['report_no_pred']}\n")
            f.write(f"Ground Truth: {row['val_unified']}\n\n")
            # Get the first query used for this report (they should all be similar for the same report)
            report_chunks = chunk_df[chunk_df.index == index]
            assert len(report_chunks) > 0
            # if not report_chunks.empty and len(report_chunks[report_chunks["has_kw"]]) > 0:
            first_query = report_chunks[report_chunks["has_kw"]]["chunk_query"].iloc[0]
            f.write(f"Query: {first_query}\n")
            # else:
            #     f.write(f"Query Function: {target_cls.query.__name__}\n")

        # Save individual chunks
        chunks_from_this_report = incorrect_chunks[incorrect_chunks.index == index]
        for i, chunk_row in enumerate(chunks_from_this_report.itertuples()):
            chunk_file_path = report_dir / f"chunk_{i+1}.txt"
            with open(chunk_file_path, "w") as f:
                f.write(f"Query: {chunk_row.chunk_query}\n")
                f.write(f"Ground Truth: {row['val_unified']}\n")
                f.write(f"Prediction: {chunk_row.chunk_pred}\n")
                f.write(f"Keywords: {', '.join(target_cls.keywords)}\n")
                f.write(f"Found keywords: {', '.join(chunk_row.found_keywords)}\n\n")
                f.write(f"Chunk:\n{chunk_row.chunk}\n")

    # Save predictions to new Excel file within feature directory
    preds_df = annot_df.copy().set_index("Report_Number").astype(str)
    preds_df["report_no_pred"] = df[
        "report_no_pred"
    ]  # results in ~10% of rows being NaN
    preds_path = feature_dir / f"{file_path.stem}.xlsx"
    preds_df = preds_df[["EPIC_PMRN", "val_unified", "report_no_pred"]]
    preds_df.to_excel(preds_path, index=True)

    return results


def main():
    labeled_data_dir = Path("labeled_data")
    output_dir = Path("preds")
    output_dir.mkdir(exist_ok=True)  # Create preds directory if it doesn't exist

    # Get all xlsx files, sort alphabetically, and filter out _preds files
    excel_files = sorted(
        [f for f in labeled_data_dir.glob("*.xlsx") if not f.stem.endswith("_preds")]
    )

    all_results = {}
    excel_files = [Path("labeled_data/antibiotic_duration_validation.xlsx")]
    # excel_files = [Path("labeled_data/")]
    for file_path in excel_files:
        print(f"\nProcessing {file_path.name}...")
        results = process_file(file_path)
        all_results[file_path.stem] = results

    # Print and write overall summary
    print("\nOverall Summary:")
    summary_path = Path("preds") / "overall_summary.txt"
    with open(summary_path, "w") as f:
        f.write("Overall Summary:\n")
        for feature, results in all_results.items():
            if results:
                summary_line = f"\n{feature}:\nAccuracy: {results['accuracy']:.2f}% ({results['correct_count']} / {results['total_count']})"
                print(summary_line)
                f.write(summary_line)


if __name__ == "__main__":
    main()
