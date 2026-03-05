from pathlib import Path

from utils import compute_similarity_metrics_table, ensure_dirs, merge_prediction_csvs, plot_metrics_bargrid

# -----------------------------
# User config
# -----------------------------
input_paths = {
    "zs_wc": Path("outputs/predict-finetuned-noctx-20260217-153151/predictions.csv"),
    "zs_woc": Path("outputs/predict-finetuned-noctx-20260217-153151/predictions.csv"),
    "ft_wc": Path("outputs/predict-finetuned-noctx-20260217-153151/predictions.csv"),
    "ft_woc": Path("outputs/predict-finetuned-noctx-20260217-153151/predictions.csv"),
}
output_path = Path("outputs/evaluate_v2")
metrics_csv_path = output_path / "evaluation_metrics_v2.csv"
figure_path = output_path / "evaluation_metrics_v2.png"

bertscore_model_type = "distilbert-base-uncased"
bertscore_batch_size = 16
bertscore_lang = "en"
bertscore_device = "cpu"

# -----------------------------
# Run
# -----------------------------
print("Starting evaluation v2...")
print(f"Output directory: {output_path}")
ensure_dirs(output_path)

print("Merging input CSVs on ['pmid', 'question']...")
merged_df, model_names = merge_prediction_csvs(
    input_paths=input_paths,
    key_cols=("pmid", "question"),
    answer_col="answer",
    prediction_col="predicted_answer",
)
print(f"Common rows kept: {len(merged_df)}")
print(f"Models to score: {', '.join(model_names)}")

print("Computing BLEU / ROUGE-L / BERTScore-F1...")
metrics_df = compute_similarity_metrics_table(
    merged_df=merged_df,
    model_names=model_names,
    answer_col="answer",
    bertscore_model_type=bertscore_model_type,
    bertscore_batch_size=bertscore_batch_size,
    bertscore_lang=bertscore_lang,
    bertscore_device=bertscore_device,
)

print(f"Saving metrics table to: {metrics_csv_path}")
metrics_df.to_csv(metrics_csv_path, index=False)
print(f"Saving figure to: {figure_path}")
plot_metrics_bargrid(metrics_df, figure_path)

print(f"Common rows used: {len(merged_df)}")
print(f"Saved metrics CSV: {metrics_csv_path}")
print(f"Saved metrics figure: {figure_path}")
