import torch
import argparse
from transformers import DistilBertTokenizer
from data_loader import SarcasmDataLoader
from preprocessing import prepare_loaders
from baseline_model import SarcasmBaselineModel
from fusion_model import SarcasmFusionModel
from train import run_training
from evaluate import get_predictions, compute_metrics, plot_confusion_matrix, get_comparison_table
from explainability import SarcasmExplainer
import pandas as pd

def parse_args():
    parser = argparse.ArgumentParser(description="Run Sarcasm Detection Pipeline")
    parser.add_argument("--data_path", type=str, default="Sarcasm_Headlines_Dataset.json", help="Path to the dataset JSON file")
    parser.add_argument("--subset_size", type=int, default=None, help="Number of samples to use (None for all)")
    parser.add_argument("--scrape_limit", type=int, default=50, help="Number of articles to scrape context for")
    parser.add_argument("--batch_size", type=int, default=16, help="Batch size for training")
    parser.add_argument("--epochs", type=int, default=3, help="Number of training epochs")
    return parser.parse_args()

def main():
    args = parse_args()
    
    # 1. Configuration
    FILE_PATH = args.data_path
    SUBSET_SIZE = args.subset_size
    SCRAPE_LIMIT = args.scrape_limit
    BATCH_SIZE = args.batch_size
    EPOCHS = args.epochs
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    print(f"Using Device: {DEVICE}")

    # 2. Data Loading & Context Generation
    loader = SarcasmDataLoader(FILE_PATH, subset_size=SUBSET_SIZE)
    df = loader.load_data()
    df = loader.add_context(scrape_limit=SCRAPE_LIMIT)
    
    # 3. Preprocessing
    tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')
    train_loader, val_loader, test_loader = prepare_loaders(df, tokenizer, batch_size=BATCH_SIZE)

    # 4. Phase 1: Baseline Model (Headline only)
    print("\n--- Training Baseline Model ---")
    baseline_model = SarcasmBaselineModel().to(DEVICE)
    optimizer_b = torch.optim.AdamW(baseline_model.parameters(), lr=2e-5)
    
    run_training(
        baseline_model, train_loader, val_loader, optimizer_b, 
        DEVICE, epochs=EPOCHS, model_name="baseline", is_fusion=False
    )

    # 5. Phase 2: Fusion Model (Headline + Context)
    print("\n--- Training Fusion Model ---")
    fusion_model = SarcasmFusionModel().to(DEVICE)
    optimizer_f = torch.optim.AdamW(fusion_model.parameters(), lr=2e-5)
    
    run_training(
        fusion_model, train_loader, val_loader, optimizer_f, 
        DEVICE, epochs=EPOCHS, model_name="fusion", is_fusion=True
    )

    # 6. Evaluation
    print("\n--- Evaluating Models ---")
    
    # Baseline
    y_pred_b, y_true = get_predictions(baseline_model, test_loader, DEVICE, is_fusion=False)
    metrics_b = compute_metrics(y_true, y_pred_b)
    
    # Fusion
    y_pred_f, _ = get_predictions(fusion_model, test_loader, DEVICE, is_fusion=True)
    metrics_f = compute_metrics(y_true, y_pred_f)
    
    # Combined Results
    comparison_df = get_comparison_table(metrics_b, metrics_f)
    print("\nBenchmark Results:")
    print(comparison_df.to_string(index=False))
    
    # 7. Explainability Sample
    print("\n--- Sample Explanation ---")
    explainer = SarcasmExplainer(baseline_model, tokenizer, DEVICE)
    sample_text = df.iloc[0]['headline']
    explainer.explain_sample(sample_text)

if __name__ == "__main__":
    main()
