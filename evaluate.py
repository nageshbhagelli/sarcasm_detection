import torch
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from tqdm import tqdm

def get_predictions(model, data_loader, device, is_fusion=False):
    model.eval()
    predictions = []
    real_values = []

    with torch.no_grad():
        for batch in tqdm(data_loader, desc="Predicting"):
            hl_input_ids = batch['hl_input_ids'].to(device)
            hl_attention_mask = batch['hl_attention_mask'].to(device)
            labels = batch['labels'].to(device)

            if is_fusion:
                ctx_input_ids = batch['ctx_input_ids'].to(device)
                ctx_attention_mask = batch['ctx_attention_mask'].to(device)
                outputs = model(
                    hl_input_ids=hl_input_ids,
                    hl_attention_mask=hl_attention_mask,
                    ctx_input_ids=ctx_input_ids,
                    ctx_attention_mask=ctx_attention_mask
                )
            else:
                outputs = model(
                    input_ids=hl_input_ids,
                    attention_mask=hl_attention_mask
                )

            preds = (outputs > 0.5).float().cpu().numpy()
            predictions.extend(preds)
            real_values.extend(labels.cpu().numpy())

    return np.array(predictions).flatten(), np.array(real_values).flatten()

def compute_metrics(y_true, y_pred):
    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred)
    recall = recall_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)
    
    return {
        "Accuracy": accuracy,
        "Precision": precision,
        "Recall": recall,
        "F1-Score": f1
    }

def plot_confusion_matrix(y_true, y_pred, model_name):
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False,
                xticklabels=['Not Sarcastic', 'Sarcastic'],
                yticklabels=['Not Sarcastic', 'Sarcastic'])
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title(f'Confusion Matrix: {model_name}')
    plt.savefig(f'confusion_matrix_{model_name}.png')
    plt.show()

def get_comparison_table(baseline_metrics, fusion_metrics):
    df_metrics = pd.DataFrame({
        'Model': ['BERT Baseline', 'Fusion Model'],
        'Accuracy': [baseline_metrics['Accuracy'], fusion_metrics['Accuracy']],
        'F1-Score': [baseline_metrics['F1-Score'], fusion_metrics['F1-Score']]
    })
    return df_metrics

if __name__ == "__main__":
    # Test Metrics
    y_true = np.array([0, 1, 0, 1])
    y_pred = np.array([0, 1, 1, 1])
    metrics = compute_metrics(y_true, y_pred)
    print("\nMetrics Sample:", metrics)
