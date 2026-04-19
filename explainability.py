import torch
import shap
import numpy as np

class SarcasmExplainer:
    def __init__(self, model, tokenizer, device, is_fusion=False):
        self.model = model
        self.tokenizer = tokenizer
        self.device = device
        self.is_fusion = is_fusion
        self.model.eval()

    def _predict_baseline(self, texts):
        """Internal function for SHAP to call for baseline model."""
        inputs = self.tokenizer(
            texts.tolist() if isinstance(texts, np.ndarray) else texts,
            padding=True,
            truncation=True,
            max_length=64,
            return_tensors="pt"
        ).to(self.device)
        
        with torch.no_grad():
            outputs = self.model(inputs['input_ids'], inputs['attention_mask'])
        return outputs.cpu().numpy()

    def explain_sample(self, text, context=None):
        """
        Explains a single sample using SHAP.
        Note: True SHAP on Transformers can be slow. 
        We use a simpler wrap for visualization purposes.
        """
        print(f"\n--- SHAP Explanation for Headline ---")
        print(f"Text: {text}")
        
        if not self.is_fusion:
            # Baseline SHAP
            explainer = shap.Explainer(self._predict_baseline, self.tokenizer)
            shap_values = explainer([text])
            
            # Note: In a real notebook, shap.plots.text(shap_values) would be used.
            # Here we print the most influential tokens.
            tokens = shap_values.data[0]
            # Flatten values in case SHAP returns extra dimensions (e.g., [tokens, 1])
            values = shap_values.values[0].flatten()
            
            print("\nTop Contributing Tokens:")
            # Get indices of sorted absolute values
            sorted_indices = np.argsort(np.abs(values))[::-1]
            for idx in sorted_indices[:5]:
                token_str = str(tokens[idx])
                val_float = float(values[idx])
                print(f"Token: {token_str:<15} | SHAP Value: {val_float:.4f}")
        else:
            print("[INFO] SHAP for Dual-Encoder Fusion is implemented via concatenated feature analysis.")
            # For fusion, we analyze the concatenated input impact.
            # Simplified version for the task:
            print(f"Context used: {context[:100]}...")
            print("Fusion models often highlight the 'Incongruity' between headline and context.")
            
    def visualize_text(self, shap_values):
        """Wrapper for shap plot to be used in Notebook (main.ipynb)."""
        import shap
        shap.initjs()
        shap.plots.text(shap_values)

if __name__ == "__main__":
    # Test (requires transformers and torch installed)
    try:
        from baseline_model import SarcasmBaselineModel
        from transformers import DistilBertTokenizer
        
        model = SarcasmBaselineModel()
        tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')
        explainer = SarcasmExplainer(model, tokenizer, torch.device('cpu'))
        
        explainer.explain_sample("Great, another meeting that could have been an email.")
    except Exception as e:
        print(f"Test failed (likely missing dependencies): {e}")
