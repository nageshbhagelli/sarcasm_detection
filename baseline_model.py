import torch
import torch.nn as nn
from transformers import DistilBertModel

class SarcasmBaselineModel(nn.Module):
    def __init__(self, drop_prob=0.3):
        super(SarcasmBaselineModel, self).__init__()
        
        # Load pre-trained DistilBERT
        self.distilbert = DistilBertModel.from_pretrained('distilbert-base-uncased')
        
        # Dropout layer
        self.dropout = nn.Dropout(drop_prob)
        
        # Classification head: Bert output is 768
        self.out = nn.Linear(self.distilbert.config.hidden_size, 1)
        
        # Sigmoid is usually handled by BCEWithLogitsLoss during training, 
        # but the user requested a sigmoid output specifically.
        self.sigmoid = nn.Sigmoid()

    def forward(self, input_ids, attention_mask):
        # DistilBERT returns (last_hidden_state,)
        # last_hidden_state shape: [batch_size, sequence_length, 768]
        distilbert_output = self.distilbert(
            input_ids=input_ids,
            attention_mask=attention_mask
        )
        
        # We use the [CLS] token representation (first token)
        hidden_state = distilbert_output[0]
        cls_token_representation = hidden_state[:, 0, :]
        
        # Apply dropout and linear layer
        output = self.dropout(cls_token_representation)
        output = self.out(output)
        
        return self.sigmoid(output)

if __name__ == "__main__":
    # Test model initialization
    model = SarcasmBaselineModel()
    print("Baseline Model Initialized.")
    print(f"Hidden size: {model.distilbert.config.hidden_size}")
    
    # Dummy forward pass
    dummy_input = torch.randint(0, 30522, (2, 64))
    dummy_mask = torch.ones((2, 64))
    out = model(dummy_input, dummy_mask)
    print(f"Output shape: {out.shape}")
