import torch
import torch.nn as nn
from transformers import DistilBertModel

class SarcasmFusionModel(nn.Module):
    def __init__(self, drop_prob=0.3):
        super(SarcasmFusionModel, self).__init__()
        
        # Two separate BERT encoders (as requested)
        self.headline_encoder = DistilBertModel.from_pretrained('distilbert-base-uncased')
        self.context_encoder = DistilBertModel.from_pretrained('distilbert-base-uncased')
        
        hidden_size = self.headline_encoder.config.hidden_size
        
        # Fusion Strategy: cat([h1, h2, |h1 - h2|])
        # Resulting vector size: hidden_size * 3
        self.fusion_size = hidden_size * 3
        
        # Classification Head
        self.fc1 = nn.Linear(self.fusion_size, 256)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(drop_prob)
        self.out = nn.Linear(256, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, hl_input_ids, hl_attention_mask, ctx_input_ids, ctx_attention_mask):
        # Encode headline
        hl_output = self.headline_encoder(
            input_ids=hl_input_ids,
            attention_mask=hl_attention_mask
        )
        h1 = hl_output[0][:, 0, :] # [CLS] token
        
        # Encode context
        ctx_output = self.context_encoder(
            input_ids=ctx_input_ids,
            attention_mask=ctx_attention_mask
        )
        h2 = ctx_output[0][:, 0, :] # [CLS] token
        
        # Fusion: [h1; h2; |h1 - h2|]
        diff = torch.abs(h1 - h2)
        fusion_vector = torch.cat([h1, h2, diff], dim=1)
        
        # Pass through classification head
        x = self.fc1(fusion_vector)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.out(x)
        
        return self.sigmoid(x)

if __name__ == "__main__":
    # Test fusion model initialization
    model = SarcasmFusionModel()
    print("Fusion Model Initialized.")
    
    # Dummy forward pass
    hl_input = torch.randint(0, 30522, (2, 64))
    hl_mask = torch.ones((2, 64))
    ctx_input = torch.randint(0, 30522, (2, 128))
    ctx_mask = torch.ones((2, 128))
    
    out = model(hl_input, hl_mask, ctx_input, ctx_mask)
    print(f"Fusion Output shape: {out.shape}")
