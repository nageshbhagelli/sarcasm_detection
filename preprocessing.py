import torch
from torch.utils.data import Dataset, DataLoader
from transformers import DistilBertTokenizer
from sklearn.model_selection import train_test_split

class SarcasmDataset(Dataset):
    def __init__(self, headlines, contexts, labels, tokenizer, hl_max_len=64, ctx_max_len=128):
        self.headlines = headlines
        self.contexts = contexts
        self.labels = labels
        self.tokenizer = tokenizer
        self.hl_max_len = hl_max_len
        self.ctx_max_len = ctx_max_len

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, item):
        headline = str(self.headlines[item])
        context = str(self.contexts[item])
        label = self.labels[item]

        # Tokenize headline
        hl_encoding = self.tokenizer(
            headline,
            add_special_tokens=True,
            max_length=self.hl_max_len,
            padding="max_length",
            truncation=True,
            return_attention_mask=True,
            return_tensors='pt',
        )

        # Tokenize context
        ctx_encoding = self.tokenizer(
            context,
            add_special_tokens=True,
            max_length=self.ctx_max_len,
            padding="max_length",
            truncation=True,
            return_attention_mask=True,
            return_tensors='pt',
        )

        return {
            'headline_text': headline,
            'hl_input_ids': hl_encoding['input_ids'].flatten(),
            'hl_attention_mask': hl_encoding['attention_mask'].flatten(),
            'ctx_input_ids': ctx_encoding['input_ids'].flatten(),
            'ctx_attention_mask': ctx_encoding['attention_mask'].flatten(),
            'labels': torch.tensor(label, dtype=torch.float)
        }

def prepare_loaders(df, tokenizer, batch_size=16, hl_max_len=64, ctx_max_len=128):
    """Splits data and returns Train, Val, and Test DataLoaders."""
    
    # Stratified split: 80% Train, 20% Temp
    df_train, df_temp = train_test_split(
        df, test_size=0.2, random_state=42, stratify=df['is_sarcastic']
    )
    
    # 50% of Temp (10% of total) for Val, 50% for Test
    df_val, df_test = train_test_split(
        df_temp, test_size=0.5, random_state=42, stratify=df_temp['is_sarcastic']
    )
    
    print(f"Split counts: Train={len(df_train)}, Val={len(df_val)}, Test={len(df_test)}")

    def create_data_loader(data_df):
        ds = SarcasmDataset(
            headlines=data_df.headline.to_numpy(),
            contexts=data_df.context.to_numpy(),
            labels=data_df.is_sarcastic.to_numpy(),
            tokenizer=tokenizer,
            hl_max_len=hl_max_len,
            ctx_max_len=ctx_max_len
        )
        return DataLoader(ds, batch_size=batch_size, num_workers=0) # num_workers=0 for stability on windows/CPU

    return (
        create_data_loader(df_train),
        create_data_loader(df_val),
        create_data_loader(df_test)
    )

if __name__ == "__main__":
    # Test Preprocessing
    import pandas as pd
    tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')
    test_df = pd.DataFrame({
        'headline': ['this is a headline', 'and another one'],
        'context': ['context content goes here', 'more context'],
        'is_sarcastic': [0, 1]
    })
    
    train_loader, val_loader, test_loader = prepare_loaders(test_df, tokenizer, batch_size=1)
    sample_batch = next(iter(train_loader))
    print("\nSample Batch Keys:", sample_batch.keys())
    print("HL ID Shape:", sample_batch['hl_input_ids'].shape)
    print("Label:", sample_batch['labels'])
