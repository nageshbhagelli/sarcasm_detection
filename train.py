import torch
import torch.nn as nn
from tqdm import tqdm
import numpy as np
import os

def train_epoch(model, data_loader, optimizer, device, is_fusion=False):
    model.train()
    losses = []
    correct_predictions = 0
    
    criterion = nn.BCELoss() # Using BCELoss because models have sigmoid at the end

    for batch in tqdm(data_loader, desc="Training"):
        hl_input_ids = batch['hl_input_ids'].to(device)
        hl_attention_mask = batch['hl_attention_mask'].to(device)
        labels = batch['labels'].to(device).unsqueeze(1)
        
        optimizer.zero_grad()
        
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
        
        loss = criterion(outputs, labels)
        preds = (outputs > 0.5).float()
        
        correct_predictions += torch.sum(preds == labels)
        losses.append(loss.item())
        
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0) # Gradient clipping
        optimizer.step()
        
    return correct_predictions.double() / len(data_loader.dataset), np.mean(losses)

def eval_model(model, data_loader, device, is_fusion=False):
    model.eval()
    losses = []
    correct_predictions = 0
    criterion = nn.BCELoss()

    with torch.no_grad():
        for batch in tqdm(data_loader, desc="Evaluating"):
            hl_input_ids = batch['hl_input_ids'].to(device)
            hl_attention_mask = batch['hl_attention_mask'].to(device)
            labels = batch['labels'].to(device).unsqueeze(1)

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

            loss = criterion(outputs, labels)
            preds = (outputs > 0.5).float()
            
            correct_predictions += torch.sum(preds == labels)
            losses.append(loss.item())

    return correct_predictions.double() / len(data_loader.dataset), np.mean(losses)

def run_training(model, train_loader, val_loader, optimizer, device, epochs=3, model_name="model", is_fusion=False):
    best_accuracy = 0
    patience = 2
    counter = 0

    for epoch in range(epochs):
        print(f"\nEpoch {epoch + 1}/{epochs}")
        print("-" * 10)

        train_acc, train_loss = train_epoch(model, train_loader, optimizer, device, is_fusion)
        print(f"Train Loss: {train_loss:.4f} accuracy: {train_acc:.4f}")

        val_acc, val_loss = eval_model(model, val_loader, device, is_fusion)
        print(f"Val Loss: {val_loss:.4f} accuracy: {val_acc:.4f}")

        if val_acc > best_accuracy:
            torch.save(model.state_dict(), f'best_{model_name}.bin')
            best_accuracy = val_acc
            counter = 0
        else:
            counter += 1
            if counter >= patience:
                print("Early stopping triggered.")
                break
                
    print(f"Best Val Accuracy: {best_accuracy:.4f}")
    return best_accuracy
