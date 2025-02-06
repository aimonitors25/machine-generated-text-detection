from torch.utils.data import DataLoader
from transformers import AdamW, get_linear_schedule_with_warmup
from datasets import Dataset
import torch
import json
from tqdm.auto import tqdm
from transformers import AutoModelForSequenceClassification, AutoTokenizer
import pandas as pd
from torch import nn
from sklearn.metrics import f1_score
import numpy as np

with open("config.json", "r") as config_file:
    config = json.load(config_file)

# Extract model and data paths
model_path = config["model"]["base_model_path"]
train_data_path = config["data"]["train_dataset"]
dev_data_path = config["data"]["dev_dataset"]
test_data_path = config["data"]["test_dataset"]
out_model_dir = config["output"]["finetuned_model_dir"]


# Extract training arguments
batch_size = config["training"]["batch_size"]
num_epochs = config["training"]["num_epochs"]
learning_rate = config["training"]["learning_rate"]
weight_decay = config["training"]["weight_decay"]
warmup_ratio = config["training"]["warmup_ratio"]
gradient_clipping = config["training"]["gradient_clipping"]


# Load the tokenizer and model for binary classification (2 labels: human vs machine-generated text)
tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForSequenceClassification.from_pretrained(model_path, num_labels=2)

# Load datasets
# `train_dfcom` and `dev_dfcom` are subsamples of train and validation datasets
# Modify the file paths accordingly

train_df = pd.read_parquet(train_data_path)
dev_df = pd.read_parquet(dev_data_path)
test = pd.read_parquet(test_data_path)

# Use a subset of the data for training and development
train_dfcom = train_df[:10000]
dev_dfcom = dev_df[:5000]

train_df = train_dfcom
dev_df = dev_dfcom

# Print class distribution for sanity check
print("-----")
print(train_df["label"].value_counts())
print(dev_df["label"].value_counts())
print(test["label"].value_counts())
print("-----")


def preprocess_function(examples):
    """
    Tokenizes input text and processes labels for training.
    """
    tokenized = tokenizer(
        examples['text'],
        truncation=True,
        padding='max_length',
        max_length=128,
        return_tensors=None  # Returns as lists instead of tensors
    )
    tokenized['labels'] = examples['label']  # Attach labels
    return tokenized


def train_model(model, train_dataloader, optimizer, scheduler, device, num_epochs=3):
    """
    Fine-tunes the model using the given training data.
    Implements early stopping based on loss improvement.
    """
    model.to(device)
    model.train()
    best_loss = float('inf')
    progress_bar = tqdm(range(num_epochs * len(train_dataloader)))
    patience = 3
    patience_counter = 0

    for epoch in range(num_epochs):
        total_loss = 0
        num_batches = 0
        
        for batch in train_dataloader:
            inputs = {
                'input_ids': batch['input_ids'].to(device),
                'attention_mask': batch['attention_mask'].to(device),
                'labels': batch['labels'].to(device)
            }
            
            optimizer.zero_grad()
            outputs = model(**inputs)
            
            # Assign weights to handle class imbalance (assuming 0 = human, 1 = machine-generated)
            class_weights = torch.tensor([2.0, 1.0]).to(device)
            criterion = nn.CrossEntropyLoss(weight=class_weights)
            loss = criterion(outputs.logits.to(device), batch['labels'].to(device))
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)  # Prevent exploding gradients
            optimizer.step()
            scheduler.step()
            
            total_loss += loss.item()
            num_batches += 1
            
            progress_bar.update(1)
            progress_bar.set_description(f"Epoch {epoch+1} - Loss: {loss.item():.4f} - LR: {scheduler.get_last_lr()[0]:.2e}")
            
            del outputs, loss  # Free memory
            torch.cuda.empty_cache()
        
        avg_loss = total_loss / num_batches
        print(f'\nEpoch {epoch+1}/{num_epochs} completed. Average Loss: {avg_loss:.4f}')
        
        # Early stopping mechanism
        if avg_loss < best_loss:
            best_loss = avg_loss
            patience_counter = 0
            torch.save(model.state_dict(), f"{out_model_dir}/Roberta_tuned_model_{epoch}.pt")  # Save best model
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print("Early stopping triggered!")
                break
    
    return model


def evaluate_model(model, test_dataloader, device):
    """
    Evaluates the trained model on test data.
    """
    model.eval()
    total_correct = 0
    total_samples = 0
    true_labels, predicted_labels = [], []
    
    with torch.no_grad():
        for batch in tqdm(test_dataloader, desc="Evaluating"):
            inputs = {
                'input_ids': batch['input_ids'].to(device),
                'attention_mask': batch['attention_mask'].to(device),
            }
            labels = batch['labels'].to(device)
            
            true_labels.append(labels.cpu().numpy())
            outputs = model(**inputs)
            predictions = torch.argmax(outputs.logits, dim=-1)
            predicted_labels.append(predictions.cpu().numpy())
            total_correct += (predictions == labels).sum().item()
            total_samples += labels.size(0)
    
    accuracy = total_correct / total_samples
    print(f"\nTest Accuracy: {accuracy:.4f}")
    
    mod_true_labels = np.concatenate(true_labels)
    mod_pred_labels = np.concatenate(predicted_labels)
    f1 = f1_score(mod_true_labels, mod_pred_labels, average='weighted')
    print(f"\nF1: {f1:.4f}")
    
    return accuracy

# Prepare datasets
train_dataset = Dataset.from_pandas(train_df)
test_dataset = Dataset.from_pandas(dev_df)

# Tokenize datasets
tokenized_train = train_dataset.map(preprocess_function, batched=True, remove_columns=train_dataset.column_names)
tokenized_test = test_dataset.map(preprocess_function, batched=True, remove_columns=test_dataset.column_names)

# Convert tokenized datasets to PyTorch format
tokenized_train.set_format('torch', columns=['input_ids', 'attention_mask', 'labels'])
tokenized_test.set_format('torch', columns=['input_ids', 'attention_mask', 'labels'])

# Create DataLoaders
train_dataloader = DataLoader(tokenized_train, batch_size, shuffle=True)
test_dataloader = DataLoader(tokenized_test, batch_size)

# Setup device and optimizer
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
optimizer = AdamW(model.parameters(),learning_rate, weight_decay=0.01)

# Scheduler with warmup steps
num_training_steps = len(train_dataloader) * 3  # 3 epochs
num_warmup_steps = num_training_steps // 10
scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=num_warmup_steps, num_training_steps=num_training_steps)

# Train the model
model = train_model(model, train_dataloader, optimizer, scheduler, device, num_epochs=3)

# Evaluate the model
accuracy = evaluate_model(model, test_dataloader, device)
