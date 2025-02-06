from torch.utils.data import DataLoader
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from datasets import Dataset
import torch
import json
from tqdm.auto import tqdm
import pandas as pd
import numpy as np
from sklearn.metrics import f1_score

with open("config.json", "r") as config_file:
    config = json.load(config_file)
    
    
# Extract model and data paths
model_path = config["model"]["base_model_path"]
test_data_path = config["data"]["test_dataset"]
out_model_dir = config["output"]["finetuned_model_dir"]
out_model_file = config["output"]["finetuned_model_name"]
output_test_file = config["output"]["output_test_file"]

# Load the tokenizer and model for inference
tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForSequenceClassification.from_pretrained(model_path, num_labels=2)

# Load and preprocess the test dataset
test = pd.read_parquet(test_data_path)
test_dfcom = test[:1000]
test = test_dfcom

print("-----")
print(test["label"].value_counts())
print("-----")

def preprocess_function(examples):
    """
    Tokenizes input text and processes labels.
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


def evaluate_model(model, test_dataloader, device, mode="machine"):
    """
    Evaluates the trained model on test data.
    """
    model.to(device)
    model.eval()
    total_correct = 0
    total_samples = 0
    
    all_outputs = []
    all_labels = []
    true_labels, predicted_labels = [], []
    
    with torch.no_grad():
        for batch in tqdm(test_dataloader, desc="Evaluating"):
            inputs = {
                'input_ids': batch['input_ids'].to(device),
                'attention_mask': batch['attention_mask'].to(device),
            }
            labels = batch['labels'].to(device)
            
            outputs = model(**inputs)
            predictions = torch.argmax(outputs.logits, dim=-1)
            
            true_labels.append(labels.cpu().numpy())
            predicted_labels.append(predictions.cpu().numpy())
            
            total_correct += (predictions == labels).sum().item()
            total_samples += labels.size(0)
            
            all_outputs.extend(predictions.cpu().numpy())  # Store model output
            all_labels.extend(labels.cpu().numpy())  # Store labels
    
    # Create and save results DataFrame
    results_df = pd.DataFrame({
        'Output': all_outputs,
        'Label': all_labels
    })
    results_df.to_csv(f"{output_test_file}")
    
    accuracy = total_correct / total_samples
    print(f"\nTest Accuracy: {accuracy:.4f}")
    
    mod_true_labels = np.concatenate(true_labels)
    mod_pred_labels = np.concatenate(predicted_labels)
    f1 = f1_score(mod_true_labels, mod_pred_labels, average='weighted')
    print(f"\nF1: {f1:.4f}")
    
    return accuracy

# Prepare and tokenize dataset
test_dataset = Dataset.from_pandas(test)
tokenized_test = test_dataset.map(
    preprocess_function,
    batched=True,
    remove_columns=test_dataset.column_names
)

tokenized_test.set_format('torch', columns=['input_ids', 'attention_mask', 'labels'])

test_dataloader = DataLoader(
    tokenized_test,
    batch_size=1024
)

# Setup device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load pre-trained model weights
model.load_state_dict(torch.load(out_model_dir+"/"+out_model_file))

# Evaluate the model
accuracy = evaluate_model(model, test_dataloader, device, "machine")
