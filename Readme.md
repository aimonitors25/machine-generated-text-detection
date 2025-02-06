# RoBERTa Fine-Tuning for Machine-Generated Text Detection

This repository contains scripts for training and evaluating a RoBERTa-based model to classify whether a given text is **machine-generated** or **human-written**.

## **Project Structure**

```
├── config.json            # Configuration file specifying model paths, dataset paths, and training parameters
├── train.py               # Script for fine-tuning RoBERTa on the classification task
├── inference.py           # Script for running inference using the fine-tuned model
├── README.md              # Documentation
├── requirements.txt       # List of dependencies
```

## **Setup Instructions**

### **1. Install Dependencies**
Ensure you have Python 3.7+ installed. Then install the required libraries:

```bash
pip install -r requirements.txt
```

### **2. Configure Paths**
Modify `config.json` to set the correct paths for:
- The pre-trained RoBERTa model
- The training, validation, and test datasets
- The directory to save fine-tuned models

Example `config.json`:

```json
{
  "model": {
    "base_model_path": "roberta-large/",
    "num_labels": 2
  },
  "data": {
    "train_dataset": "path/to/train_data.parquet",
    "dev_dataset": "path/to/dev_data.parquet",
    "test_dataset": "path/to/test_data.jsonl"
  },
  "training": {
    "batch_size": 64,
    "num_epochs": 3,
    "learning_rate": 2e-5,
    "weight_decay": 0.01,
    "warmup_ratio": 0.1,
    "gradient_clipping": 1.0
  },
  "output": {
    "finetuned_model_dir": "path/to/save/finetuned_models/",
     "finetuned_model_name":"path/to/save/output_file",
     "output_test_file":"path/to/save/output_csvfile"
  }
}
```

### **3. Train the Model**
To start fine-tuning RoBERTa on the dataset, run:

```bash
python train.py
```

The model will be trained for the specified number of epochs, and the best-performing model will be saved in the output directory specified in `config.json`.

### **4. Run Inference**
To test the trained model on new data, run:

```bash
python inference.py
```

The script will load the fine-tuned model and evaluate it on the test dataset.

## **Evaluation Metrics**
The script reports:
- **Accuracy**: Measures how many texts were correctly classified.
- **F1 Score**: A weighted metric that balances precision and recall.

## **Model and Data**
- **Pre-trained Model**: [RoBERTa](https://huggingface.co/roberta-large) fine-tuned on machine-generated vs. human-written text classification.
- **Dataset**: The dataset should contain `text` and `label` columns, where:
  - `label = 0` → Human-written
  - `label = 1` → Machine-generated

## **References**
This work is inspired by the research presented in:
- AI-Monitors at GenAI Detection Task 1: Fast and Scalable Machine Generated Text Detection. Published in ACL Anthology Coling Workshop 2025. [Paper Link](https://aclanthology.org/2025.genaidetect-1.25/)

## **License**
This project is licensed under the MIT License.
