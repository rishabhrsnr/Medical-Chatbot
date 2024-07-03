import json
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import T5Tokenizer, T5ForConditionalGeneration, Trainer, TrainingArguments

# Load the tokenizer and model
tokenizer_path = "medical_chatbot_tokenizer"
model_path = "medical_chatbot_model"
tokenizer = T5Tokenizer.from_pretrained(tokenizer_path)
model = T5ForConditionalGeneration.from_pretrained(model_path)

# Load the training data
train_data = pd.read_csv('train_data.csv')

# Ensure the data is in the correct format
def tokenize_data(data):
    questions = data['question'].tolist()
    answers = data['answer'].tolist()

    if not all(isinstance(q, str) for q in questions):
        raise ValueError("All questions must be strings")
    if not all(isinstance(a, str) for a in answers):
        raise ValueError("All answers must be strings")

    tokenized_inputs = tokenizer(questions, padding=True, truncation=True, return_tensors="pt")
    tokenized_labels = tokenizer(answers, padding=True, truncation=True, return_tensors="pt")

    return tokenized_inputs, tokenized_labels

train_inputs, train_labels = tokenize_data(train_data)

class MedicalChatbotDataset(Dataset):
    def __init__(self, inputs, labels):
        self.inputs = inputs
        self.labels = labels

    def __len__(self):
        return len(self.inputs["input_ids"])

    def __getitem__(self, idx):
        input_ids = self.inputs["input_ids"][idx]
        attention_mask = self.inputs["attention_mask"][idx]
        labels = self.labels["input_ids"][idx]
        return {"input_ids": input_ids, "attention_mask": attention_mask, "labels": labels}

train_dataset = MedicalChatbotDataset(train_inputs, train_labels)

training_args = TrainingArguments(
    output_dir="./results",
    evaluation_strategy="epoch",
    learning_rate=2e-5,
    per_device_train_batch_size=4,
    per_device_eval_batch_size=4,
    num_train_epochs=3,
    weight_decay=0.01,
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=train_dataset,
)

trainer.train()
