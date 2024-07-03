# from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
# import torch
# import pandas as pd

# def main():
#     # Load the tokenizer and model
#     print("Loading model and tokenizer...")
#     tokenizer = AutoTokenizer.from_pretrained('t5-small')
#     model = AutoModelForSeq2SeqLM.from_pretrained('t5-small')
#     print("Model and tokenizer loaded successfully.")

#     # Load the dataset
#     print("Loading training data...")
#     train_data = pd.read_csv('train_data.csv')
#     print("Training data loaded successfully.")

#     # Ensure the data is in the correct format
#     questions = train_data['question'].astype(str).tolist()
#     answers = train_data['answer'].astype(str).tolist()

#     # Tokenize inputs and outputs separately
#     print("Tokenizing data...")
#     inputs = tokenizer(questions, truncation=True, padding=True, max_length=512, return_tensors='pt')
#     outputs = tokenizer(answers, truncation=True, padding=True, max_length=512, return_tensors='pt')

#     input_ids = inputs['input_ids']
#     attention_mask = inputs['attention_mask']
#     labels = outputs['input_ids']

#     # Replace padding token id's in the labels by -100 so it's ignored by the loss function
#     labels[labels == tokenizer.pad_token_id] = -100

#     # Create a DataLoader
#     print("Creating DataLoader...")
#     train_dataset = torch.utils.data.TensorDataset(input_ids, attention_mask, labels)
#     train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=8, shuffle=True)

#     # Train the model
#     print("Starting training...")
#     model.train()
#     optimizer = torch.optim.AdamW(model.parameters(), lr=5e-5)

#     for epoch in range(3):  # Train for 3 epochs
#         total_loss = 0
#         for batch in train_loader:
#             optimizer.zero_grad()
#             input_ids, attention_mask, labels = batch
#             outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
#             loss = outputs.loss
#             loss.backward()
#             optimizer.step()
#             total_loss += loss.item()
#         print(f"Epoch: {epoch + 1}, Loss: {total_loss / len(train_loader)}")

#     # Save the model
#     print("Saving model...")
#     model.save_pretrained('medical_chatbot_model')
#     tokenizer.save_pretrained('medical_chatbot_tokenizer')
#     print("Model saved successfully.")

# if __name__ == "__main__":
#     main()
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
import torch
import pandas as pd

def main():
    # Load the tokenizer and model
    print("Loading model and tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained('t5-small')
    model = AutoModelForSeq2SeqLM.from_pretrained('t5-small')
    print("Model and tokenizer loaded successfully.")

    # Load the dataset
    print("Loading training data...")
    train_data = pd.read_csv('train_data.csv')
    print("Training data loaded successfully.")

    # Ensure the data is in the correct format
    questions = train_data['question'].astype(str).tolist()
    answers = train_data['answer'].astype(str).tolist()

    # Tokenize inputs and outputs separately
    print("Tokenizing data...")
    inputs = tokenizer(questions, truncation=True, padding=True, max_length=512, return_tensors='pt')
    outputs = tokenizer(answers, truncation=True, padding=True, max_length=512, return_tensors='pt')

    input_ids = inputs['input_ids']
    attention_mask = inputs['attention_mask']
    labels = outputs['input_ids']

    # Replace padding token id's in the labels by -100 so it's ignored by the loss function
    labels[labels == tokenizer.pad_token_id] = -100

    # Create a DataLoader
    print("Creating DataLoader...")
    train_dataset = torch.utils.data.TensorDataset(input_ids, attention_mask, labels)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=8, shuffle=True)

    # Train the model
    print("Starting training...")
    model.train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=5e-5)

    for epoch in range(3):  # Train for 3 epochs
        total_loss = 0
        for batch in train_loader:
            optimizer.zero_grad()
            input_ids, attention_mask, labels = batch
            outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
            loss = outputs.loss
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print(f"Epoch: {epoch + 1}, Loss: {total_loss / len(train_loader)}")

    # Save the model
    print("Saving model...")
    model.save_pretrained('medical_chatbot_model')
    tokenizer.save_pretrained('medical_chatbot_tokenizer')
    print("Model saved successfully.")

if __name__ == "__main__":
    main()
