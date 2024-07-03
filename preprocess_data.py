
# import pandas as pd
# import json
# from sklearn.model_selection import train_test_split

# # Load the dataset
# print("Loading dataset...")
# with open('health-data.json') as f:
#     data = json.load(f)
# print("Dataset loaded successfully.")

# # Convert the data to a pandas DataFrame
# diseases = data['diseases']
# print(f"Number of diseases loaded: {len(diseases)}")

# # Prepare data for training
# questions = []
# answers = []

# for disease in diseases:
#     description = disease['description']
#     symptoms = disease['symptoms']
#     treatments = disease['treatments']
    
#     for symptom in symptoms:
#         questions.append(f"What are the symptoms of {disease['name']}?")
#         answers.append(", ".join(symptoms))
    
#     for treatment in treatments:
#         questions.append(f"What are the treatments for {disease['name']}?")
#         answers.append(", ".join(treatments))
    
#     questions.append(f"What is {disease['name']}?")
#     answers.append(description)

# # Create a DataFrame
# df = pd.DataFrame({'question': questions, 'answer': answers})

# # Print some of the data to ensure it's correct
# print("Sample data:")
# print(df.head())

# # Preprocess the data
# df['question'] = df['question'].apply(lambda x: x.lower())

# # Split the data
# train_data, test_data = train_test_split(df, test_size=0.2, random_state=42)

# # Save the preprocessed data
# print("Saving train_data.csv...")
# train_data.to_csv('train_data.csv', index=False)
# print("train_data.csv saved successfully.")

# print("Saving test_data.csv...")
# test_data.to_csv('test_data.csv', index=False)
# print("test_data.csv saved successfully.")

# print("Preprocessing complete. Files saved as 'train_data.csv' and 'test_data.csv'.")
import pandas as pd
import json
from sklearn.model_selection import train_test_split

# Load the dataset
print("Loading dataset...")
with open('health-data.json') as f:
    data = json.load(f)
print("Dataset loaded successfully.")

# Convert the data to a pandas DataFrame
diseases = data['diseases']
print(f"Number of diseases loaded: {len(diseases)}")

# Prepare data for training
questions = []
answers = []

for disease in diseases:
    description = disease['description']
    symptoms = disease['symptoms']
    treatments = disease['treatments']
    
    for symptom in symptoms:
        questions.append(f"What are the symptoms of {disease['name']}?")
        answers.append(", ".join(symptoms))
    
    for treatment in treatments:
        questions.append(f"What are the treatments for {disease['name']}?")
        answers.append(", ".join(treatments))
    
    questions.append(f"What is {disease['name']}?")
    answers.append(description)

# Create a DataFrame
df = pd.DataFrame({'question': questions, 'answer': answers})

# Print some of the data to ensure it's correct
print("Sample data:")
print(df.head())

# Preprocess the data
df['question'] = df['question'].apply(lambda x: x.lower())

# Split the data
train_data, test_data = train_test_split(df, test_size=0.2, random_state=42)

# Save the preprocessed data
print("Saving train_data.csv...")
train_data.to_csv('train_data.csv', index=False)
print("train_data.csv saved successfully.")

print("Saving test_data.csv...")
test_data.to_csv('test_data.csv', index=False)
print("test_data.csv saved successfully.")

print("Preprocessing complete. Files saved as 'train_data.csv' and 'test_data.csv'.")
