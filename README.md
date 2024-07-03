# Medical-Chatbot

This project is a Medical Chatbot designed to provide medical information and answer questions about various health-related topics. It leverages advanced NLP techniques using the T5 model from Hugging Face Transformers.

## Project Overview

The Medical Chatbot uses a healthcare dataset containing information on 275 diseases. The chatbot is designed to answer questions about symptoms and treatments of various medical conditions.

## Objectives

- Provide accurate and quick answers to health-related questions.
- Cover a wide range of diseases and medical conditions.
- Deliver detailed symptoms and treatment information.
- Utilize advanced NLP techniques for better user interaction.

## Dataset Description

The dataset contains information on 275 diseases, including the following details:
- Disease Name
- Description
- Symptoms
- Treatments

The dataset is sourced from [Kaggle](https://www.kaggle.com/datasets/rudravpatel/healthcare-dataset-for-chatbot).

## Analysis

### Model Training
- Preprocessed the dataset to create training and testing datasets.
- Fine-tuned the T5 model using the processed dataset.
- Evaluated the model's performance and made necessary adjustments.

### Example Questions

- What are the symptoms of Eczema?
- What are the symptoms of Diabetes?
- What are the symptoms of the flu?

## Results

- The chatbot can accurately answer a wide range of health-related questions.
- It provides detailed information on symptoms and treatments of various diseases.
- The chatbot's responses are based on the trained dataset, ensuring reliable and relevant answers.

## Installation and Usage

1. **Clone the Repository:**
2. **Install Dependencies:**
3. **Preprocess the Data:**
4. **Fine-Tune the Model:** 
5. **Run the Flask App:**
6. **Open your browser and navigate to:**

## Project Structure
Medical-Chatbot/
├── medical_chatbot_model/
│ ├── config.json
│ ├── generation_config.json
│ ├── pytorch_model.bin
│ └── ...
├── medical_chatbot_tokenizer/
│ ├── special_tokens_map.json
│ ├── spiece.model
│ ├── tokenizer_config.json
│ └── ...
├── static/
│ ├── style.css
│ └── script.js
├── templates/
│ └── index.html
├── app.py
├── preprocess_data.py
├── fine_tune_model.py
├── health-data.json
├── requirements.txt
└── Dockerfile


## Contact Information

For any questions or feedback, please reach out to me at:

- **Email:** rishabh.rsnr@gmail.com
- **LinkedIn:** [Rishabh Bansal](https://www.linkedin.com/in/rishabhrsnr)

## Acknowledgements

- The dataset used in this project is sourced from [Kaggle](https://www.kaggle.com/datasets/rudravpatel/healthcare-dataset-for-chatbot).
- Thanks to Hugging Face for their amazing [Transformers library](https://huggingface.co/transformers/).

## Author

Rishabh Bansal

Feel free to reach out with any questions or feedback!



