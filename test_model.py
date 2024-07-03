from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

def main():
    # Load the tokenizer and model
    print("Loading model and tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained('medical_chatbot_tokenizer')
    model = AutoModelForSeq2SeqLM.from_pretrained('medical_chatbot_model')
    print("Model and tokenizer loaded successfully.")

    # Test input
    test_question = "What are the symptoms of Diabetes?"

    # Tokenize the input
    inputs = tokenizer.encode(test_question, return_tensors='pt', truncation=True, padding=True, max_length=512)

    # Generate the response
    outputs = model.generate(
        inputs, 
        max_length=512, 
        num_beams=5, 
        early_stopping=True, 
        no_repeat_ngram_size=2,
        num_return_sequences=1,
        do_sample=True,
        temperature=0.7
    )

    # Decode the response
    answer = tokenizer.decode(outputs[0], skip_special_tokens=True)
    print(f"Question: {test_question}")
    print(f"Answer: {answer}")

if __name__ == "__main__":
    main()
