# from flask import Flask, render_template, request, jsonify
# from transformers import T5Tokenizer, T5ForConditionalGeneration
# import json

# app = Flask(__name__)

# # Load the tokenizer and model
# tokenizer_path = "medical_chatbot_tokenizer"
# model_path = "medical_chatbot_model"
# tokenizer = T5Tokenizer.from_pretrained(tokenizer_path)
# model = T5ForConditionalGeneration.from_pretrained(model_path)

# # Load the medical data
# with open('health-data.json', 'r') as f:
#     medical_data = json.load(f)

# @app.route('/')
# def home():
#     return render_template('index.html')

# @app.route('/chat', methods=['POST'])
# def chat():
#     data = request.get_json()
#     question = data.get("question")
#     if not question:
#         return jsonify({"answer": "Please ask a valid question."})

#     inputs = tokenizer.encode("question: " + question + " </s>", return_tensors="pt")
#     outputs = model.generate(inputs, max_length=512, num_beams=4, early_stopping=True)
#     predicted_answer = tokenizer.decode(outputs[0], skip_special_tokens=True)

#     # Extract symptoms related to the predicted answer
#     for disease in medical_data["diseases"]:
#         if predicted_answer.lower() in disease["name"].lower():
#             symptoms = ", ".join(disease["symptoms"])
#             return jsonify({"answer": f"{predicted_answer}: {symptoms}"})

#     return jsonify({"answer": "Sorry, I don't have information on that topic."})

# if __name__ == '__main__':
#     app.run(host='0.0.0.0', port=5000)
from flask import Flask, render_template, request, jsonify
from transformers import T5Tokenizer, T5ForConditionalGeneration
import json

app = Flask(__name__)

# Load the tokenizer and model
tokenizer_path = "medical_chatbot_tokenizer"
model_path = "medical_chatbot_model"
tokenizer = T5Tokenizer.from_pretrained(tokenizer_path)
model = T5ForConditionalGeneration.from_pretrained(model_path)

# Load the medical data
with open('health-data.json', 'r') as f:
    medical_data = json.load(f)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/chat', methods=['POST'])
def chat():
    data = request.get_json()
    question = data.get("question")
    if not question:
        return jsonify({"answer": "Please ask a valid question."})

    # Ensure there are no existing </s> tokens in the question
    question = question.replace('</s>', '')

    inputs = tokenizer.encode("question: " + question, return_tensors="pt")
    outputs = model.generate(inputs, max_length=512, num_beams=4, early_stopping=True)
    predicted_answer = tokenizer.decode(outputs[0], skip_special_tokens=True)

    # Extract symptoms related to the predicted answer
    for disease in medical_data["diseases"]:
        if predicted_answer.lower() in disease["name"].lower():
            symptoms = ", ".join(disease["symptoms"])
            return jsonify({"answer": f"{predicted_answer}: {symptoms}"})

    return jsonify({"answer": "Sorry, I don't have information on that topic."})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
