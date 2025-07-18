from flask import Flask, request, jsonify
from flask_cors import CORS
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
import pickle
from tensorflow.keras.models import model_from_json
import re
import openai

openai.api_key = ""

# Load models and data
model = load_model("D:\\PROJECT\\cyber_bullyingvi\\cyber_bullyingvi\\cyber_bullying\\BI LSTM\\models\\model.keras") 
with open("D:\\PROJECT\\cyber_bullyingvi\\cyber_bullyingvi\\cyber_bullying\\BI LSTM\\models\\model.pkl", 'rb') as f:
    loaded_pickle_data = pickle.load(f)

model = model_from_json(loaded_pickle_data['model'])
model.set_weights(loaded_pickle_data['weights'])

tokenizer = loaded_pickle_data['tokenizer']
label_encoder = loaded_pickle_data['label_encoder']

MAX_SEQUENCE_LENGTH = 100  

def clean_text(text):
    text = re.sub(r"http\S+|www\S+", "", text)  # Remove URLs
    text = re.sub(r"@\w+", "", text)  # Remove mentions
    text = text.lower()  # Convert to lowercase
    text = re.sub(r"[^a-zA-Z0-9\s]", "", text)  # Remove non-alphanumeric characters
    text = re.sub(r"\s+", " ", text).strip()  # Remove extra spaces
    return text

# Create a Flask app
app = Flask(_name_)
CORS(app)

@app.route('/analyze', methods=['POST'])
def analyze_chat():
    data = request.get_json()
    text = data.get('text', None)
    text= clean_text(text)



    if text is None:
        return jsonify({"error": "No text provided"}), 400

    sequences = tokenizer.texts_to_sequences([text])
    padded_sequence = pad_sequences(sequences, maxlen=MAX_SEQUENCE_LENGTH, padding='post')

    prediction = model.predict(padded_sequence)
    response = {
        "text": text,
        "prediction": prediction.tolist()  
    }

    return jsonify(response)

@app.route('/chatbot', methods=['POST'])
def analyze_text():
    data = request.get_json()
    text = data.get('text', None)
    text = clean_text(text)

    if text is None:
        return jsonify({"error": "No text provided"}), 400

    sequences = tokenizer.texts_to_sequences([text])
    padded_sequence = pad_sequences(sequences, maxlen=MAX_SEQUENCE_LENGTH, padding='post')

    # Predict bullying label
    prediction = model.predict(padded_sequence)
    prediction_label = "Bullying" if prediction[0][0] > 0.5 else "Non-Bullying"

    # Predict bullying category (optional, based on text)
    bullying_category = "general cyberbullying"  # Default category
    if prediction_label == "Bullying":
        # Optionally, use keywords to further classify bullying type
        if "stalk" in text or "harass" in text:
            bullying_category = "cyberstalking"
        elif "defame" in text or "reputation" in text:
            bullying_category = "defamation"
        elif "threat" in text:
            bullying_category = "cyberthreats"
        else:
            bullying_category = "general cyberbullying"

    # Pass the user's message directly to OpenAI for legal advice
    openai_response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": "You are a legal expert in cyberbullying laws in India."},
            {"role": "user", "content": f"The user has reported the following: '{text}'. What are the legal consequences of {bullying_category} in India?"}
        ]
    )

    ai_response = openai_response["choices"][0]["message"]["content"]

    response = {
        "text": text,
        "prediction": prediction_label,
        "legal_advice": ai_response
    }

    return jsonify(response)


if _name_ == '_main_':
    app.run(debug=True)
