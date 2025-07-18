import numpy as np
import pandas as pd
import re
import json
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Bidirectional, Dense, Dropout
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, precision_score, recall_score, f1_score
import pickle

# Define dataset paths
jsonpath = "Dataset for Detection of Cyber-Trolls.json"
csvpath = "CyberBullyingTypesDataset.csv"
pickle_model_path = "BI_LSTM/models/model.pkl"
keras_model_path = "BI_LSTM/models/model.keras"

# Load cleaned dataset
cleaned_df = pd.read_csv("cleaned_dataset.csv")
texts = cleaned_df['texts'].tolist()
combined_labels = cleaned_df['label'].tolist()

# Preprocess text
def clean_text(text):
    text = re.sub(r"http\S+|www\S+", "", text)
    text = re.sub(r"@\w+", "", text)
    text = text.lower()
    text = re.sub(r"[^a-zA-Z0-9\s]", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text

cleaned_df["texts"] = cleaned_df["texts"].apply(clean_text)

# Tokenization and padding
tokenizer = Tokenizer(num_words=10000, oov_token="<OOV>")
tokenizer.fit_on_texts(cleaned_df["texts"])
sequences = tokenizer.texts_to_sequences(cleaned_df["texts"])
padded_sequences = pad_sequences(sequences, maxlen=100, padding="post")

# Label encoding
label_encoder = LabelEncoder()
encoded_labels = label_encoder.fit_transform(cleaned_df["label"])

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(padded_sequences, encoded_labels, test_size=0.2, random_state=42)

# Define BiLSTM model
model = Sequential([
    Embedding(input_dim=10000, output_dim=128, input_length=100),
    Bidirectional(LSTM(64, return_sequences=True)),
    Dropout(0.2),
    Bidirectional(LSTM(32)),
    Dense(32, activation="relu"),
    Dropout(0.2),
    Dense(1, activation="sigmoid")
])

model.compile(loss="binary_crossentropy", optimizer="adam", metrics=["accuracy"])

# Train model
history = model.fit(X_train, y_train, epochs=5, batch_size=32, validation_data=(X_test, y_test))

# Evaluate model
loss, accuracy = model.evaluate(X_test, y_test)
print(f"Test Accuracy: {accuracy*100:.2f}%")

y_pred = model.predict(X_test)
y_pred_labels = (y_pred > 0.5).astype(int)

print("\nClassification Report:")
print(classification_report(y_test, y_pred_labels))

precision = precision_score(y_test, y_pred_labels)
recall = recall_score(y_test, y_pred_labels)
f1 = f1_score(y_test, y_pred_labels)

print(f"Precision: {precision:.2f}")
print(f"Recall: {recall:.2f}")
print(f"F1 Score: {f1:.2f}")
