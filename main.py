import streamlit as st
import pickle
from tensorflow.keras.preprocessing.sequence import pad_sequences


# Load model and tokenizer
model = pickle.load(open('lstm_model.pkl', 'rb'))
tokenizer = pickle.load(open('tokenizer.pkl', 'rb'))

# Define maxlen (must match training setup)
MAXLEN = 100

# Prediction function
def predict_toxicity(text):
    # Convert text to sequence
    seq = tokenizer.texts_to_sequences([text])
    # Pad sequence
    padded_seq = pad_sequences(seq, maxlen=MAXLEN)
    # Predict
    prediction = model.predict(padded_seq)
    label = (prediction[0][0] > 0.5).astype(int)  # binary classification
    return "Toxic" if label == 1 else "Not Toxic", float(prediction[0][0])

# Streamlit 
st.title( "Toxicity Detection ")
st.write("Enter a comment below to check if it is toxic or not.")

user_input = st.text_area("Your comment:")

if st.button("Predict"):
    if user_input.strip():
        result, confidence = predict_toxicity(user_input)
        st.success(f"Prediction: {result} (Confidence: {confidence:.2f})")
    else:
        st.warning("Please enter a comment before predicting.")