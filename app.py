import streamlit as st
import torch
from transformers import MarianMTModel, MarianTokenizer

# Load the trained model and tokenizer
model_name = 'Helsinki-NLP/opus-mt-en-es'  # Example model for English to Spanish translation
tokenizer = MarianTokenizer.from_pretrained(model_name)
model = MarianMTModel.from_pretrained(model_name)

# Function to translate text
def translate_text(text):
    # Tokenize the input text
    inputs = tokenizer(text, return_tensors='pt', padding=True)
    # Generate translation using the model
    translated_tokens = model.generate(**inputs)
    # Decode the translated tokens to text
    translated_text = tokenizer.decode(translated_tokens[0], skip_special_tokens=True)
    return translated_text

# Streamlit app layout
st.title("Machine Translation Web Application")
st.write("Enter a sentence or phrase in English to translate it to Spanish:")

# Input text box
input_text = st.text_area("Input Text", "")

# Button to trigger translation
if st.button("Translate"):
    if input_text.strip():
        translated_text = translate_text(input_text)
        st.write("Translated Text:", translated_text)
    else:
        st.write("Please enter a sentence or phrase to translate.")

# Run the Streamlit app
if __name__ == '__main__':
    st._is_running_with_streamlit = True
    st._is_running_with_streamlit = False
