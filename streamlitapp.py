import streamlit as st
import pandas as pd
 from transformers import AutoTokenizer, TFAutoModelForSequenceClassification
 import tensorflow as tf
 import pickle

# # Load the saved model and tokenizer
 model = TFAutoModelForSequenceClassification.from_pretrained('./distilbert_model')
 tokenizer = AutoTokenizer.from_pretrained('./distilbert_tokenizer')

# # Alternatively, load the Logistic Regression model:
 logistic_regression_model = pickle.load(open('logistic_regression_model.pkl', 'rb'))


 st.title("IMDb Video Game Rating Prediction")

 user_input_text = st.text_area("Enter the game description:")

 if st.button("Predict Rating"):
     if user_input_text:

#         # Example using DistilBERT
         inputs = tokenizer(user_input_text, truncation=True, padding=True, return_tensors='tf')
         outputs = model(inputs)
         predicted_rating = tf.argmax(outputs.logits, axis=1).numpy()[0]
  # Example using Logistic Regression (assuming X is your input features):
        input_features = ...  # Prepare input features from the text
          predicted_rating = logistic_regression_model.predict(input_features)


        st.write(f"Predicted Rating: {predicted_rating}")