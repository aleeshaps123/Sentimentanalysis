import streamlit as st
import pickle
from sklearn.feature_extraction.text import TfidfVectorizer

# Load the trained model and the TF-IDF vectorizer
with open('best_sentiment_model.pkl', 'rb') as model_file:
    model = pickle.load(model_file)

with open('tfidf_vectorizer.pkl', 'rb') as vectorizer_file:
    vectorizer = pickle.load(vectorizer_file)

# Define a function to predict sentiment
def predict_sentiment(review_text):
    review_tfidf = vectorizer.transform([review_text])
    sentiment_encoded = model.predict(review_tfidf)[0]
    sentiment_map = {1: 'Positive', 0: 'Neutral', -1: 'Negative'}
    return sentiment_map[sentiment_encoded]

# Streamlit interface
st.title("Sentiment Analysis App")
st.write("Enter a review text to predict its sentiment:")

user_input = st.text_area("Review Text")

if st.button("Predict"):
    if user_input:
        prediction = predict_sentiment(user_input)
        st.write(f"Predicted Sentiment: {prediction}")
    else:
        st.write("Please enter a review text.")

