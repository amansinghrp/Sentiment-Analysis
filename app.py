import streamlit as st
import pickle
import re
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer

model_path = 'D:/Study/College/IV Sem/Mini Project/Codes/Trained Models/New Models/svm_tfidf.sav'

vectorizer_path = 'D:/Study/College/IV Sem/Mini Project/Codes/Trained Models/FreshModels/tfidf'
#load the model
model = pickle.load(open(model_path, 'rb'))
#load the vectorizer path
vectorizer = pickle.load(open(vectorizer_path, 'rb'))

# Preprocess the input review
def clean_text(text):
    #substituite the irrelevant characters
    text = re.sub(r'^[a-zA-Z\s]', '', text)
    #convert text to lower case
    text = text.lower()
    #tokenise the sentence in form of words
    tokens = word_tokenize(text)
    #set the stopwords for english language
    stop_words = set(stopwords.words('english'))
    #remove the stopwords
    filtered_tokens = [word  for word in tokens  if word not in stop_words]
    #applyign word net lemmatizer
    lemmatizer = WordNetLemmatizer()
    lemmatized_tokens = [lemmatizer.lemmatize(word)   for word in filtered_tokens]
    cleaned_text = ' '.join(lemmatized_tokens)
    return cleaned_text

# Predict sentiment
def predict_sentiment(review, model, vectorizer):
    review = clean_text(review)
    review_vector = vectorizer.transform([review])
    sentiment = model.predict(review_vector)
    return sentiment[0]

# Streamlit app
st.title("Sentiment Analysis of Student Reviews")
st.write("Enter a review and find out the sentiment.")

review = st.text_area("Review", "Type your review here...")

if st.button("Analyze Sentiment"):    
    sentiment = predict_sentiment(review, model, vectorizer)
    if sentiment == 'Positive':
        st.success(f"The sentiment of the review is: {sentiment}")
    elif sentiment == 'Negative':
        st.error(f"The sentiment of the review is: {sentiment}")
    else:
        st.warning(f"The sentiment of the review is: {sentiment}")
