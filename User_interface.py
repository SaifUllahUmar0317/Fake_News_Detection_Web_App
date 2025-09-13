import streamlit as st
import joblib as jb
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import pandas as pd

st.title("📰 Fake News Detector")
st.write("Enter a news headline and article to check if it's **likely Real or Fake**. ")

@st.cache_resource
def load_model():
    with open("Fake_News_Detector_Model.pkl", "rb") as md:
        return jb.load(md)

@st.cache_resource
def load_vectorizer():
    with open("Vectorizer.pkl", "rb") as vec:
        return jb.load(vec)

model = load_model()
tfidf = load_vectorizer()

title = st.text_area("Enter the News Headline")
text = st.text_area("Enter the Article Body")

if st.button("Check"):
    if not title.strip() and not text.strip():
        st.error("⚠️ Please enter a headline or article text.")
    else:
        combo_text = (title + " " + text).lower()
        tokens = word_tokenize(combo_text)
        stop_words = set(stopwords.words("english"))
        lemmatizer = WordNetLemmatizer()

        clean_text = " ".join([lemmatizer.lemmatize(word) 
                               for word in tokens if word.isalpha() and word not in stop_words])

        X_tfidf = tfidf.transform([clean_text])
        prediction = model.predict(X_tfidf)
        prob = model.predict_proba(X_tfidf)[0]

        if prediction[0] == 0:
            st.warning(f"🛑 This News is **Fake** (Confidence: {prob[0]:.2%})")
        else:
            st.success(f"✅ This News is **Real** (Confidence: {prob[1]:.2%})")
