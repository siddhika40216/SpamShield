import warnings
warnings.filterwarnings("ignore")
import os
os.environ["STREAMLIT_WATCHER_TYPE"] = "none"
import nltk
nltk.download('punkt')
nltk.download('stopwords')

import streamlit as st
import pickle 
import pandas as pd
import string
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer

ps = PorterStemmer()


def transform_text(text):
    text = text.lower()
    text = text.split()

    y = []
    for i in text:
        if i.isalnum():
            y.append(i)

    text = y[:]
    y.clear()

    for i in text:
        if i not in stopwords.words('english') and i not in string.punctuation:
            y.append(i)

    text = y[:]
    y.clear()

    for i in text:
        y.append(ps.stem(i))

    return " ".join(y)

tk = pickle.load(open("vectorizer.pkl", 'rb'))
model = pickle.load(open("model.pkl", 'rb'))

import sqlite3

conn = sqlite3.connect("spam_history.db")
cursor = conn.cursor()

cursor.execute("""
CREATE TABLE IF NOT EXISTS history(
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    message TEXT,
    prediction TEXT,
    probability REAL
)
""")

conn.commit()
   

st.title("SpamShield - SMS & Email Spam Detection")
st.markdown("""
    <style>

    /* App background */
    .stApp {
        background: linear-gradient(
135deg,
#bbf7d0,
#d9f99d,
#dcfce7
);
    }

    /* Title styling */
    h1 {
        text-align: center;
        color: #0f172a;
        font-size: 40px;
        margin-bottom: 10px;
    }

    /* Subtitle / text */
    .stMarkdown p {
        text-align: center;
        color: #334155;
        font-size: 16px;
    }

    /* Input box */
    .stTextInput>div>div>input {
        border-radius: 12px;
        padding: 12px;
        border: 1px solid #cbd5e1;
        background-color: white;
    }

    /* Button */
    .stButton>button {
        width: 100%;
        border-radius: 12px;
        padding: 10px;
        background-color: #2563eb;
        color: white;
        font-size: 16px;
        font-weight: bold;
        transition: 0.3s;
    }

    .stButton>button:hover {
        background-color: #1d4ed8;
    }

    /* Result box */
    .result-box {
        margin-top: 20px;
        padding: 18px;
        border-radius: 14px;
        text-align: center;
        font-size: 20px;
        font-weight: bold;
        box-shadow: 0px 4px 12px rgba(0,0,0,0.1);
    }

    .spam {
        background-color: #fee2e2;
        color: #b91c1c;
        border-left: 6px solid #ef4444;
    }

    .not-spam {
        background-color: #dcfce7;
        color: #166534;
        border-left: 6px solid #22c55e;
    }

    </style>
""", unsafe_allow_html=True)
st.write("*Made with ❤️‍🔥 by Siddhika*")
    
input_sms = st.text_area(
    "Enter SMS or Email Content"
)
st.write("Characters:", len(input_sms))
st.write("Words:", len(input_sms.split()))

if st.button('Predict'):

    # 1. preprocess
    transformed_sms = transform_text(input_sms)
    # 2. vectorize
    vector_input = tk.transform([transformed_sms])
    # 3. predict
    result = model.predict(vector_input)[0]
    prob = model.predict_proba(vector_input)[0][1]
    st.write(
    f"Spam Probability: {prob*100:.2f}%"
    )

    result = 1 if prob >= 0.40 else 0
    # 4. Display
    if result == 1:
        prediction_text = "Spam"
        st.markdown('<div class="result-box spam">🚨 Spam Message detected</div>', unsafe_allow_html=True)
    else:
        prediction_text="Not Spam"
        st.markdown('<div class="result-box not-spam">✅ Not Spam....Legitimate Message</div>', unsafe_allow_html=True)    

   

    st.write("Raw Prediction:", result)
    transformed_sms = transform_text(input_sms)

    st.write("Processed:", transformed_sms)

    prob = model.predict_proba(vector_input)[0]

    st.write("Probabilities:", prob)