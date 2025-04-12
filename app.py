import streamlit as st
import pickle
import re
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer

# Download NLTK resources
nltk.download('punkt')
nltk.download('stopwords')

# Initialize stemmer and stopwords
stemmer = PorterStemmer()
stop_words = set(stopwords.words('english'))

# Function to preprocess and transform input text
def transform_text(text):
    # Convert to lowercase
    text = text.lower()

    # Remove URLs, mentions, hashtags, numbers, and special characters
    text = re.sub(r"http\S+|www\S+|https\S+", '', text)  # URLs
    text = re.sub(r'@\w+|#\w+', '', text)  # Mentions/hashtags
    text = re.sub(r'[^a-z\s]', '', text)  # Non-alphabets

    # Tokenization
    words = word_tokenize(text)

    # Remove stopwords and short words
    filtered_words = [word for word in words if word not in stop_words and len(word) > 2]

    # Stemming
    stemmed_words = [stemmer.stem(word) for word in filtered_words]

    return " ".join(stemmed_words)

# Load pre-trained model and vectorizer
tfidf = pickle.load(open("vectorizer.pkl", "rb"))
model = pickle.load(open("model.pkl", "rb"))

# Custom CSS for background and content box styling
st.markdown("""
    <style>
        .stApp {
            background-color:#25aedb; /* Light gray background */
        }
        .main-box {
            border: 2px solid #53227a; /* Blue border */
            border-radius: 10px;
            padding: 20px;
            background-color: #ffffff; /* White background */
            box-shadow: 0 4px 8px rgba(0, 0, 0, 0.1);
        }
    </style>
""", unsafe_allow_html=True)

# Streamlit app layout
with st.container():
    st.markdown('<div class="main-box">', unsafe_allow_html=True)
    st.title("Email/SMS Classifier")
    input_sms = st.text_input("Enter the message")

    if st.button("Predict"):
        # Preprocess input
        transformed_sms = transform_text(input_sms)

        # Vectorize input
        vector_input = tfidf.transform([transformed_sms])

        # Predict using the model
        result = model.predict(vector_input)

        # Display result
        if result == 1:
            st.header("Spam")
        else:
            st.header("Not a Spam")

    st.markdown('</div>', unsafe_allow_html=True)
