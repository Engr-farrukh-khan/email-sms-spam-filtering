import streamlit as st
import pickle
import string
from nltk.corpus import stopwords
import nltk
from nltk.stem.porter import PorterStemmer

# Set up the NLTK dependencies
nltk.download('punkt')
nltk.download('stopwords')

ps = PorterStemmer()

# Text transformation function
def transform_text(Text):
    Text = Text.lower()
    Text = nltk.word_tokenize(Text)

    y = []
    for i in Text:
        if i.isalnum():
            y.append(i)
    
    Text = y[:]
    y.clear()

    for i in Text:
        if i not in stopwords.words('english') and i not in string.punctuation:
            y.append(i)

    Text = y[:]
    y.clear()

    for i in Text:
        y.append(ps.stem(i))
    
    return " ".join(y)

# Load vectorizer and model
tfidf = pickle.load(open('vectorizer.pkl','rb'))
model = pickle.load(open('model.pkl','rb'))

# Streamlit app design
st.set_page_config(page_title="Email/SMS Spam Classifier", page_icon="📧", layout="centered")

# Title and description
st.title("📧 Email/SMS Spam Classifier")
st.markdown("""
    This app uses **machine learning** to classify whether a given message is **Spam** or **Not Spam**. 
    Simply enter your message below and hit the **Predict** button to see the result.
""")

# Input text area
input_sms = st.text_area(
    "💬 Enter the Message", 
    height=250,  # Adjust height as needed for a fixed-size box
    max_chars=None,  # No limit on the number of characters
    placeholder="Type your message here..."
)


# Add a 'Predict' button with customized styling
if st.button('🚀 Predict'):

    # Preprocess the input text
    transform_sms = transform_text(input_sms)

    # Vectorize the input text
    vector_input = tfidf.transform([transform_sms])

    # Make the prediction
    result = model.predict(vector_input)[0]

    # Display the result with enhanced UI
    if result == 1:
        st.error("⚠️ **Spam Message Detected!**")
    else:
        st.success("✅ **Not Spam**")

# Footer
st.markdown("""
    ---
    Developed by [Farrukh Noor Khan](https://www.linkedin.com/in/farrukhkhan-f12). Powered by **Streamlit** and **Machine Learning**.
""")

# Apply some custom CSS to style the UI
st.markdown("""
    <style>
    .stButton>button {
        background-color: #4CAF50;
        color: white;
        padding: 10px 24px;
        font-size: 16px;
        border-radius: 8px;
    }
    .stButton>button:hover {
        background-color: #45a049;
    }
    </style>
    """, unsafe_allow_html=True)
