import streamlit as st
import pickle 
import string
from nltk.corpus import stopwords
import nltk    
from nltk.stem.porter import PorterStemmer

nltk.download("punkt")
nltk.download("stopwords")

ps = PorterStemmer()

def transform_text(text):
    text = text.lower()
    text = nltk.word_tokenize(text)
    y = []
    for i in text :
        if i.isalnum():
            y.append(i)
    text = y[:]
    y.clear()

    for i in text :
        if i not in stopwords.words("english") and i not in string.punctuation :
            y.append(i)
    text = y[:]
    y.clear()
    for i in text:
        y.append(ps.stem(i))
    return " ".join(y)
tfidf = pickle.load(open("Vectorizer.pkl", "rb"))
model = pickle.load(open("Model.pkl", "rb"))  

st.title("SMS Classifier")

input_msg = st.text_area("Enter The Message")   
if st.button("Predict"):
    if input_msg.strip() == "":
        st.warning("Please Enter A Message")
    else:

        transform_msg = transform_text(input_msg)
        vector_input = tfidf.transform([transform_msg])
        result = model.predict(vector_input)[0]
        if result == 1 :
            st.text("Spam")
        else :
            st.text("Not Spam")
# import streamlit as st
# import pickle 
# import string
# import nltk    
# from nltk.corpus import stopwords
# from nltk.stem.porter import PorterStemmer

# nltk.download('punkt')
# nltk.download('stopwords')

# ps = PorterStemmer()

# def transform_text(text):
#     text = text.lower()
#     text = nltk.word_tokenize(text)
#     y = []
#     for i in text:
#         if i.isalnum():
#             y.append(i)
#     text = y[:]
#     y.clear()

#     for i in text:
#         if i not in stopwords.words("english") and i not in string.punctuation:
#             y.append(i)
#     text = y[:]
#     y.clear()

#     for i in text:
#         print(f"Stemming: {i}")  # For debugging
#         y.append(ps.stem(i))

#     return " ".join(y)

# # Load model and vectorizer
# tfidf = pickle.load(open("Vectorizer.pkl", "rb"))
# model = pickle.load(open("Model.pkl", "rb"))  

# st.title("SMS Classifier")

# input_msg = st.text_area("Enter The Message")   

# if st.button("Predict"):
#     if input_msg.strip() == "":
#         st.warning("Please enter a message.")
#     else:
#         transform_msg = transform_text(input_msg)
#         vector_input = tfidf.transform([transform_msg])
#         result = model.predict(vector_input)[0]
#         if result == 1:
#             st.text("Spam")
#         else:
#             st.text("Not Spam")

        
