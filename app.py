import pickle
import streamlit as st
import re

# Load the spam model and vectorizer
spam_model = pickle.load(open('Spam_Model.sav', 'rb'))
vectorizer = pickle.load(open('Vectorizer.sav', 'rb'))

st.title("Spam Message Detection Web App")
st.write("Enter a message to detect if it's spam or not")

user_message = st.text_area("Enter your message here", value='')

if st.button("Detect Spam"):
    if user_message:
        # Preprocess and vectorize the user input
        user_message = re.sub('\W', ' ', user_message)  # Clean the input
        user_message = user_message.lower()  # Convert to lowercase
        user_message_vectorized = vectorizer.transform([user_message])
        
        # Predict spam probability
        spam_prediction = spam_model.predict(user_message_vectorized)
        spam_probability = spam_model.predict_proba(user_message_vectorized)[:, 1][0]
        
        if spam_prediction[0] == 1:
            spam_diagnosis = "The message is classified as spam"
        else:
            spam_diagnosis = "The message is not classified as spam"
        
        st.success(spam_diagnosis)
        st.write(f"Spam Probability: {spam_probability:.2f}")
    else:
        st.warning("Please enter a message to detect spam.")
