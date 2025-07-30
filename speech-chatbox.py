import streamlit as st
import nltk
import speech_recognition as sr
import random
import string

# Download required nltk data
nltk.download('punkt')
nltk.download('wordnet')

from nltk.stem import WordNetLemmatizer

# --- Load and preprocess chatbot corpus ---
def load_corpus():
    with open("chatbox.txt", "r", encoding="utf-8") as file:
        return file.read().lower()

lemmatizer = WordNetLemmatizer()

def preprocess_text(text):
    tokens = nltk.word_tokenize(text.lower())
    tokens = [lemmatizer.lemmatize(word) for word in tokens if word not in string.punctuation]
    return tokens

# --- Response generation logic ---
def generate_response(user_input, sentence_tokens):
    user_input = user_input.lower()
    sentence_tokens.append(user_input)

    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.metrics.pairwise import cosine_similarity

    vectorizer = TfidfVectorizer(tokenizer=preprocess_text, stop_words='english')
    tfidf = vectorizer.fit_transform(sentence_tokens)
    similarity = cosine_similarity(tfidf[-1], tfidf)

    idx = similarity.argsort()[0][-2]
    flat = similarity.flatten()
    flat.sort()
    score = flat[-2]

    sentence_tokens.pop()

    if score < 0.2:
        return "I didn't quite get that. Can you say it differently?"
    else:
        return sentence_tokens[idx]

# --- Speech recognition function ---
def recognize_speech():
    recognizer = sr.Recognizer()
    with sr.Microphone() as source:
        st.info("Listening... Speak now.")
        audio = recognizer.listen(source, timeout=5)
        try:
            text = recognizer.recognize_google(audio)
            st.success(f"You said: {text}")
            return text
        except sr.UnknownValueError:
            st.error("Sorry, I couldn't understand the audio.")
        except sr.RequestError:
            st.error("Could not request results from Google Speech Recognition service.")
        return None

# --- Main Streamlit App ---
def main():
    st.title("🗣️ Speech-Enabled Chatbot")
    st.write("You can either **type** or **speak** to the chatbot.")

    corpus = load_corpus()
    sentence_tokens = nltk.sent_tokenize(corpus)

    input_mode = st.radio("Choose input method:", ("Text", "Speech"))

    if input_mode == "Text":
        user_input = st.text_input("You:")
        if st.button("Send"):
            if user_input:
                response = generate_response(user_input, sentence_tokens)
                st.text_area("Chatbot:", response, height=100)
    else:
        if st.button("Speak"):
            speech_text = recognize_speech()
            if speech_text:
                response = generate_response(speech_text, sentence_tokens)
                st.text_area("Chatbot:", response, height=100)

if __name__ == "__main__":
    main()
