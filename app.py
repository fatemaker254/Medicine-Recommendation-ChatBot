import base64
import streamlit as st
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel

# ------------------ Load Dataset ------------------
data = pd.read_csv("Medicine_Details.csv")

# Concatenate symptom and side effects columns to create a corpus
data["Symptom_SideEffects"] = data["Uses"].fillna("") + " " + data["Side_effects"].fillna("")

# Initialize the TF-IDF vectorizer
tfidf_vectorizer = TfidfVectorizer(stop_words="english")

# Fit and transform the corpus
tfidf_matrix = tfidf_vectorizer.fit_transform(data["Symptom_SideEffects"])

# ------------------ Recommendation Function ------------------
def recommend_medicines(symptoms, cosine_sim_matrix=tfidf_matrix):
    # Transform input symptoms to TF-IDF vector
    symptoms_tfidf = tfidf_vectorizer.transform([symptoms])

    # Calculate similarity scores
    cosine_scores = linear_kernel(symptoms_tfidf, cosine_sim_matrix).flatten()

    # Get indices of top medicines
    top_indices = cosine_scores.argsort()[::-1]

    # Check max similarity score
    max_score = cosine_scores[top_indices[0]]

    # If similarity is too low, return None
    if max_score < 0.2:  # threshold (tune this if needed)
        return None

    # Get top 5 medicines
    top_medicines = data.iloc[top_indices[:5]]["Medicine Name"].values
    return top_medicines

# ------------------ Background Image ------------------
def add_bg_from_local(image_file):
    with open(image_file, "rb") as f:
        encoded_img = base64.b64encode(f.read()).decode()

    page_bg_img = f"""
    <style>
    .stApp {{
        background-image: url("data:image/jpg;base64,{encoded_img}");
        background-size: cover;
        background-position: center;
        background-attachment: fixed;
    }}
    .chat-box {{
        background: rgba(255, 255, 255, 0.85);
        padding: 10px;
        border-radius: 10px;
        margin: 5px 0;
    }}
    .user-msg {{
        color: blue;
        font-weight: bold;
    }}
    .bot-msg {{
        color: green;
        font-weight: bold;
    }}
    </style>
    """
    st.markdown(page_bg_img, unsafe_allow_html=True)

# Call background
add_bg_from_local("back.jpg")  # make sure "back.jpg" is in the same folder

# ------------------ Streamlit UI ------------------
st.title("💊 Medicine Recommendation Chatbot")

# Initialize session state for chat history
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# User input
symptoms_input = st.text_input("Enter your symptoms:")

if st.button("Submit"):
    if symptoms_input.strip():
        recommended_medicines = recommend_medicines(symptoms_input)

        if recommended_medicines is not None:
            reply = "Recommended Medicines:\n" + "\n".join([f"- {med}" for med in recommended_medicines])
        else:
            reply = "❌ No relevant medicines found. Please enter valid symptoms."

        # Save chat history
        st.session_state.chat_history.append(("You", symptoms_input))
        st.session_state.chat_history.append(("Bot", reply))
    else:
        st.warning("Please enter your symptoms.")

# Display chat history
st.subheader("Chat History")
for speaker, msg in st.session_state.chat_history:
    if speaker == "You":
        st.markdown(f"<div class='chat-box user-msg'>👤 {msg}</div>", unsafe_allow_html=True)
    else:
        st.markdown(f"<div class='chat-box bot-msg'>🤖 {msg}</div>", unsafe_allow_html=True)

# Disclaimer
st.markdown(
    """
    <hr>
    <p style="text-align:center; color:red; font-size:14px;">
    ⚠️ Disclaimer: This chatbot is for demonstration purposes only. 
    Please consult a qualified doctor for medical advice. 
    Use at your own risk.
    </p>
    """,
    unsafe_allow_html=True
)
