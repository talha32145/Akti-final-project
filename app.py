import streamlit as st
import pandas as pd
import string
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import google.generativeai as genai
from dotenv import load_dotenv
import os
import logging
from productList import product_list


logger=logging.getLogger()


load_dotenv("secret_key.env")

my_api_key=os.getenv("YOUR_API_KEY")

# ------------------------ LOAD DATA ----------------------------
try:
    df = pd.read_csv("improved_faq_dataset.csv")
except FileNotFoundError:
    logger.error("Faqs dataset not found")
    st.error("Faqs dataset is missing. Please upload Faqs dataset")
except pd.errors.EmptyDataError:
    logger.error("CSV file is empty")
    st.error("Faqs dataset is empty.")


#--------------------------History----------------------------------
def get_conversation_history(messages, max_history=30):
    """Extract conversation history from messages for AI context"""
    history = []
    # Take last N messages for context (avoid token limits)
    recent_messages = messages[-(max_history * 2):] if len(messages) > (max_history * 2) else messages
    
    for msg in recent_messages:
        if msg["role"] == "user":
            history.append({"role": "user", "content": msg["content"]})
        elif msg["role"] == "bot":
            history.append({"role": "assistant", "content": msg["content"]})
    
    return history

# ------------------------ PREPROCESSING ----------------------------
def preprocessing(text):
    text = text.translate(str.maketrans("", "", string.punctuation))
    tokenization = word_tokenize(text)
    stop_words = set(stopwords.words("english"))
    tokenization = [word for word in tokenization if word.lower() not in stop_words]
    lemmatizer = WordNetLemmatizer()
    tokenization = [lemmatizer.lemmatize(word, pos="v") for word in tokenization]
    return " ".join(tokenization)

df["question"] = df["question"].apply(preprocessing)

vectorizer = TfidfVectorizer()
try:
    x = vectorizer.fit_transform(df["question"])
except Exception as e:
    logger.error(f"Vectorizer error {e}")
    st.error("⚠️ Something went wrong while understanding your question.")


# ------------------------ PRODUCT LIST ----------------------------
# product_list = [
#     "Electronics","Home Appliances","Accessories","Gadgets",
#     "Mobile Phones","Smartphones",
#     "LG Refrigerators","LG Washing Machines","LG Microwave Ovens","LG Air Conditioners","LG Televisions","LG Irons",
#     "Laptops:HP,Dell,Asus ROG,Lenovo,MSI,Acer",
#     "Gaming Laptops:Asus ROG,Lenovo,MSI,Acer",
#     "Samsung Mobiles","Apple Mobiles","Xiaomi Mobiles",
# ]

# ------------------------ GEMINI MODEL ----------------------------
genai.configure(api_key=my_api_key)

model = genai.GenerativeModel(
    "gemini-2.5-flash-lite",
    generation_config={"max_output_tokens": 1080, "temperature": 0.2}
)

# ------------------------ PRODUCT SUGGESTION ----------------------------

def Product_suggestion(user_input,conversation_history):
    prompt = f"""
You are a professional product recommendation assistant.

You can only suggest products from this list:
{product_list}

When a user asks for a product suggestion, follow these rules:

1. Only suggest items that exist in the provided list. Never invent products.
2. If the user asks "Suggest me X under Y price":
    - Suggest the top 3 products that match the category/brand and are under the price limit.
    - Include for each product:
        - Product name
        - Brand
        - Category
        - Price in PKR
        - Key specifications
    - Highlight one product as the "best choice" with a short reason.
3. If nothing matches, respond with a friendly message like:
   "Sorry, no product is available under that price."
   {user_input}
Previous conversation history:
{conversation_history}
ALWAYS continue the context based on the full history above.
Maintain conversation flow and remember previous topics discussed.
"""

    try:
        response = model.generate_content(prompt)
        return response.text
    except Exception as e:
        logger.error(f"Gemini API Error: {e}")
        return "⚠️ Sorry, our AI recommendation service is temporarily unavailable."

# ------------------------ FAQ CHATBOT ----------------------------
def faq_chatbot(user_input):
    clean = preprocessing(user_input)
    vectorized = vectorizer.transform([clean])
    similarity = cosine_similarity(x, vectorized)
    index = similarity.argmax()
    if similarity.max() > 0.3:  # Confidence threshold
        return df["answer"].iloc[index]
    else:
        return "I'm not sure I understand. Could you please rephrase your question or contact our support team at support@example.com?"

# ------------------------ STREAMLIT UI (CHATGPT STYLE) ----------------------------
st.set_page_config(page_title="AI Support Chatbot", layout="wide")
st.markdown("""
    <style>
        
        .user-msg {
            background: #0a84ff;
            padding: 10px 15px;
            border-radius: 12px;
            margin: 10px 0;
            color: white;
            width: fit-content;
            max-width: 80%;
            margin-left: auto;
        }
        .bot-msg {
            background: #2e2e2e;
            padding: 10px 15px;
            border-radius: 12px;
            margin: 10px 0;
            color: white;
            width: fit-content;
            max-width: 80%;
        }
    </style>
""", unsafe_allow_html=True)

st.title("💬 AI Customer Support & Product Suggestion Chatbot")

if "chat" not in st.session_state:
    st.session_state.chat = []

# ------------------------ CHAT WINDOW ----------------------------
st.markdown("<div class='chat-container'>", unsafe_allow_html=True)
for sender, msg in st.session_state.chat:
    if sender == "user":
        st.markdown(f"<div class='user-msg'>{msg}</div>", unsafe_allow_html=True)
    else:
        st.markdown(f"<div class='bot-msg'>{msg}</div>", unsafe_allow_html=True)
st.markdown("</div>", unsafe_allow_html=True)

# ------------------------ USER INPUT ----------------------------
if "messages" not in st.session_state:
    st.session_state.messages = []

user_input = st.chat_input("Ask something...")
if user_input is not None:
    msg=user_input.strip()

    if not msg:
        st.warning("⚠️ Please type something.")
    elif len(msg) > 300:
        st.warning("⚠️ Your message is too long, please keep it shorter.")
    else:

        st.session_state.chat.append(("user", user_input))
        st.session_state.messages.append({"role": "user", "content": user_input})

        typing_placeholder = st.empty()
        typing_placeholder.markdown(
            "<div class='bot-msg'><span class='bot-icon'>🤖</span>Typing...</div>",
            unsafe_allow_html=True
        )

        conversation_history = get_conversation_history(st.session_state.messages)


        if msg.lower() in ['hello','hi','hey']:
            bot_reply=f"{user_input} sir i am your product assistance how can i help you"
        elif msg.lower() in ["okay","ok"]:
            bot_reply="If you want further detail contact us at -> support@example.com"
        # Product Suggestion
        elif any(k in user_input.lower() for k in ["suggest", "recommend", "phone", "mobile", "electronics", "under"]):
            bot_reply = Product_suggestion(user_input,conversation_history)

        # FAQ Search
        else:
            bot_reply = faq_chatbot(user_input)

        st.session_state.chat.append(("bot", bot_reply))
        st.session_state.messages.append({"role": "assistant", "content": bot_reply})

        st.rerun()