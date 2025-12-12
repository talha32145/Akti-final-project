# 💬 AI Customer Support & Product Suggestion Chatbot

This is a **Streamlit-based AI chatbot** that serves as a **customer support assistant** and **product recommendation system**. It can answer FAQs from a dataset and provide product suggestions based on a predefined product list. The chatbot leverages **Google's Gemini AI** for advanced product recommendation.

 Features

1.FAQ Chatbot
   - Answers user questions based on a CSV FAQ dataset.
   - Uses TF-IDF vectorization and cosine similarity for matching questions.
   - Handles queries with confidence thresholds and fallback messages.

2.Product Suggestion
   - Provides product recommendations from a predefined product list.
   - Suggests the **top 3 products** based on user requirements (brand, category, price).
   - Highlights the **best choice** with a short explanation.
   - Powered by **Google Gemini-2.5 AI model** for professional recommendations.

3.Conversational UI
   - Streamlit ChatGPT-style interface.
   - Maintains conversation history.
   - Supports simple greetings and follow-up messages.

4.Robust Error Handling
   - Handles missing or empty FAQ CSV datasets.
   - Provides fallback messages if AI service is unavailable.

5.Tech Stack
   - Python 3.10
   - Streamlit – Frontend chat interface
   - Pandas – Data handling
   - NLTK– Text preprocessing (tokenization, stopwords removal, lemmatization)
   - Scikit-learn– TF-IDF vectorizer, cosine similarity
   - Google Generative AI (Gemini) – Product recommendation
   - dotenv– Manage API keys securely


Quick Start

1.Clone & Install
```bash
git clone <your-repo-url>
cd ai-chatbot
