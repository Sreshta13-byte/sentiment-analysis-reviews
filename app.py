import streamlit as st
import pandas as pd
from textblob import TextBlob
import plotly.express as px
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report, accuracy_score

# Streamlit page config
st.set_page_config(
    page_title="Amazon Product Sentiment Analyzer",
    page_icon="🛒",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
    .stApp {
        background-color: #84AE92;  /*sage green*/
        color: black;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# Custom CSS to hide watermark and improve layout
hide_streamlit_style = """
    <style>
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    .block-container {padding-top: 2rem;}
    </style>
"""
st.markdown(hide_streamlit_style, unsafe_allow_html=True)

# Title
st.markdown("<h1 style='text-align: center; color: white;'>AMAZON PRODUCT REVIEWS</h1>", unsafe_allow_html=True)
st.markdown("<h3 style='text-align: center; color: grey;'>Sentiment Analysis Dashboard</h3>", unsafe_allow_html=True)
st.write("")

# Load datasets
@st.cache_data
def load_data():
    df1 = pd.read_csv("1429_1.csv", on_bad_lines='skip')
    df2 = pd.read_csv("Datafiniti_Amazon_Consumer_Reviews_of_Amazon_Products.csv", on_bad_lines='skip')
    df3 = pd.read_csv("Datafiniti_Amazon_Consumer_Reviews_of_Amazon_Products_May19.csv", on_bad_lines='skip')
    return df1, df2, df3

df1, df2, df3 = load_data()

# Merge and clean
df = pd.concat([df1, df2, df3], ignore_index=True)
df = df.rename(columns=lambda x: x.lower())
if 'reviews.text' in df.columns:
    df.rename(columns={'reviews.text': 'text'}, inplace=True)
elif 'review' in df.columns:
    df.rename(columns={'review': 'text'}, inplace=True)

df = df[['text']].dropna().drop_duplicates()
df = df[df['text'].str.strip().astype(bool)]  # Remove empty strings

# Sidebar
with st.sidebar:
    st.subheader("📁 Dataset Overview")
    st.write(f"Total Reviews Combined: **{len(df)}**")
    st.write("Use the options below to explore and analyze sentiment.")

# Sentiment Analysis
def get_sentiment(text):
    return TextBlob(text).sentiment.polarity

with st.spinner("Analyzing Sentiments..."):
    df["polarity"] = df["text"].apply(get_sentiment)
    df["sentiment"] = df["polarity"].apply(lambda x: "Positive" if x > 0 else ("Negative" if x < 0 else "Neutral"))

# Tabs
tab1, tab2, tab3 = st.tabs([" Dashboard", " Sample Predictions", "Train the Model"])

with tab1:
    st.subheader(" Sentiment Distribution based on the Reviews")
    sentiment_count = df["sentiment"].value_counts()
    fig = px.pie(
        names=sentiment_count.index,
        values=sentiment_count.values,
        color=sentiment_count.index,
        color_discrete_map={"Positive": "green", "Negative": "red", "Neutral": "gray"},
        hole=0.4,
    )
    st.plotly_chart(fig, use_container_width=True)

    st.subheader("🔍 Sample Reviews")
    st.dataframe(df.sample(10), use_container_width=True)

with tab2:
    st.subheader("Try Your Own Review")
    user_review = st.text_area("Write a product review below:")

    if st.button("Analyze Sentiment"):
        if user_review.strip():
            polarity = get_sentiment(user_review)
            sentiment = (
                "Positive" if polarity > 0 else "Negative" if polarity < 0 else "Neutral"
            )
            color = "green" if sentiment == "Positive" else "red" if sentiment == "Negative" else "gray"
            st.markdown(f"**Predicted Sentiment:** <span style='color:{color}'>{sentiment}</span>", unsafe_allow_html=True)
        else:
            st.warning("Please enter a review.")

with tab3:
    st.subheader("🤖 Train a Simple Sentiment Classifier")
    df_filtered = df[df["sentiment"] != "Neutral"]  # Binary classification
    X = df_filtered["text"]
    y = df_filtered["sentiment"]

    cv = CountVectorizer()
    X_vec = cv.fit_transform(X)

    X_train, X_test, y_train, y_test = train_test_split(X_vec, y, test_size=0.2, random_state=42)
    model = MultinomialNB()
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    acc = accuracy_score(y_test, y_pred)
    st.write(f"✅ **Model Accuracy:** {acc:.2f}")
    st.text("Classification Report:")
    st.code(classification_report(y_test, y_pred), language='text')
