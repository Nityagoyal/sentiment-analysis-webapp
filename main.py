import glob
import streamlit as st
import plotly.express as px
import time
import nltk

# Download the VADER lexicon if it's not already available
try:
    nltk.data.find('sentiment/vader_lexicon')
except LookupError:
    nltk.download('vader_lexicon')

# Now you can initialize the SentimentIntensityAnalyzer
from nltk.sentiment.vader import SentimentIntensityAnalyzer



# Initialize sentiment analyzer
analyzer = SentimentIntensityAnalyzer()

# Streamlit title and entry section
st.title("Diary Tone")

# Initialize lists to store positivity and negativity values
positivity = []
negativity = []
dates = []

# Create a text input box for the user to write their diary entry
diary_entry = st.text_area("Write your diary entry here:")

# Create a button to submit the diary entry
if st.button("Analyze Sentiment"):
    if diary_entry:
        # Analyze the sentiment of the diary entry
        sentiment = analyzer.polarity_scores(diary_entry)

        # Display the sentiment analysis results
        st.write("Sentiment Analysis Results:")
        st.write(f"Positive: {sentiment['pos']*100:.2f}%")
        st.write(f"Neutral: {sentiment['neu']*100:.2f}%")
        st.write(f"Negative: {sentiment['neg']*100:.2f}%")

        # Append the new entry's sentiment to the lists
        positivity.append(sentiment['pos'])
        negativity.append(sentiment['neg'])
        dates.append(time.strftime("%Y-%m-%d_%H-%M-%S"))
    else:
        st.write("Please write something in your diary entry.")

# Display graphs of positivity and negativity over time
if dates:
    st.subheader("Positivity Over Time")
    pos_figure = px.line(x=dates, y=positivity,
                         labels={"x": "Date", "y": "Positivity"})
    st.plotly_chart(pos_figure)

    st.subheader("Negativity Over Time")
    neg_figure = px.line(x=dates, y=negativity,
                         labels={"x": "Date", "y": "Negativity"})
    st.plotly_chart(neg_figure)
