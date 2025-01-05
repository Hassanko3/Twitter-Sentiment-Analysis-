import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
from wordcloud import WordCloud
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.utils import resample
import streamlit as st

# Preprocessing
st.title("Twitter Sentiment Analysis")
st.subheader("Dataset Loading and Preprocessing")
data_path = "twee_semantic.csv"  # Update with your dataset path

def advanced_preprocess(text):
    lemmatizer = WordNetLemmatizer()
    stop_words = set(stopwords.words('english'))
    words = text.split()
    words = [lemmatizer.lemmatize(word) for word in words if word not in stop_words]
    return " ".join(words)

data = pd.read_csv(data_path)
data['text'] = data['text'].fillna('').str.lower()
data['text'] = data['text'].apply(advanced_preprocess)

# Balancing the Dataset
st.write("Balancing the Dataset")
positive = data[data['sentiment'] == 'positive']
neutral = data[data['sentiment'] == 'neutral']
negative = data[data['sentiment'] == 'negative']

neutral_resampled = resample(neutral, replace=True, n_samples=len(positive), random_state=42)
negative_resampled = resample(negative, replace=True, n_samples=len(positive), random_state=42)

data_balanced = pd.concat([positive, neutral_resampled, negative_resampled])
st.write("Dataset after balancing:")
st.dataframe(data_balanced['sentiment'].value_counts())

# Visualization of Sentiment Distribution
st.subheader("Sentiment Distribution")
sentiment_counts = data_balanced['sentiment'].value_counts()
fig, ax = plt.subplots()
sentiment_counts.plot(kind='bar', color=['blue', 'green', 'red'], ax=ax)
ax.set_title("Distribution of Sentiments")
ax.set_xlabel("Sentiment")
ax.set_ylabel("Count")
st.pyplot(fig)

# WordCloud for each sentiment
def plot_wordcloud(sentiment):
    sentiment_text = " ".join(data_balanced[data_balanced['sentiment'] == sentiment]['text'])
    wordcloud = WordCloud(width=800, height=400, background_color='white').generate(sentiment_text)
    plt.figure(figsize=(10, 5))
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis('off')
    plt.title(f"WordCloud for {sentiment} Sentiment", fontsize=16)
    st.pyplot(plt)

st.subheader("WordClouds by Sentiment")
st.write("Positive Sentiment")
plot_wordcloud("positive")
st.write("Neutral Sentiment")
plot_wordcloud("neutral")
st.write("Negative Sentiment")
plot_wordcloud("negative")

# Splitting Data
st.subheader("Training Sentiment Classification Models")
X = data_balanced['text']
y = data_balanced['sentiment']

tfidf = TfidfVectorizer(max_features=5000)
X_tfidf = tfidf.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(X_tfidf, y, test_size=0.2, random_state=42)

# Models and Evaluation
models = {
    "Naive Bayes": MultinomialNB(),
    "Logistic Regression": LogisticRegression(max_iter=1000),
    "Support Vector Machine": SVC(kernel='linear')
}

results = {}

for name, model in models.items():
    st.write(f"Training {name}...")
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred, output_dict=True)
    results[name] = {
        "Accuracy": accuracy,
        "Precision": report['weighted avg']['precision'],
        "Recall": report['weighted avg']['recall'],
        "F1-Score": report['weighted avg']['f1-score']
    }
    st.write(f"{name} Classification Report")
    st.text(classification_report(y_test, y_pred))

# Comparison Table
st.subheader("Model Comparison")
results_df = pd.DataFrame(results).T
st.dataframe(results_df)

# Visualization of Model Performance
st.subheader("Model Performance")
fig, ax = plt.subplots()
results_df[['Accuracy', 'Precision', 'Recall', 'F1-Score']].plot(kind='bar', ax=ax)
ax.set_title("Model Performance Metrics")
ax.set_xlabel("Model")
ax.set_ylabel("Scores")
plt.xticks(rotation=45)
st.pyplot(fig)

# User Input for Prediction
st.subheader("Predict Sentiment for Custom Text")
user_input = st.text_area("Enter text:")
model_choice = st.selectbox("Choose a model", list(models.keys()))
if st.button("Predict Sentiment"):
    chosen_model = models[model_choice]
    input_tfidf = tfidf.transform([user_input])
    prediction = chosen_model.predict(input_tfidf)[0]
    st.write(f"Predicted Sentiment: {prediction}")
