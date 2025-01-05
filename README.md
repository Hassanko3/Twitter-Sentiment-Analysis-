# Twitter Sentiment Analysis

This project focuses on classifying tweets into **positive**, **neutral**, or **negative** sentiments using machine learning models. It provides an interactive interface for custom sentiment prediction and detailed visualizations of linguistic patterns.

---

## Features

- **Custom Sentiment Prediction**: Input text and get real-time predictions using Naïve Bayes, Logistic Regression, or SVM.
- **Data Visualization**: WordClouds and sentiment distribution charts to understand patterns in the dataset.
- **Model Comparison**: Evaluation of models with metrics like accuracy, precision, recall, and F1-score.
- **Interactive Interface**: Built using **Streamlit** for a user-friendly experience.

---

 Dataset

The dataset contains tweets labeled with sentiments:
-  Positive 
-  Neutral
-  Negative

  Dataset File
- File name: `twee_semantic.csv`

---

 Installation and Setup

Follow these steps to set up the project locally:

1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/twitter-sentiment-analysis.git
   cd twitter-sentiment-analysis

   pip install -r requirements.txt
streamlit run app.py
Model Overview
Three machine learning models are used:

Naïve Bayes: Efficient for text classification, but struggles with complex patterns.
Logistic Regression: Balances simplicity and performance.
Support Vector Machine (SVM): Best for high-dimensional data with high accuracy.
<img width="595" alt="image" src="https://github.com/user-attachments/assets/5469ac17-5e6c-4bd3-8729-5a32b02f4819" />

Developed by [Hassan].
For queries, feel free to contact me hasanmohamednoor2002@gmail.com

