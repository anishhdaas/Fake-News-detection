📰 Fake News Detection using Machine Learning

A supervised Natural Language Processing (NLP) project that classifies news articles as Fake or Real using multiple machine learning algorithms and TF-IDF feature engineering.

📌 Project Overview

This project builds an end-to-end text classification pipeline to detect fake news articles. It includes:

Text preprocessing and cleaning

Stopword removal using NLTK

TF-IDF vectorization

Training multiple classification models

Performance evaluation and visualization

The objective is to compare different ML algorithms for binary text classification on high-dimensional textual data.

🛠 Tech Stack

Python

Pandas

NLTK

Scikit-learn

Matplotlib

Seaborn

📂 Dataset

The dataset consists of two CSV files:

Fake.csv – Contains fake news articles

True.csv – Contains real news articles

Both files include a text column used for training and evaluation.

⚙️ Workflow
1️⃣ Data Preprocessing

Convert text to lowercase

Remove URLs

Remove non-alphabetic characters

Remove extra whitespaces

Remove English stopwords

2️⃣ Feature Engineering

TF-IDF Vectorization (max_features=5000)

3️⃣ Model Training

The following classifiers were implemented:

Logistic Regression

Multinomial Naive Bayes

Support Vector Machine (LinearSVC)

4️⃣ Model Evaluation

Accuracy Score

Precision, Recall, F1-Score

Confusion Matrix

Visual comparison of model performance

📊 Results

All models achieved strong classification performance.

Logistic Regression and SVM performed particularly well on sparse TF-IDF features.

Confusion matrices provided insight into false positives and false negatives.

🚀 Installation & Usage
1️⃣ Clone the Repository
git clone https://github.com/your-username/fake-news-detection.git
cd fake-news-detection
2️⃣ Install Dependencies
pip install pandas nltk scikit-learn matplotlib seaborn
3️⃣ Download NLTK Stopwords
import nltk
nltk.download('stopwords')
4️⃣ Run the Script
python fake_news_detection.py
📈 Sample Output

Classification reports for each model

Confusion matrix heatmaps

Accuracy comparison bar chart

📌 Key Learnings

Practical NLP preprocessing pipeline

Feature extraction using TF-IDF

Comparative analysis of classification models

Evaluation of high-dimensional text data

Visualization of model performance

🔮 Future Improvements

Hyperparameter tuning (GridSearchCV)

Cross-validation

Deep learning models (LSTM / BERT)

Deployment using Flask or FastAPI

Real-time prediction interface
