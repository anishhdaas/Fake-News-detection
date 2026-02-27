# 📦 Import libraries
import pandas as pd
import re
import nltk
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# 📥 Download NLTK stopwords
nltk.download('stopwords')
from nltk.corpus import stopwords
stop_words = set(stopwords.words('english'))

# 🧹 Clean the text
def clean_text(text):
    text = text.lower()
    text = re.sub(r"http\S+", "", text)
    text = re.sub(r"[^a-zA-Z]", " ", text)
    text = re.sub(r"\s+", " ", text)
    text = " ".join([word for word in text.split() if word not in stop_words])
    return text

# 🗂️ Load data
df_fake = pd.read_csv("/Users/ishaan/Python/Fake News Detection/Fake.csv")
df_real = pd.read_csv("/Users/ishaan/Python/Fake News Detection/True.csv")
df_fake["label"] = 0
df_real["label"] = 1

# 👯 Combine and clean
df = pd.concat([df_fake, df_real])[["text", "label"]].sample(frac=1).reset_index(drop=True)
df["text"] = df["text"].apply(clean_text)

# ✂️ Split data
X_train, X_test, y_train, y_test = train_test_split(df["text"], df["label"], test_size=0.2, random_state=42)

# 🔠 TF-IDF vectorization
vectorizer = TfidfVectorizer(max_features=5000)
X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)

# 📊 Store model results
results = {}
conf_matrices = {}

# 🔹 Logistic Regression
lr = LogisticRegression()
lr.fit(X_train_vec, y_train)
lr_pred = lr.predict(X_test_vec)
results["Logistic Regression"] = accuracy_score(y_test, lr_pred)
conf_matrices["Logistic Regression"] = confusion_matrix(y_test, lr_pred)
print("🔹 Logistic Regression")
print(classification_report(y_test, lr_pred))

# 🔹 Naive Bayes
nb = MultinomialNB()
nb.fit(X_train_vec, y_train)
nb_pred = nb.predict(X_test_vec)
results["Naive Bayes"] = accuracy_score(y_test, nb_pred)
conf_matrices["Naive Bayes"] = confusion_matrix(y_test, nb_pred)
print("\n🔹 Naive Bayes")
print(classification_report(y_test, nb_pred))

# 🔹 SVM
svm = LinearSVC()
svm.fit(X_train_vec, y_train)
svm_pred = svm.predict(X_test_vec)
results["SVM"] = accuracy_score(y_test, svm_pred)
conf_matrices["SVM"] = confusion_matrix(y_test, svm_pred)
print("\n🔹 Support Vector Machine (SVM)")
print(classification_report(y_test, svm_pred))

# 📈 Plotting confusion matrices
for model_name, matrix in conf_matrices.items():
    plt.figure(figsize=(5, 4))
    sns.heatmap(matrix, annot=True, fmt='d', cmap='Blues', xticklabels=["Fake", "Real"], yticklabels=["Fake", "Real"])
    plt.title(f"Confusion Matrix: {model_name}")
    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")
    plt.tight_layout()
    plt.show()

# 📊 Compare accuracy scores
plt.figure(figsize=(7, 5))
sns.barplot(x=list(results.keys()), y=list(results.values()), palette="viridis")
plt.title("Model Accuracy Comparison")
plt.ylabel("Accuracy")
plt.ylim(0.8, 1.0)
plt.xticks(rotation=10)
plt.tight_layout()
plt.show()
