# 📧 Spam Email Classifier

A machine learning project that detects and classifies emails as **Spam** or **Not Spam** using a **Naive Bayes classifier** trained on the SMS Spam Collection dataset.

## 🚀 Problem Statement

Email spam is a major issue that clutters inboxes and exposes users to potential scams. The goal is to create a reliable model that classifies messages based on their content.

## ✅ Proposed System

- Preprocesses SMS/email text
- Extracts features using `CountVectorizer`
- Trains a Naive Bayes model
- Evaluates performance and provides predictions

## 🔍 System Approach

1. **Data Preprocessing**
2. **Train/Test Split**
3. **Feature Extraction**
4. **Model Training**
5. **Evaluation**
6. **Prediction via user input**

## 🤖 Algorithm

- **Multinomial Naive Bayes**:
  A probabilistic learning method commonly used for text classification, especially effective in spam detection.

## 🛠️ Tech Stack

- Python
- Pandas
- Scikit-learn
- Pickle

## 📊 Results

- **Accuracy**: 96.68%
- **Precision** (Spam): 1.00
- **Recall** (Spam): 0.75
- **F1-score** (Spam): 0.86

## 💡 Future Scope

- Integrate with email clients or web apps
- Use NLP techniques like TF-IDF or BERT
- Deploy via Flask/Django web interface
- Add language support for non-English messages

## 📦 How to Run

1. Clone the repository:
   ```bash
   git clone https://github.com/Madan686/Spam-Email-Classifier.git
   cd Spam-Email-Classifier
