from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report
import joblib
from preprocess import load_data, split_data, train_test_split_data

# Correct path to the spam.csv file
file_path = r'C:\projects\Spam\data\spam.csv'

# Load and preprocess the data
df = load_data(file_path)
X, y = split_data(df)
X_train, X_test, y_train, y_test = train_test_split_data(X, y)

# Vectorize the text data
vectorizer = TfidfVectorizer(stop_words='english')
X_train_tfidf = vectorizer.fit_transform(X_train)
X_test_tfidf = vectorizer.transform(X_test)

# Train the Naive Bayes classifier
model = MultinomialNB()
model.fit(X_train_tfidf, y_train)

# Make predictions on the test set
y_pred = model.predict(X_test_tfidf)

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
print(f'Accuracy: {accuracy:.4f}')
print("Classification Report:")
print(classification_report(y_test, y_pred))

# Save the trained model and vectorizer
joblib.dump(model, r'C:\projects\Spam\models\spam_classifier_model.pkl')  # Path to save the model
joblib.dump(vectorizer, r'C:\projects\Spam\models\vectorizer.pkl')  # Path to save the vectorizer
