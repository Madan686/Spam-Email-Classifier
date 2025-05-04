import joblib
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from preprocess import load_data, split_data, train_test_split_data
from sklearn.feature_extraction.text import TfidfVectorizer

# Correct path to the spam.csv file
file_path = r'C:\projects\Spam\data\spam.csv'

# Load and preprocess the data
df = load_data(file_path)
X, y = split_data(df)
X_train, X_test, y_train, y_test = train_test_split_data(X, y)

# Load the trained model and vectorizer
model = joblib.load(r'C:\projects\Spam\models\spam_classifier_model.pkl')  # Path to the trained model
vectorizer = joblib.load(r'C:\projects\Spam\models\vectorizer.pkl')  # Path to the vectorizer

# Transform the test data using the loaded vectorizer
X_test_tfidf = vectorizer.transform(X_test)

# Make predictions on the test set
y_pred = model.predict(X_test_tfidf)

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
print(f'Accuracy: {accuracy:.4f}')

# Classification report (precision, recall, F1-score)
print("\nClassification Report:")
print(classification_report(y_test, y_pred))

# Confusion Matrix
print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))
