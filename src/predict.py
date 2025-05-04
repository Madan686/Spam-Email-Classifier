import joblib

# Load the trained model and vectorizer
model = joblib.load(r'C:\projects\Spam\models\spam_classifier_model.pkl')
vectorizer = joblib.load(r'C:\projects\Spam\models\vectorizer.pkl')

# Function to predict if an email is spam or ham
def predict_email(email_text):
    # Convert the input email to the same vector format as the training data
    email_tfidf = vectorizer.transform([email_text])
    
    # Make a prediction (0 = ham, 1 = spam)
    prediction = model.predict(email_tfidf)
    
    # Return a human-readable result
    if prediction == 1:
        return "The email is SPAM."
    else:
        return "The email is HAM."

# Main function to take user input
if __name__ == "__main__":
    print("Welcome to the Spam Email Classifier!")
    
    # Take user input for the email message
    email_message = input("Please enter the email message to classify: ")
    
    # Classify the email
    result = predict_email(email_message)
    
    # Output the classification result
    print(result)
