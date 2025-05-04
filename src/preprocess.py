import pandas as pd
from sklearn.model_selection import train_test_split

# Load the dataset
def load_data(file_path):
    df = pd.read_csv(file_path, encoding='latin-1')  # Adjust encoding if needed
    df = df[['v1', 'v2']]  # Assuming the dataset has these columns (label and text)
    df.columns = ['label', 'text']  # Rename columns for clarity
    df = df.dropna()  # Drop any rows with missing values
    return df

# Split the data into features (X) and labels (y)
def split_data(df):
    X = df['text']
    y = df['label'].map({'spam': 1, 'ham': 0})  # Convert 'spam' -> 1, 'ham' -> 0
    return X, y

# Split into train and test sets
def train_test_split_data(X, y, test_size=0.2):
    return train_test_split(X, y, test_size=test_size, random_state=42)

if __name__ == "__main__":
    # Correct path to the spam.csv file
    file_path = r'C:\projects\Spam\data\spam.csv'

    # Load the data from the given path
    df = load_data(file_path)

    # Split the data into features and labels
    X, y = split_data(df)

    # Split into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split_data(X, y)

    # Print some basic info to ensure everything looks good
    print(f"Training data: {X_train.shape[0]} samples")
    print(f"Test data: {X_test.shape[0]} samples")
