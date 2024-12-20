import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.pipeline import Pipeline
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report, accuracy_score
from nltk.corpus import stopwords
import nltk

# Download NLTK stopwords
def download_stopwords():
    try:
        nltk.data.find('corpora/stopwords')
    except LookupError:
        nltk.download('stopwords')
download_stopwords()

# Example dataset (replace with your own data)
data = {
    'Email': [
        "Congratulations! You have won a lottery.",
        "Reminder: Your appointment is scheduled for tomorrow.",
        "Claim your free prize now!",
        "Meeting agenda for next week.",
        "Exclusive offer just for you!",
        "Please find the attached invoice."
    ],
    'Label': ['spam', 'not spam', 'spam', 'not spam', 'spam', 'not spam']
}

# Load the dataset
df = pd.DataFrame(data)

# Train-test split
X = df['Email']
y = df['Label']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Create a pipeline for preprocessing and classification
pipeline = Pipeline([
    ('vectorizer', CountVectorizer(stop_words=stopwords.words('english'))),
    ('tfidf', TfidfTransformer()),
    ('classifier', MultinomialNB()),
])

# Train the model
pipeline.fit(X_train, y_train)

# Make predictions
y_pred = pipeline.predict(X_test)

# Evaluate the model
print("Classification Report:\n", classification_report(y_test, y_pred))
print("Accuracy:", accuracy_score(y_test, y_pred))

# Example of predicting new emails
new_emails = [
    "Congratulations, you've won a free trip to the Bahamas!",
    "Don't forget about the team meeting tomorrow."
]
predictions = pipeline.predict(new_emails)
for email, label in zip(new_emails, predictions):
    print(f"Email: '{email}' -> Predicted Label: {label}")
