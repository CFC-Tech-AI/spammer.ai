
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
import joblib

def classify_email(message):
    # Load the pre-trained model
    model = joblib.load('spammer_app/spam_classifier_model.pkl')

    # Vectorize the input message
    vectorizer = CountVectorizer()
    message_vector = vectorizer.transform([message])

    # Make a prediction (0 for ham, 1 for spam)
    prediction = model.predict(message_vector)

    return bool(prediction[0])
