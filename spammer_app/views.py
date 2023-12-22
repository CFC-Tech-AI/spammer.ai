import joblib
from django.shortcuts import render
from django.http import HttpResponseRedirect
from .models import UserEmail
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer


nb_model = joblib.load('spammer_app/spam_classifier_model.pkl')
vectorizer = joblib.load('spammer_app/email_vectorizer.pkl')


def preprocess_email(email):    
    email = email.lower()   
    tokens = word_tokenize(email)
    tokens = [PorterStemmer().stem(token) for token in tokens if token.isalpha() and token not in stopwords.words('english')]
    return ' '.join(tokens)

def index(request):
    if request.method == 'POST':
        subject = request.POST.get('subject', '')
        message = request.POST.get('message', '')
        UserEmail.objects.create(subject=subject, message=message)
        is_spam = classify_email(message)
        return render(request, 'result.html', {'is_spam': is_spam})
    return render(request, 'index.html')

def classify_email(message):    
    processed_message = preprocess_email(message)    
    message_vectorized = vectorizer.transform([processed_message])    
    prediction = nb_model.predict(message_vectorized)
    return bool(prediction[0])
