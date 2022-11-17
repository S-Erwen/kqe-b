import json
import string
import random 
import nltk
import numpy as np
from nltk.stem import WordNetLemmatizer 
import tensorflow as tf 
import data_w

# from keras.models import Sequential
# from keras.layers import Dense, Dropout

from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense, Dropout
nltk.download("punkt")
nltk.download("wordnet")

lemmatizer = WordNetLemmatizer()

words = []
classes = []
doc_X = []
doc_y = []

for intent in data_w["intents"]:
    for pattern in intent["patterns"]:
        tokens = nltk.word_tokenize(pattern)
        words.extend(tokens)
        doc_X.append(pattern)
        doc_y.append(intent["tag"])
    
    if intent["tag"] not in classes:
        classes.append(intent["tag"])

words = [lemmatizer.lemmatize(word.lower())\
    for word in words if word not in string.punctuation]
words = sorted(set(words))
classes = sorted(set(classes))

print(words)
print(classes)
print(doc_X)
