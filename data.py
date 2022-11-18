import json
import string
import random 
import nltk
import numpy as np
import speak
import datetime
from nltk.stem import WordNetLemmatizer 
import tensorflow as tf 

nltk.download('omw-1.4')

from keras.models import Sequential
from keras.layers import Dense, Dropout

# from tensorflow.keras import Sequential
# from tensorflow.keras.layers import Dense, Dropout

set_time = 0;

nltk.download("punkt")
nltk.download("wordnet")

lemmatizer = WordNetLemmatizer()

with open("data_w.json") as file:
    data = json.load(file)

words = []
classes = []
doc_X = []
doc_Y = []

for intent in data["intents"]:
    for pattern in intent["patterns"]:
        tokens = nltk.word_tokenize(pattern)
        words.extend(tokens)
        doc_X.append(pattern)
        doc_Y.append(intent["tag"])
    
    if intent["tag"] not in classes:
        classes.append(intent["tag"])

words = [lemmatizer.lemmatize(word.lower())\
    for word in words if word not in string.punctuation]
words = sorted(set(words))
classes = sorted(set(classes))

training = []
out_empty = [0] * len(classes)

# création du modèle d'ensemble de mots

for idx, doc in enumerate(doc_X):
    bow = []
    text = lemmatizer.lemmatize(doc.lower())
    for word in words:
        bow.append(1) if word in text else bow.append(0)

    # marque l'index de la classe à laquelle le pattern atguel est associé à
    
    output_row = list(out_empty)
    output_row[classes.index(doc_Y[idx])] = 1
    # ajoute le one hot encoded BoW et les classes associées à la liste training
    training.append([bow, output_row])
# mélanger les données et les convertir en array
random.shuffle(training)
training = np.array(training, dtype=object)
# séparer les features et les labels target
train_X = np.array(list(training[:, 0]))
train_Y = np.array(list(training[:, 1]))

input_shape = (len(train_X[0]),)
output_shape = len(train_Y[0])
epochs = 200

model = Sequential()
model.add(Dense(128, input_shape=input_shape, activation="relu"))
model.add(Dropout(0.5))
model.add(Dense(64, activation="relu"))
model.add(Dropout(0.3))
model.add(Dense(output_shape, activation = "softmax"))
adam = tf.keras.optimizers.Adam(learning_rate=0.01, decay=1e-6)
model.compile(loss='categorical_crossentropy', optimizer=adam, metrics=["accuracy"])

print(model.summary())

model.fit(x = train_X, y = train_Y, epochs = 200, verbose = 1)

def clean_text(text): 
	tokens = nltk.word_tokenize(text)
	tokens = [lemmatizer.lemmatize(word) for word in tokens]
	return tokens

def bag_of_words(text, vocab): 
	tokens = clean_text(text)
	bow = [0] * len(vocab)
	for w in tokens: 
		for idx, word in enumerate(vocab):
			if word == w: 
				bow[idx] = 1
	return np.array(bow)

def pred_class(text, vocab, labels): 
	bow = bag_of_words(text, vocab)
	result = model.predict(np.array([bow]))[0]
	thresh = 0.2
	y_pred = [[idx, res] for idx, res in enumerate(result) if res > thresh]
	y_pred.sort(key=lambda x: x[1], reverse=True)
	return_list = []
	for r in y_pred:
		return_list.append(labels[r[0]])
	return return_list

def get_response(intents_list, intents_json): 
	tag = intents_list[0]
	list_of_intents = intents_json["intents"]
	for i in list_of_intents: 
		if i["tag"] == tag:
			result = random.choice(i["responses"])
			break
	return result

def get_mess(client):
	@client.event
	async def on_message(message):
		intents = pred_class(message.content, words, classes)
		if message.author == client.user:
			return 
		time_obj = int(str(datetime.datetime.now())[11:19].replace(':', ""))
		global set_time
		print(time_obj - set_time)
		if time_obj - set_time < 100:
			await message.channel.send(get_response(intents, data))
		elif message.content.lower().startswith('kentin')\
			or message.content.startswith("<@1042421290407047168>"):
			set_time = time_obj;
			if (len(message.content) == 6 or message.content == ("<@1042421290407047168>")):
				await message.channel.send('Quoi ?')
			else:
			 	await message.channel.send(get_response(intents, data))
		pass_true = 1;
