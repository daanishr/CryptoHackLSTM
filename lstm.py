import tflearn
from tflearn.data_utils import to_categorical, pad_sequences
from tflearn.datasets import imdb


import tweepy
from textblob import TextBlob
wiki = TextBlob("Daanish is very angry")
print(wiki.sentiment.polarity)

consumer_key = 'eofvfSE2IueXsMlbEAuylAxNL'
consumer_secret = 'Griv6wHexwCfGwWpifGacZfd4etoG3EVMCC10qFnLpVFUOeJmG'

access_token = '140024424-UrjXzn1nIkE7CkjXZGb37XxI6JAuRwve1awBqj1Q'
access_token_secret = 'bdKQwQB6OGPSGlvD9L3jKul68X8Xgvi4uY1XaqTreQPW3'

auth = tweepy.OAuthHandler(consumer_key, consumer_secret)
auth.set_access_token(access_token,access_token_secret)

api = tweepy.API(auth)

public_tweets = api.search('Ethereum')

#for tweet in public_tweets:
#    print(tweet.text)
#    analysis = TextBlob(tweet.text)
#    print(analysis.sentiment)
#    print('\n')

train, test, _ = imdb.load_data(path='imdb.pkl', n_words=10000,valid_portion=0.1)

trainX, trainY = train
testX, testY = test

trainX = pad_sequences(trainX, maxlen=100, value=0.)
testX = pad_sequences(testX, maxlen=100, value=0.)

#convert to vectors
trainY = to_categorical(trainY, nb_classes=2)
testY = to_categorical(testY, nb_classes=2)

#building network
net = tflearn.input_data([None, 100])
net = tflearn.embedding(net, input_dim=10000, output_dim=128)
net = tflearn.lstm(net, 128,dropout=0.8)
net = tflearn.fully_connected(net, 2, activation= 'softmax')
net = tflearn.regression(net, optimizer = 'adam', learning_rate=0.0001,loss ='categorical_crossentropy')

#Training data
model = tflearn.DNN(net, tensorboard_verbose=0)
model.fit(trainX, trainY, validation_set=0.1, show_metric=True, batch_size=32)