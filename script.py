import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix

# Import 'json' file and Discover datasets
new_york_tweets = pd.read_json("new_york.json", lines=True)
#print(len(new_york_tweets))
#print(new_york_tweets.columns)
#print(new_york_tweets.loc[12]["text"])
london_tweets = pd.read_json("london.json", lines=True)
paris_tweets = pd.read_json("paris.json", lines=True)
#print(len(london_tweets))
#print(len(paris_tweets))

# Get tweet's text
new_york_text = new_york_tweets["text"].tolist()
london_text = london_tweets["text"].tolist()
paris_text = paris_tweets["text"].tolist()

# Mix all dataset
all_tweets = new_york_text + london_text + paris_text
labels = [0] * len(new_york_text) + [1] * len(london_text) + [2] * len(paris_text)
train_data, test_data, train_labels, test_labels = train_test_split(all_tweets, labels, test_size = 0.2, random_state = 1)
#print(len(train_data))
#print(len(test_data))

counter = CountVectorizer()
counter.fit(train_data)
train_counts = counter.transform(train_data)
test_counts = counter.transform(test_data)

# Train and Test the Naive Bayes Classifier
classifier = MultinomialNB()
classifier.fit(train_counts, train_labels)
predictions = classifier.predict(test_counts)

# Evaluating Your Model

acc_score= accuracy_score(test_labels, predictions)
print(acc_score)

# Confusion matrix

confusion_matrix_value = confusion_matrix(test_labels, predictions)
print(confusion_matrix_value)

# Test Your Own Tweet
tweet = "The weather is pretty bad."
tweet_counts = counter.transform([tweet])
tweet_predict = classifier.predict(tweet_counts)

if tweet_predict == 0:
    print("Location : New York")
elif tweet_predict == 1:
    print("Location : London")
else:
    print("Location : Paris")