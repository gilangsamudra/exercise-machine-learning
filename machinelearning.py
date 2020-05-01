import json
import random
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn import svm
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score


class Sentiment:
    Negative = 'NEGATIVE'
    Neutral = 'NEUTRAL'
    Positive = 'POSITIVE'


class Review:
    def __init__(self, text, score):
        self.text = text
        self.score = score
        self.sentiment = self.get_sentiment()

    def get_sentiment(self):
        if self.score <= 2:
            return Sentiment.Negative
        elif self.score == 3:
            return Sentiment.Neutral
        else:
            return Sentiment.Positive


class ReviewContainer:
    def __init__(self, reviews):
        self.reviews = reviews

    def get_text(self):
        return [x.text for x in self.reviews]

    def get_sentiment(self):
        return [x.sentiment for x in self.reviews]

    def evenly_distribute(self):
        negative = list(filter(lambda x: x.sentiment == Sentiment.Negative, self.reviews))
        positive = list(filter(lambda x: x.sentiment == Sentiment.Positive, self.reviews))
        positive_shrunk = positive[:len(negative)]
        self.reviews = negative + positive_shrunk
        random.shuffle(self.reviews)


file_name = 'D:\Phyton Code\Contoh dari Github\sklearn-master\data\sentiment\Books_small_10000.json'

comment = []
with open(file_name) as f:
    for line in f:
        review = json.loads(line)
        comment.append(Review(review['reviewText'], review['overall']))

print(comment[5].sentiment)


# Prep Data
# split the data into training and test data
training, test = train_test_split(comment, test_size=0.33, random_state=42)
print(training[0].text)

# filter the number of training data to be evenly distributed
train_container = ReviewContainer(training)
test_container = ReviewContainer(test)

train_container.evenly_distribute()
test_container.evenly_distribute()

train_x = train_container.get_text()
train_y = train_container.get_sentiment()

test_x = test_container.get_text()
test_y = test_container.get_sentiment()

print(train_y.count(Sentiment.Positive))
print(train_y.count(Sentiment.Negative))


# Bag of Words Vectorization
count_vect = CountVectorizer()
train_x_vector = count_vect.fit_transform(train_x)
print(train_x[669])
print(train_x_vector[669].toarray())

test_x_vector = count_vect.transform(test_x)

# Classifier - sklearn classifier
# Linear SVM
clf_svm = svm.SVC(kernel='linear')
clf_svm.fit(train_x_vector, train_y)
clf_svm.predict(test_x_vector[0])

# Decision Tree
clf_dec = DecisionTreeClassifier()
clf_dec.fit(train_x_vector, train_y)
clf_dec.predict(test_x_vector[0])

# Naive Bayes
clf_gnb = GaussianNB()
clf_gnb.fit(train_x_vector.toarray(), train_y)
clf_gnb.predict(test_x_vector[0].toarray())

# Logistic Regression
clf_lg = LogisticRegression()
clf_lg.fit(train_x_vector.toarray(), train_y)
clf_lg.predict(test_x_vector[0].toarray())


# Evaluation
# Mean Accuracy
print(clf_svm.score(test_x_vector, test_y))
print(clf_dec.score(test_x_vector, test_y))
print(clf_gnb.score(test_x_vector.toarray(), test_y))
print(clf_lg.score(test_x_vector, test_y))

# F1 Scores
print(f1_score(test_y, clf_svm.predict(test_x_vector), average=None, labels=[Sentiment.Positive, Sentiment.Negative]))
print(f1_score(test_y, clf_dec.predict(test_x_vector), average=None, labels=[Sentiment.Positive, Sentiment.Negative]))
print(f1_score(test_y, clf_gnb.predict(test_x_vector.toarray()), average=None, labels=[Sentiment.Positive, Sentiment.Negative]))
print(f1_score(test_y, clf_lg.predict(test_x_vector), average=None, labels=[Sentiment.Positive, Sentiment.Negative]))


# Test the model
test_set = ['I thoroughly enjoyed this, 5 stars', 'bad book do not buy', 'horrible waste of time']
new_test = count_vect.transform(test_set)

clf_svm.predict(new_test)
