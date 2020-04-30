import json
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


file_name = 'D:\Phyton Code\Contoh dari Github\sklearn-master\data\sentiment\Books_small.json'

comment = []
with open(file_name) as f:
    for line in f:
        review = json.loads(line)
        comment.append(Review(review['reviewText'], review['overall']))

print(comment[5].sentiment)


# Prep Data
training, test = train_test_split(comment, test_size=0.33, random_state=42)
print(training[0].text)

train_x = [x.text for x in training]
train_y = [x.sentiment for x in training]
print(train_y[0])

test_x = [x.text for x in test]
test_y = [x.sentiment for x in test]
print(test_y[0])

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
print(f1_score(test_y, clf_svm.predict(test_x_vector), average=None, labels=[Sentiment.Positive, Sentiment.Neutral, Sentiment.Negative]))
print(f1_score(test_y, clf_dec.predict(test_x_vector), average=None, labels=[Sentiment.Positive, Sentiment.Neutral, Sentiment.Negative]))
print(f1_score(test_y, clf_gnb.predict(test_x_vector.toarray()), average=None, labels=[Sentiment.Positive, Sentiment.Neutral, Sentiment.Negative]))
print(f1_score(test_y, clf_lg.predict(test_x_vector), average=None, labels=[Sentiment.Positive, Sentiment.Neutral, Sentiment.Negative]))

# Istirahat dulu
