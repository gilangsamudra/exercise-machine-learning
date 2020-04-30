import json
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer


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
print(train_x_vector[0].toarray())
