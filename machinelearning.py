import json


class Sentiment:
    Negative = 'Negative'
    Neutral = 'Neutral'
    Positive = 'Positive'


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
