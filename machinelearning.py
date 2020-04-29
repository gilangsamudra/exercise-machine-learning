import json

file_name = 'D:\Phyton Code\Contoh dari Github\sklearn-master\data\sentiment\Books_small.json'

comment = []
with open(file_name) as f:
    for line in f:
        review = json.loads(line)
        comment.append((review['reviewText'], review['overall']))
        print(comment)


comment[5]
