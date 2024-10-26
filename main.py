"""
This NLP project analyzes the Sentiment(positive, negative and neutral) of reviews/comments given by users/consumers/buyers.
NLTK is used to compute the positive/negative/neutral orientation of the reviews/comments.
Data Visualization is done using 'matplotlib'
"""
# imports
import matplotlib.pyplot as plt
from nltk.sentiment.vader import SentimentIntensityAnalyzer

# initialization
sid = SentimentIntensityAnalyzer()

# counters
positive_count = 0
negative_count = 0
neutral_count = 0

# computing scores
with open('reviews.txt', encoding='ISO-8859-2') as f:
    for text in f.read().split('\n'):
        if text:
            scores = sid.polarity_scores(text)
            compound_score = scores['compound']

            if compound_score >= 0.05:
                positive_count += 1 # positive scores
            elif compound_score <= -0.05:
                negative_count += 1 # negative scores
            else:
                neutral_count += 1 # neutral scores

total_reviews = positive_count + negative_count + neutral_count # total reviews

# printing summary
print("\nSummary:")
print("Total Reviews:", total_reviews)
print("Positive Reviews:", positive_count, "("+str(round((positive_count / total_reviews) * 100))+"%)")
print("Negative Reviews:", negative_count, "("+str(round((negative_count / total_reviews) * 100))+"%)")
print("Neutral Reviews:", neutral_count, "("+str(round((neutral_count / total_reviews) * 100))+"%)")


# visualization using 'pie chart'
labels = 'Positive', 'Negative', 'Neutral' # labels of slices
sizes = [positive_count, negative_count, neutral_count] # size values of slices
colors = ['#00BFFF', '#87CEEB', '#B0C4DE']  # colours for the pies
explode = (0.1, 0, 0) # exploding the first slice

plt.figure(figsize=(8, 6), facecolor='#ADD8E6') # background size and colour
plt.pie(
    sizes,
    explode=explode,
    labels=labels,
    colors=colors,
    autopct='%1.1f%%', # to display (%) of slices
    shadow=True,
    startangle=140 # to display the pie at an angle
    )
plt.axis('equal') # to display pie in circular shape
plt.title('Sentiment Analysis of Kindle Reviews', fontsize=14)
plt.show()