import math
from typing import List, Dict, Union
import os
from utils.sentiment_detection import read_tokens, load_reviews, split_data
from exercises.tick1 import accuracy, predict_sentiment, read_lexicon

'''
Developing a Naive Bayes Classifier on all words in a given text, comparing results with and without smoothing.
'''

def calculate_class_log_probabilities(training_data: List[Dict[str, Union[List[str], int]]]) \
        -> Dict[int, float]:
    """
    Calculate the prior class probability P(c).

    @param training_data: list of training instances, where each instance is a dictionary with two fields: 'text' and
        'sentiment'. 'text' is the tokenized review and 'sentiment' is +1 or -1, for positive and negative sentiments.
    @return: dictionary from sentiment to prior log probability
    """
    # Initialise the count of each sentiment
    sentiment_count = {1:0, -1:0}

    # Count the number of instances of pos and neg sentiment
    for instance in training_data:
        sentiment = instance['sentiment']
        sentiment_count[sentiment] += 1

    n_instances = len(training_data)
    class_log_probabilities = {}

    # Calculate prior log probability from sentiment
    for sentiment, count in sentiment_count.items():
        class_log_probabilities[sentiment] = math.log(count / n_instances)

    return class_log_probabilities


def calculate_unsmoothed_log_probabilities(training_data: List[Dict[str, Union[List[str], int]]]) \
        -> Dict[int, Dict[str, float]]:
    """
    Calculate the unsmoothed log likelihood log (P(x|c)) of a word in the vocabulary given a sentiment.

    @param training_data: list of training instances, where each instance is a dictionary with two fields: 'text' and
        'sentiment'. 'text' is the tokenized review and 'sentiment' is +1 or -1, for positive and negative sentiments.
    @return: dictionary from sentiment to dictionary of tokens with respective log probability
    """
    word_count = {1:{}, -1:{}}
    n_words = {1:0, -1:0}
    word_log_probabilities = {1:{}, -1:{}}

    # Count the number of occurrences of each word appearing with a particular class label
    for instance in training_data:
        sentiment = instance["sentiment"]
        for word in instance['text']:
            word_count[sentiment].setdefault(word, 0)
            word_count[sentiment][word] += 1
            n_words[sentiment] += 1

    # Calculate the log probability of each word given a sentiment
    for sentiment in [-1, 1]:
        for word in word_count[sentiment]:
            probability = word_count[sentiment][word] / n_words[sentiment]
            word_log_probabilities[sentiment][word] = math.log(probability)

    return word_log_probabilities


def calculate_smoothed_log_probabilities(training_data: List[Dict[str, Union[List[str], int]]]) \
        -> Dict[int, Dict[str, float]]:
    """
    Calculate the smoothed log likelihood log (P(x|c)) of a word in the vocabulary given a sentiment. Use the smoothing
    technique described in the instructions (Laplace smoothing).

    @param training_data: list of training instances, where each instance is a dictionary with two fields: 'text' and
        'sentiment'. 'text' is the tokenized review and 'sentiment' is +1 or -1, for positive and negative sentiments.
    @return: Dictionary from sentiment to Dictionary of tokens with respective log probability
    """
    # Smoothing - assume that each word has been seen at least once with each sentiment
    word_count = {1:{}, -1:{}}
    n_words = {1:0, -1:0}
    word_log_probabilities = {1:{}, -1:{}}
    vocabulary = set()

    for instance in training_data:
        sentiment = instance["sentiment"]
        for word in instance['text']:
            vocabulary.add(word)
            word_count[sentiment].setdefault(word, 0)
            word_count[sentiment][word] += 1
            n_words[sentiment] += 1

    total_word_count = {1: n_words[1] + len(vocabulary), -1: n_words[-1] + len(vocabulary)}

    for sentiment in [-1, 1]:
        for word in vocabulary:
            word_count[sentiment].setdefault(word, 0)
            word_count[sentiment][word] += 1
        for word in word_count[sentiment]:
            probability = (word_count[sentiment][word] + 1)/ total_word_count[sentiment]
            word_log_probabilities[sentiment][word] = math.log(probability)

    return word_log_probabilities


def predict_sentiment_nbc(review: List[str], log_probabilities: Dict[int, Dict[str, float]],
                          class_log_probabilities: Dict[int, float]) -> int:
    """
    Use the estimated log probabilities to predict the sentiment of a given review.

    @param review: a single review as a list of tokens
    @param log_probabilities: dictionary from sentiment to Dictionary of tokens with respective log probability
    @param class_log_probabilities: dictionary from sentiment to prior log probability
    @return: predicted sentiment [-1, 1] for the given review
    """
    total_log_probabilities = {1: class_log_probabilities[1], -1: class_log_probabilities[-1]}

    for token in review:
        for sentiment in [1, -1]:
            if token in log_probabilities[sentiment]:
                total_log_probabilities[sentiment] += log_probabilities[sentiment][token]

    predicted_sentiment = 1 if total_log_probabilities[1] > total_log_probabilities[-1] else -1

    return predicted_sentiment


def main():
    """
    Code to check your work locally (run this from the root directory, 'mlrd/')
    """
    review_data = load_reviews(os.path.join('data', 'sentiment_detection', 'reviews'))
    training_data, validation_data = split_data(review_data, seed=0)
    train_tokenized_data = [{'text': read_tokens(x['filename']), 'sentiment': x['sentiment']} for x in training_data]
    dev_tokenized_data = [read_tokens(x['filename']) for x in validation_data]
    validation_sentiments = [x['sentiment'] for x in validation_data]

    lexicon = read_lexicon(os.path.join('data', 'sentiment_detection', 'sentiment_lexicon'))

    preds_simple = []
    for review in dev_tokenized_data:
        pred = predict_sentiment(review, lexicon)
        preds_simple.append(pred)

    acc_simple = accuracy(preds_simple, validation_sentiments)
    print(f"Your accuracy using simple classifier: {acc_simple}")

    class_priors = calculate_class_log_probabilities(train_tokenized_data)
    unsmoothed_log_probabilities = calculate_unsmoothed_log_probabilities(train_tokenized_data)
    preds_unsmoothed = []
    for review in dev_tokenized_data:
        pred = predict_sentiment_nbc(review, unsmoothed_log_probabilities, class_priors)
        preds_unsmoothed.append(pred)

    acc_unsmoothed = accuracy(preds_unsmoothed, validation_sentiments)
    print(f"Your accuracy using unsmoothed probabilities: {acc_unsmoothed}")

    smoothed_log_probabilities = calculate_smoothed_log_probabilities(train_tokenized_data)
    preds_smoothed = []
    for review in dev_tokenized_data:
        pred = predict_sentiment_nbc(review, smoothed_log_probabilities, class_priors)
        preds_smoothed.append(pred)

    acc_smoothed = accuracy(preds_smoothed, validation_sentiments)
    print(f"Your accuracy using smoothed probabilities: {acc_smoothed}")


if __name__ == '__main__':
    main()
