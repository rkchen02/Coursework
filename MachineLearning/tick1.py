from typing import List, Dict
import os
from utils.sentiment_detection import read_tokens, load_reviews

'''
Use a sentiment lexicon to classify a number of film reviews as positive or negative.
'''

def read_lexicon(filename: str) -> Dict[str, int]:
    """
    Read the lexicon from a given path.

    @param filename: path to file
    @return: dictionary from word to sentiment (+1 or -1 for positive or negative sentiments respectively).
    """
    lexicon = {}

    with open(filename, 'r') as file:
        for line in file:
            parts = line.strip().split()

            word = parts[0][5:]
            intensity = parts[1][10:]
            polarity = parts[2][9:]

            # Polarity can be positive or negative
            sentiment = 1 if polarity == "positive" else -1
            # Intensity can be weak or strong
            if intensity == "strong":
                sentiment *= 2

            lexicon[word] = sentiment

        return lexicon


def predict_sentiment(review: List[str], lexicon: Dict[str, int]) -> int:
    """
    Given a list of tokens from a tokenized review and a lexicon, determine whether the sentiment of each review in the
    test set is positive or negative based on whether there are more positive or negative words.

    @param review: list of tokens from tokenized review
    @param lexicon: dictionary from word to sentiment (+1 or -1 for positive or negative sentiments respectively).
    @return: calculated sentiment for each review (+1 or -1 for positive or negative sentiments respectively).
    """
    sentiment_score = sum(lexicon.get(word, 0) for word in review)
    return 1 if sentiment_score >= 0 else -1


def accuracy(pred: List[int], true: List[int]) -> float:
    """
    Calculate the proportion of predicted sentiments that were correct.

    @param pred: list of calculated sentiment for each review
    @param true: list of correct sentiment for each review
    @return: the overall accuracy of the predictions
    """
    # Sum all predictions that were same as the true sentiment
    correct = sum(1 for p, t in zip(pred, true) if p == t)
    # Sum all predictions that were different from the true sentiment
    incorrect = sum(1 for p, t in zip(pred, true) if p != t)
    return correct / (correct + incorrect) if (correct + incorrect) > 0 else 0


def predict_sentiment_improved(review: List[str], lexicon: Dict[str, int]) -> int:
    """
    Use the training data to improve your classifier, perhaps by choosing an offset for the classifier cutoff which
    works better than 0.

    @param review: list of tokens from tokenized review
    @param lexicon: dictionary from word to sentiment (+1 or -1 for positive or negative sentiments respectively).
    @return: calculated sentiment for each review (+1, -1 for positive and negative sentiments, respectively).
    """
    sentiment_score = sum(lexicon.get(word, 0) for word in review)
    # Choose 14 as the offset for the classifier cutoff
    return 1 if sentiment_score >= 14 else -1


def main():
    """
    Code to check your work locally (run this from the root directory, 'mlrd/')
    """
    review_data = load_reviews(os.path.join('data','sentiment_detection', 'reviews'))
    tokenized_data = [read_tokens(x['filename']) for x in review_data]

    lexicon = read_lexicon(os.path.join('data', 'sentiment_detection', 'sentiment_lexicon'))

    pred1 = [predict_sentiment(t, lexicon) for t in tokenized_data]
    acc1 = accuracy(pred1, [x['sentiment'] for x in review_data])
    print(f"Your accuracy: {acc1}")

    pred2 = [predict_sentiment_improved(t, lexicon) for t in tokenized_data]
    acc2 = accuracy(pred2, [x['sentiment'] for x in review_data])
    print(f"Your improved accuracy: {acc2}")


if __name__ == '__main__':
    main()
