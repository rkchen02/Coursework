import os
from typing import List, Dict, Union
from utils.sentiment_detection import load_reviews, read_tokens, read_student_review_predictions, print_agreement_table
from exercises.tick5 import generate_random_cross_folds, cross_validation_accuracy
import math

'''
Naive Bayes Classifier for a three-outcome classification (positive, neutral, negative)
'''
def nuanced_class_log_probabilities(training_data: List[Dict[str, Union[List[str], int]]]) \
        -> Dict[int, float]:
    """
    Calculate the prior class probability P(c) for nuanced sentiments.

    @param training_data: list of training instances, where each instance is a dictionary with two fields: 'text' and
        'sentiment'. 'text' is the tokenized review and 'sentiment' is +1, 0 or -1, for positive, neutral, and negative sentiments.
    @return: dictionary from sentiment to prior probability
    """
    sentiment_count = {1: 0, -1: 0, 0: 0}

    # Count the number of instances for each sentiment
    for instance in training_data:
        sentiment = instance['sentiment']
        sentiment_count[sentiment] += 1

    n_instances = len(training_data)
    class_log_probabilities = {}

    for sentiment, count in sentiment_count.items():
        class_log_probabilities[sentiment] = math.log(count / n_instances)

    return class_log_probabilities


def nuanced_conditional_log_probabilities(training_data: List[Dict[str, Union[List[str], int]]]) \
        -> Dict[int, Dict[str, float]]:
    """
    Calculate the smoothed log likelihood log (P(x|c)) of a word in the vocabulary given a nuanced sentiment.

    @param training_data: list of training instances, where each instance is a dictionary with two fields: 'text' and
        'sentiment'. 'text' is the tokenized review and 'sentiment' is +1, 0 or -1, for positive, neutral, and negative sentiments.
    @return: dictionary from sentiment to Dictionary of tokens with respective log probability
    """
    word_count = {1: {}, -1: {}, 0: {}}
    n_words = {1: 0, -1: 0, 0: 0}
    word_log_probabilities = {1: {}, -1: {}, 0: {}}
    vocabulary = set()

    # Count number of occurrences of each word for each sentiment
    for instance in training_data:
        sentiment = instance["sentiment"]
        for word in instance['text']:
            vocabulary.add(word)
            word_count[sentiment].setdefault(word, 0)
            word_count[sentiment][word] += 1
            n_words[sentiment] += 1

    total_word_count = {1: n_words[1] + len(vocabulary), -1: n_words[-1] + len(vocabulary), 0: n_words[0] + len(vocabulary)}

    for sentiment in [-1, 0, 1]:
        for word in vocabulary:
            word_count[sentiment].setdefault(word, 0)
            word_count[sentiment][word] += 1
        for word in word_count[sentiment]:
            probability = (word_count[sentiment][word] + 1) / total_word_count[sentiment]
            word_log_probabilities[sentiment][word] = math.log(probability)

    return word_log_probabilities


def nuanced_accuracy(pred: List[int], true: List[int]) -> float:
    """
    Calculate the proportion of predicted sentiments that were correct.

    @param pred: list of calculated sentiment for each review
    @param true: list of correct sentiment for each review
    @return: the overall accuracy of the predictions
    """
    correct = sum(1 for p, t in zip(pred, true) if p == t)
    incorrect = sum(1 for p, t in zip(pred, true) if p != t)
    return correct / (correct + incorrect) if (correct + incorrect) > 0 else 0


def predict_nuanced_sentiment_nbc(review: List[str], log_probabilities: Dict[int, Dict[str, float]],
                                  class_log_probabilities: Dict[int, float]) -> int:
    """
    Use the estimated log probabilities to predict the sentiment of a given review.

    @param review: a single review as a list of tokens
    @param log_probabilities: dictionary from sentiment to Dictionary of tokens with respective log probability
    @param class_log_probabilities: dictionary from sentiment to prior probability
    @return: predicted sentiment [-1, 0, 1] for the given review
    """
    total_log_probabilities = {1: class_log_probabilities[1], -1: class_log_probabilities[-1], 0: class_log_probabilities[0]}

    # Summing log probabilities of each token in the review for each sentiment
    for token in review:
        for sentiment in [1, 0, -1]:
            if token in log_probabilities[sentiment]:
                total_log_probabilities[sentiment] += log_probabilities[sentiment][token]

    # Return the sentiment with the highest probability
    predicted_sentiment = max(total_log_probabilities, key=total_log_probabilities.get)

    return predicted_sentiment


def calculate_kappa(agreement_table: Dict[int, Dict[int,int]]) -> float:
    """
    Using your agreement table, calculate the kappa value for how much agreement there was; 1 should mean total agreement and -1 should mean total disagreement.

    @param agreement_table:  For each review (1, 2, 3, 4) the number of predictions that predicted each sentiment
    @return: The kappa value, between -1 and 1
    """
    N = len(agreement_table)                                # number of reviews
    k = sum(list(agreement_table.values())[0].values())     # total number of predictions

    # mean of the squared proportions of assignments to each class
    p_e = 0
    for sentiment in (-1, 0, 1):
        n_ij = sum(agreement_table[review].get(sentiment, 0) for review in agreement_table)
        p_e += (n_ij / (N * k)) ** 2

    # mean of the proportions of prediction pairs which are in agreement for each item
    p_a = 0
    for review in agreement_table:
        total = 0
        for sentiment in (-1, 0, 1):
            n_ij = agreement_table[review].get(sentiment, 0)
            total += n_ij * (n_ij - 1)
        p_a += total / (k * (k - 1))
    p_a = p_a / N

    kappa = (p_a - p_e) / (1 - p_e)

    return kappa


def get_agreement_table(review_predictions: List[Dict[int, int]]) -> Dict[int, Dict[int,int]]:
    """
    Builds an agreement table from the student predictions.

    @param review_predictions: a list of predictions for each student, the predictions are encoded as dictionaries, with the key being the review id and the value the predicted sentiment
    @return: an agreement table, which for each review contains the number of predictions that predicted each sentiment.
    """
    agreement_table = {}

    # Count the number of predictions that predicted each sentiment for each review
    for student in review_predictions:
        for review, sentiment in student.items():
            if review not in agreement_table:
                agreement_table[review] = {1: 0, -1: 0, 0: 0}
            agreement_table[review][sentiment] += 1

    return agreement_table


def main():
    """
    Code to check your work locally (run this from the root directory, 'mlrd/')
    """
    review_data = load_reviews(os.path.join('data', 'sentiment_detection', 'reviews_nuanced'), include_nuance=True)
    tokenized_data = [{'text': read_tokens(fn['filename']), 'sentiment': fn['sentiment']} for fn in review_data]

    split_training_data = generate_random_cross_folds(tokenized_data, n=10)

    n = len(split_training_data)
    accuracies = []
    for i in range(n):
        test = split_training_data[i]
        train_unflattened = split_training_data[:i] + split_training_data[i+1:]
        train = [item for sublist in train_unflattened for item in sublist]

        dev_tokens = [x['text'] for x in test]
        dev_sentiments = [x['sentiment'] for x in test]

        class_priors = nuanced_class_log_probabilities(train)
        nuanced_log_probabilities = nuanced_conditional_log_probabilities(train)
        preds_nuanced = []
        for review in dev_tokens:
            pred = predict_nuanced_sentiment_nbc(review, nuanced_log_probabilities, class_priors)
            preds_nuanced.append(pred)
        acc_nuanced = nuanced_accuracy(preds_nuanced, dev_sentiments)
        accuracies.append(acc_nuanced)

    mean_accuracy = cross_validation_accuracy(accuracies)
    print(f"Your accuracy on the nuanced dataset: {mean_accuracy}\n")

    review_predictions = read_student_review_predictions(os.path.join('data', 'sentiment_detection', 'class_predictions_2023_2024.csv'))

    print('Agreement table for this year.')

    agreement_table = get_agreement_table(review_predictions)
    print_agreement_table(agreement_table)

    fleiss_kappa = calculate_kappa(agreement_table)

    print(f"The cohen kappa score for the review predictions is {fleiss_kappa}.")

    fleiss_kappa = calculate_kappa({x:y for x,y in agreement_table.items() if x in [0, 1]})

    print(f"The cohen kappa score for the review predictions of review 1 and 2 is {fleiss_kappa}.")

    fleiss_kappa = calculate_kappa({x:y for x,y in agreement_table.items() if x in [2, 3]})

    print(f"The cohen kappa score for the review predictions of review 3 and 4 is {fleiss_kappa}.\n")

    review_predictions_four_years = read_student_review_predictions(os.path.join('data', 'sentiment_detection', 'class_predictions_2019_2023.csv'))
    agreement_table_four_years = get_agreement_table(review_predictions_four_years)

    print('Agreement table for the years 2019 to 2023.')
    print_agreement_table(agreement_table_four_years)

    fleiss_kappa = calculate_kappa(agreement_table_four_years)

    print(f"The cohen kappa score for the review predictions from 2019 to 2023 is {fleiss_kappa}.")



if __name__ == '__main__':
    main()
