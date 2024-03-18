from typing import List, Dict, Union
import os
from utils.sentiment_detection import read_tokens, load_reviews, print_binary_confusion_matrix
from exercises.tick1 import accuracy, predict_sentiment, read_lexicon, predict_sentiment_improved
from exercises.tick2 import predict_sentiment_nbc, calculate_smoothed_log_probabilities, \
    calculate_class_log_probabilities
import random
from exercises.tick4 import sign_test

'''
n-fold cross validation and evaluation sets; splitting data into folds using random & stratified random methods
'''

def generate_random_cross_folds(training_data: List[Dict[str, Union[List[str], int]]], n: int = 10) \
        -> List[List[Dict[str, Union[List[str], int]]]]:
    """
    Split training data into n folds, random.

    @param training_data: list of training instances, where each instance is a dictionary with two fields: 'text' and
        'sentiment'. 'text' is the tokenized review and 'sentiment' is +1 or -1, for positive and negative sentiments.
    @param n: the number of cross-folds
    @return: a list of n folds, where each fold is a list of training instances
    """
    random.shuffle(training_data)
    return [training_data[i::n] for i in range(n)]


def generate_stratified_cross_folds(training_data: List[Dict[str, Union[List[str], int]]], n: int = 10) \
        -> List[List[Dict[str, Union[List[str], int]]]]:
    """
    Split training data into n folds, stratified.

    @param training_data: list of training instances, where each instance is a dictionary with two fields: 'text' and
        'sentiment'. 'text' is the tokenized review and 'sentiment' is +1 or -1, for positive and negative sentiments.
    @param n: the number of cross-folds
    @return: a list of n folds, where each fold is a list of training instances
    """
    # Separate the positive and negative instances
    positive_instances = [instance for instance in training_data if instance['sentiment'] == 1]
    negative_instances = [instance for instance in training_data if instance['sentiment'] == -1]

    # Shuffle the positive and negative instances
    random.shuffle(positive_instances)
    random.shuffle(negative_instances)

    # Split the positive and negative instances into n folds
    positive_folds = [positive_instances[i::n] for i in range(n)]
    negative_folds = [negative_instances[i::n] for i in range(n)]

    # Combine the positive and negative folds
    folds = []
    for i in range(n):
        fold = positive_folds[i] + negative_folds[i]
        folds.append(fold)

    return folds


def cross_validate_nbc(split_training_data: List[List[Dict[str, Union[List[str], int]]]]) -> List[float]:
    """
    Perform an n-fold cross validation, and return the mean accuracy and variance.

    @param split_training_data: a list of n folds, where each fold is a list of training instances, where each instance
        is a dictionary with two fields: 'text' and 'sentiment'. 'text' is the tokenized review and 'sentiment' is +1 or
        -1, for positive and negative sentiments.
    @return: list of accuracy scores for each fold
    """
    n = len(split_training_data)
    accs = []

    for i in range(n):
        # Each fold is only used for testing once, and the rest for training
        test_data = split_training_data[i]
        training_data = [instance
                         for fold in split_training_data[:i] + split_training_data[i+1:]
                         for instance in fold]

        # Calculate class priors and smoothed log probabilities based on training set
        class_priors = calculate_class_log_probabilities(training_data)
        smoothed_log_probabilities = calculate_smoothed_log_probabilities(training_data)

        preds = []

        # Predict sentiment for each review in the test set
        for review in test_data:
            pred = predict_sentiment_nbc(review['text'], smoothed_log_probabilities, class_priors)
            preds.append(pred)

        # Calculate accuracy of predictions
        acc = accuracy(preds, [instance['sentiment'] for instance in test_data])
        accs.append(acc)

    return accs

def cross_validation_accuracy(accuracies: List[float]) -> float:
    """Calculate the mean accuracy across n cross fold accuracies.

    @param accuracies: list of accuracy scores for n cross folds
    @returns: mean accuracy over the cross folds
    """
    return sum(accuracies) / len(accuracies)


def cross_validation_variance(accuracies: List[float]) -> float:
    """Calculate the variance of n cross fold accuracies.

    @param accuracies: list of accuracy scores for n cross folds
    @returns: variance of the cross fold accuracies
    """
    return sum((acc - cross_validation_accuracy(accuracies))**2 for acc in accuracies) / len(accuracies)


def confusion_matrix(predicted_sentiments: List[int], actual_sentiments: List[int]) -> List[List[int]]:
    """
    Calculate the number of times (1) the prediction was POS and it was POS [correct], (2) the prediction was POS but
    it was NEG [incorrect], (3) the prediction was NEG and it was POS [incorrect], and (4) the prediction was NEG and it
    was NEG [correct]. Store these values in a list of lists, [[(1), (2)], [(3), (4)]], so they form a confusion matrix:
                     actual:
                     pos     neg
    predicted:  pos  [[(1),  (2)],
                neg   [(3),  (4)]]

    @param actual_sentiments: a list of the true (gold standard) sentiments
    @param predicted_sentiments: a list of the sentiments predicted by a system
    @returns: a confusion matrix
    """
    # Generate confusion matrix based on predicted and actual sentiments

    # True positives
    pos_pos = sum(1 for p, a in zip(predicted_sentiments, actual_sentiments) if p == a == 1)
    # False positives
    pos_neg = sum(1 for p, a in zip(predicted_sentiments, actual_sentiments) if p == 1 and a == -1)
    # False negatives
    neg_pos = sum(1 for p, a in zip(predicted_sentiments, actual_sentiments) if p == -1 and a == 1)
    # True negatives
    neg_neg = sum(1 for p, a in zip(predicted_sentiments, actual_sentiments) if p == a == -1)

    return [[pos_pos, pos_neg], [neg_pos, neg_neg]]


def main():
    """
    Code to check your work locally (run this from the root directory, 'mlrd/')
    """
    review_data = load_reviews(os.path.join('data', 'sentiment_detection', 'reviews'))
    tokenized_data = [{'text': read_tokens(fn['filename']), 'sentiment': fn['sentiment']} for fn in review_data]

    # First test cross-fold validation
    folds = generate_random_cross_folds(tokenized_data, n=10)
    accuracies = cross_validate_nbc(folds)
    print(f"Random cross validation accuracies: {accuracies}")
    mean_accuracy = cross_validation_accuracy(accuracies)
    print(f"Random cross validation mean accuracy: {mean_accuracy}")
    variance = cross_validation_variance(accuracies)
    print(f"Random cross validation variance: {variance}\n")

    folds = generate_stratified_cross_folds(tokenized_data, n=10)
    accuracies = cross_validate_nbc(folds)
    print(f"Stratified cross validation accuracies: {accuracies}")
    mean_accuracy = cross_validation_accuracy(accuracies)
    print(f"Stratified cross validation mean accuracy: {mean_accuracy}")
    variance = cross_validation_variance(accuracies)
    print(f"Stratified cross validation variance: {variance}\n")

    # Now evaluate on 2016 and test
    class_priors = calculate_class_log_probabilities(tokenized_data)
    smoothed_log_probabilities = calculate_smoothed_log_probabilities(tokenized_data)

    preds_test_nb = []
    test_data = load_reviews(os.path.join('data', 'sentiment_detection', 'reviews_test'))
    test_tokens = [read_tokens(x['filename']) for x in test_data]
    test_sentiments = [x['sentiment'] for x in test_data]
    for review in test_tokens:
        pred = predict_sentiment_nbc(review, smoothed_log_probabilities, class_priors)
        preds_test_nb.append(pred)

    acc_smoothed = accuracy(preds_test_nb, test_sentiments)
    print(f"Smoothed Naive Bayes accuracy on held-out data: {acc_smoothed}")
    print("Confusion matrix:")
    print_binary_confusion_matrix(confusion_matrix(preds_test_nb, test_sentiments))

    preds_recent_nb = []
    recent_review_data = load_reviews(os.path.join('data', 'sentiment_detection', 'reviews_2016'))
    recent_tokens = [read_tokens(x['filename']) for x in recent_review_data]
    recent_sentiments = [x['sentiment'] for x in recent_review_data]
    for review in recent_tokens:
        pred = predict_sentiment_nbc(review, smoothed_log_probabilities, class_priors)
        preds_recent_nb.append(pred)

    acc_smoothed = accuracy(preds_recent_nb, recent_sentiments)
    print(f"Smoothed Naive Bayes accuracy on 2016 data: {acc_smoothed}")
    print("Confusion matrix:")
    print_binary_confusion_matrix(confusion_matrix(preds_recent_nb, recent_sentiments))

    # Step 4
    lexicon = read_lexicon(os.path.join('data', 'sentiment_detection', 'sentiment_lexicon'))
    preds_test = [predict_sentiment_improved(t, lexicon) for t in test_tokens]
    acc_lexicon = accuracy(preds_test, test_sentiments)
    print(f"Improved lexicon accuracy on held-out data: {acc_lexicon}")
    print("Confusion matrix:")
    print_binary_confusion_matrix(confusion_matrix(preds_test, test_sentiments))

    preds_recent = [predict_sentiment_improved(t, lexicon) for t in recent_tokens]
    acc_lexicon = accuracy(preds_recent, recent_sentiments)
    print(f"Improved lexicon accuracy on 2016 data: {acc_lexicon}")
    print("Confusion matrix:")
    print_binary_confusion_matrix(confusion_matrix(preds_recent, recent_sentiments))

    p_value_test = sign_test(test_sentiments, preds_test, preds_test_nb)
    print(
        f"The p-value of the two-sided sign test for classifier_a \"improved lexicon\" and classifier_b \"naive bayes classifier\" on held-out data: {p_value_test}\n")

    p_value_recent = sign_test(recent_sentiments, preds_recent, preds_recent_nb)
    print(
        f"The p-value of the two-sided sign test for classifier_a \"improved lexicon\" and classifier_b \"naive bayes classifier\" on 2016 data: {p_value_recent}\n")

if __name__ == '__main__':
    main()
