from utils.markov_models import load_dice_data
import os
from exercises.tick7 import estimate_hmm
import random
import math
from typing import List, Dict, Tuple

'''
Use the Viterbi algorithm to calculate the most likely sequence of 
hidden states underlying an observed sequence.
'''

def viterbi(observed_sequence: List[str], transition_probs: Dict[Tuple[str, str], float], emission_probs: Dict[Tuple[str, str], float]) -> List[str]:
    """
    Uses the Viterbi algorithm to calculate the most likely single sequence of hidden states given the observed sequence and a model. Use the same symbols for the start and end observations as in tick 7 ('B' for the start observation and 'Z' for the end observation).

    @param observed_sequence: A sequence of observed die rolls
    @param transition_probs: A dictionary from a (hidden_state, hidden_state) tuple to a probability.
    @param emission_probs: A dictionary from a (hidden_state, observed_state) tuple to a probability.
    @return: The most likely single sequence of hidden states
    """
    start_state = 'B'
    states = set([state for state, _ in transition_probs.keys()])

    T = len(observed_sequence)

    # List of dictionaries, each dictionary maps each current hidden state to the most probable previous hidden state
    # List[Dict[str, str]]
    prev_hidden_state = [{} for _ in range(T+2)]

    # List of dictionaries, each dictionary maps each hidden state to its path probability at each step
    # List[Dict[str, float]]
    path_prob = [{} for _ in range(T+2)]

    # Initialize at t = 0
    for state in states:
        if emission_probs[(state, observed_sequence[0])] > 0:
            path_prob[0][state] = math.log(emission_probs[(state, observed_sequence[0])])
        else:
            path_prob[0][state] = -math.inf
        prev_hidden_state[0][state] = start_state

    # Run Viterbi for t = 1 to T
    for t in range(1, T):
        for state in states:
            max_prob = -math.inf
            max_state = None
            for prev_state in states:
                if transition_probs[(prev_state, state)] > 0 and emission_probs[(state, observed_sequence[t])] > 0:
                    prob = path_prob[t-1][prev_state] + math.log(transition_probs[(prev_state, state)]) + math.log(emission_probs[(state, observed_sequence[t])])
                else:
                    prob = -math.inf
                if prob > max_prob:
                    max_prob = prob
                    max_state = prev_state
            path_prob[t][state] = max_prob
            prev_hidden_state[t][state] = max_state

    # Termination
    last_state = max(path_prob[T-1], key=path_prob[T-1].get)
    most_likely_hidden_sequence = [last_state]

    # Backtracking
    for t in range(T-1, 0, -1):
        last_state = prev_hidden_state[t][last_state]
        most_likely_hidden_sequence.insert(0, last_state)

    return most_likely_hidden_sequence


def precision_score(pred: List[List[int]], true: List[List[int]]) -> float:
    """
    Calculates the precision of the estimated sequence with respect to the positive class (weighted state), i.e. here the proportion of predicted weighted states that were actually weighted.

    @param pred: List of binarized sequence predictions. 1 if positive (weighted), 0 if not to be considered.
    @param true:  List of gold binarized sequences. 1 if positive (weighted), 0 if not to be considered.
    @return: The precision of the estimated sequence with respect to the positive class(es) averaged over all the test sequences.
    """
    total_positive = 0
    total_predicted_positive = 0

    for pred_seq, true_seq in zip(pred, true):
        for p, t in zip(pred_seq, true_seq):
            if p == 1:
                total_predicted_positive += 1
                if t == 1:
                    total_positive += 1

    precision = total_positive / total_predicted_positive if total_predicted_positive else 0

    return precision


def recall_score(pred: List[List[int]], true: List[List[int]]) -> float:
    """
    Calculates the recall of the estimated sequence with respect to the positive class (weighted state), i.e. here the proportion of actual weighted states that were predicted weighted.

    @param pred: List of binarized sequence predictions. 1 if positive (weighted), 0 if not to be considered.
    @param true:  List of gold binarized sequences. 1 if positive (weighted), 0 if not to be considered.
    @return: The recall of the estimated sequence with respect to the positive class(es) averaged over all the test sequences.
    """
    total_positive = 0
    total_actual_positive = 0

    for pred_seq, true_seq in zip(pred, true):
        for p, t in zip(pred_seq, true_seq):
            if t == 1:
                total_actual_positive += 1
                if p == 1:
                    total_positive += 1

    recall = total_positive / total_actual_positive if total_actual_positive else 0

    return recall


def f1_score(pred: List[List[int]], true: List[List[int]]) -> float:
    """
    Calculates the F1 measure of the estimated sequence with respect to the positive class (weighted state), i.e. the harmonic mean of precision and recall.

    @param pred: List of binarized sequence predictions. 1 if positive (weighted), 0 if not to be considered.
    @param true:  List of gold binarized sequences. 1 if positive (weighted), 0 if not to be considered.
    @return: The F1 measure of the estimated sequence with respect to the positive class(es) averaged over all the test sequences.
    """
    precision = precision_score(pred, true)
    recall = recall_score(pred, true)

    if precision + recall == 0:  # Check to avoid division by zero
        return 0

    return 2 * (precision * recall) / (precision + recall)


def cross_validation_sequence_labeling(data: List[Dict[str, List[str]]]) -> Dict[str, float]:
    """
    Run 10-fold cross-validation for evaluating the HMM's prediction with Viterbi decoding. Calculate precision, recall, and F1 for each fold and return the average over the folds.

    @param data: the sequence data encoded as a list of dictionaries with each consisting of the fields 'observed', and 'hidden'
    @return: a dictionary with keys 'recall', 'precision', and 'f1' and its associated averaged score.
    """
    n = 10          # 10 fold cross validation

    precisions = []
    recalls = []
    f1s = []

    random.shuffle(data)
    folds = [data[i::n] for i in range(n)]

    # Calculate precision, recall, and F1 for each fold
    for i in range(n):
        test = folds[i]
        train = [x for fold in folds[:i] + folds[i+1:] for x in fold]

        dev_observed_sequences = [x['observed'] for x in test]
        dev_hidden_sequences = [x['hidden'] for x in test]
        predictions = []
        transition_probs, emission_probs = estimate_hmm(train)

        for sample in dev_observed_sequences:
            prediction = viterbi(sample, transition_probs, emission_probs)
            predictions.append(prediction)

        predictions_binarized = [[1 if x == 'W' else 0 for x in pred] for pred in predictions]
        dev_hidden_sequences_binarized = [[1 if x == 'W' else 0 for x in dev] for dev in dev_hidden_sequences]

        precisions.append(precision_score(predictions_binarized, dev_hidden_sequences_binarized))
        recalls.append(recall_score(predictions_binarized, dev_hidden_sequences_binarized))
        f1s.append(f1_score(predictions_binarized, dev_hidden_sequences_binarized))

    # Calculate average scores
    avg_precision = sum(precisions) / n
    avg_recall = sum(recalls) / n
    avg_f1 = sum(f1s) / n

    return {'precision': avg_precision, 'recall': avg_recall, 'f1': avg_f1}


def main():
    """
    Code to check your work locally (run this from the root directory, 'mlrd/')
    """
    dice_data = load_dice_data(os.path.join('data', 'markov_models', 'dice_dataset'))

    seed = 2
    print(f"Evaluating HMM on a single training and dev split using random seed {seed}.")
    random.seed(seed)
    dice_data_shuffled = random.sample(dice_data, len(dice_data))
    dev_size = int(len(dice_data) / 10)
    train = dice_data_shuffled[dev_size:]
    dev = dice_data_shuffled[:dev_size]
    dev_observed_sequences = [x['observed'] for x in dev]
    dev_hidden_sequences = [x['hidden'] for x in dev]
    predictions = []
    transition_probs, emission_probs = estimate_hmm(train)

    for sample in dev_observed_sequences:
        prediction = viterbi(sample, transition_probs, emission_probs)
        predictions.append(prediction)

    predictions_binarized = [[1 if x == 'W' else 0 for x in pred] for pred in predictions]
    dev_hidden_sequences_binarized = [[1 if x == 'W' else 0 for x in dev] for dev in dev_hidden_sequences]

    p = precision_score(predictions_binarized, dev_hidden_sequences_binarized)
    r = recall_score(predictions_binarized, dev_hidden_sequences_binarized)
    f1 = f1_score(predictions_binarized, dev_hidden_sequences_binarized)

    print(f"Your precision for seed {seed} using the HMM: {p}")
    print(f"Your recall for seed {seed} using the HMM: {r}")
    print(f"Your F1 for seed {seed} using the HMM: {f1}\n")

    print(f"Evaluating HMM using cross-validation with 10 folds.")

    cv_scores = cross_validation_sequence_labeling(dice_data)

    print(f" Your cv average precision using the HMM: {cv_scores['precision']}")
    print(f" Your cv average recall using the HMM: {cv_scores['recall']}")
    print(f" Your cv average F1 using the HMM: {cv_scores['f1']}")



if __name__ == '__main__':
    main()
