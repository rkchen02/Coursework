from utils.markov_models import load_bio_data
import os
import random
from exercises.tick8 import recall_score, precision_score, f1_score
from collections import defaultdict
import math
from typing import List, Dict, Tuple
import copy

'''
Application of Hidden Markov models to biological data
with a simple semi-supervised approach of incorporating additional data 
of only observed sequences to refine the HMM model
'''

STATES = ['B', 'Z', 'i', 'o', 'M']

def get_transition_probs_bio(hidden_sequences: List[List[str]]) -> Dict[Tuple[str, str], float]:
    """
    Calculate the transition probabilities for the hidden feature types using maximum likelihood estimation.

    @param hidden_sequences: A list of feature sequences
    @return: A dictionary from a (hidden_state, hidden_state) tuple to a probability.
    """
    transition_counts = defaultdict(int)
    state_counts = defaultdict(int)

    # Count the number of each transition and states
    for sequence in hidden_sequences:
        for i in range(len(sequence) - 1):
            transition_counts[(sequence[i], sequence[i + 1])] += 1
            state_counts[sequence[i]] += 1

    # Calculate probabilities
    transition_probs = {
        transition: count / state_counts[transition[0]]
        for transition, count in transition_counts.items()
    }

    # Ensure all transitions have an entry in the dictionary
    for s1 in STATES:
        for s2 in STATES:
            if (s1, s2) not in transition_probs:
                transition_probs[(s1, s2)] = 0.0

    return transition_probs


def get_emission_probs_bio(hidden_sequences: List[List[str]], observed_sequences: List[List[str]]) -> Dict[Tuple[str, str], float]:
    """
    Calculate the emission probabilities from hidden feature states to visible amino acids, using maximum likelihood estimation.
    @param hidden_sequences: A list of feature sequences
    @param observed_sequences: A list of amino acids sequences
    @return: A dictionary from a (hidden_state, observed_state) tuple to a probability.
    """
    emission_counts = defaultdict(int)
    state_counts = defaultdict(int)

    # Count number of times each amino acid appears for the given state
    for hidden_seq, observed_seq in zip(hidden_sequences, observed_sequences):
        for hidden_state, observed_state in zip(hidden_seq, observed_seq):
            emission_counts[(hidden_state, observed_state)] += 1
            state_counts[hidden_state] += 1

    # Calculate probabilities
    emission_probs = {
        (h_state, o_state): count / state_counts[h_state]
        for (h_state, o_state), count in emission_counts.items()
    }

    all_observations = set([obs for seq in observed_sequences for obs in seq])

    # Ensure all emissions have a value in the dictionary
    for h_state in state_counts:
        for o_state in all_observations:
            if (h_state, o_state) not in emission_probs:
                emission_probs[(h_state, o_state)] = 0.0

    return emission_probs


def estimate_hmm_bio(training_data:List[Dict[str, List[str]]]) -> List[Dict[Tuple[str, str], float]]:
    """
    The parameter estimation (training) of the HMM. It consists of the calculation of transition and emission probabilities.

    @param training_data: The biological sequence data (amino acids and feature), encoded as a list of dictionaries with each consisting of the fields 'observed', and 'hidden'.
    @return A list consisting of two dictionaries, the first for the transition probabilities, the second for the emission probabilities.
    """
    start_state = 'B'
    end_state = 'Z'

    hidden_sequences = [[start_state] + x['hidden'] + [end_state] for x in training_data]
    observed_sequences = [[start_state] + x['observed'] + [end_state] for x in training_data]

    transition_probs = get_transition_probs_bio(hidden_sequences)
    emission_probs = get_emission_probs_bio(hidden_sequences, observed_sequences)

    return [transition_probs, emission_probs]


def viterbi_bio(observed_sequence, transition_probs: Dict[Tuple[str, str], float], emission_probs: Dict[Tuple[str, str], float]) -> List[str]:
    """
    Uses the Viterbi algorithm to calculate the most likely single sequence of hidden states given the observed sequence and a model.

    @param observed_sequence: A sequence of observed die rolls
    @param: A dictionary from a (hidden_state, hidden_state) tuple to a probability.
    @param: A dictionary from a (hidden_state, observed_state) tuple to a probability.
    @return: The most likely single sequence of hidden states
    """
    T = len(observed_sequence)

    # List of dictionaries, each dictionary maps each current hidden state to the most probable previous hidden state
    # List[Dict[str, str]]
    prev_hidden_state = [{} for _ in range(T + 2)]

    # List of dictionaries, each dictionary maps each hidden state to its path probability at each step
    # List[Dict[str, float]]
    path_prob = [{} for _ in range(T + 2)]

    # Initialize at t = 0
    for state in STATES:
        if emission_probs[(state, observed_sequence[0])] > 0 and transition_probs[('B', state)] > 0:
            path_prob[0][state] = math.log(emission_probs[(state, observed_sequence[0])]) + math.log(transition_probs[('B', state)])
        else:
            path_prob[0][state] = -math.inf
        prev_hidden_state[0][state] = STATES[0]

    # Run Viterbi for t = 1 to T-1
    for t in range(1, T):
        for state in STATES:
            max_prob = -math.inf
            max_state = None
            for prev_state in STATES:
                if transition_probs[(prev_state, state)] > 0 and emission_probs[(state, observed_sequence[t])] > 0:
                    prob = path_prob[t - 1][prev_state] + math.log(transition_probs[(prev_state, state)]) + math.log(
                        emission_probs[(state, observed_sequence[t])])
                else:
                    prob = -math.inf
                if prob > max_prob:
                    max_prob = prob
                    max_state = prev_state
            path_prob[t][state] = max_prob
            prev_hidden_state[t][state] = max_state

    # Transition probabilities for the end state
    for state in STATES:
        if transition_probs[(state, 'Z')] > 0:
            path_prob[T][state] = path_prob[T-1][state] + math.log(transition_probs[(state, 'Z')])
        else:
            path_prob[T][state] = -math.inf
        prev_hidden_state[T][state] = max(path_prob[T - 1], key=path_prob[T - 1].get)

    # Termination
    last_state = max(path_prob[T], key=path_prob[T].get)
    most_likely_hidden_sequence = [last_state]

    # Backtracking
    for t in range(T-1, 0, -1):
        last_state = prev_hidden_state[t][last_state]
        most_likely_hidden_sequence.insert(0, last_state)

    return most_likely_hidden_sequence


def self_training_hmm(training_data: List[Dict[str, List[str]]], dev_data: List[Dict[str, List[str]]],
    unlabeled_data: List[List[str]], num_iterations: int) -> List[Dict[str, float]]:
    """
    The self-learning algorithm for your HMM for a given number of iterations, using a training, development, and unlabeled dataset (no cross-validation to be used here since only very limited computational resources are available.)

    @param training_data: The training set of biological sequence data (amino acids and feature), encoded as a list of dictionaries with each consisting of the fields 'observed', and 'hidden'.
    @param dev_data: The development set of biological sequence data (amino acids and feature), encoded as a list of dictionaries with each consisting of the fields 'observed', and 'hidden'.
    @param unlabeled_data: Unlabeled sequence data of amino acids, encoded as a list of sequences.
    @param num_iterations: The number of iterations of the self_training algorithm, with the initial HMM being the first iteration.
    @return: A list of dictionaries of scores for 'recall', 'precision', and 'f1' for each iteration.
    """
    scores = []
    training_data_copy = copy.deepcopy(training_data)

    for i in range(num_iterations + 1):
        # Train HMM on labelled data
        transition_probs, emission_probs = estimate_hmm_bio(training_data_copy)

        # Predict hidden states for unlabeled data
        predictions = []
        for sequence in unlabeled_data:
            prediction = viterbi_bio(sequence, transition_probs, emission_probs)
            predictions.append({'observed': sequence, 'hidden': prediction})

        # Merge the labelled data with the new pseudo-labelled data
        training_data_copy = copy.deepcopy(training_data)
        training_data_copy += predictions

        # Evaluate HMM on dev data
        dev_predicted = [viterbi_bio(seq['observed'], transition_probs, emission_probs) for seq in dev_data]
        dev_true = [seq['hidden'] for seq in dev_data]

        predictions_binarized = [[1 if x == 'M' else 0 for x in pred] for pred in dev_predicted]
        dev_hidden_sequences_binarized = [[1 if x == 'M' else 0 for x in dev] for dev in dev_true]

        recall = recall_score(predictions_binarized, dev_hidden_sequences_binarized)
        precision = precision_score(predictions_binarized, dev_hidden_sequences_binarized)
        f1 = f1_score(predictions_binarized, dev_hidden_sequences_binarized)

        print(f"Iteration {i}: Recall: {recall}, Precision: {precision}, F1: {f1}")
        scores.append({'recall': recall, 'precision': precision, 'f1': f1})

    return scores


def main():
    """
    Code to check your work locally (run this from the root directory, 'mlrd/')
    """
    bio_data = load_bio_data(os.path.join('data', 'markov_models', 'bio_dataset.txt'))

    seed = 2
    print(f"Evaluating HMM on a single training and dev split using random seed {seed}.")
    random.seed(seed)
    bio_data_shuffled = random.sample(bio_data, len(bio_data))
    dev_size = int(len(bio_data_shuffled) / 10)
    train = bio_data_shuffled[dev_size:]
    dev = bio_data_shuffled[:dev_size]
    dev_observed_sequences = [x['observed'] for x in dev]
    dev_hidden_sequences = [x['hidden'] for x in dev]
    predictions = []
    transition_probs, emission_probs = estimate_hmm_bio(train)

    for sample in dev_observed_sequences:
        prediction = viterbi_bio(sample, transition_probs, emission_probs)
        predictions.append(prediction)
    predictions_binarized = [[1 if x=='M' else 0 for x in pred] for pred in predictions]
    dev_hidden_sequences_binarized = [[1 if x=='M' else 0 for x in dev] for dev in dev_hidden_sequences]

    p = precision_score(predictions_binarized, dev_hidden_sequences_binarized)
    r = recall_score(predictions_binarized, dev_hidden_sequences_binarized)
    f1 = f1_score(predictions_binarized, dev_hidden_sequences_binarized)

    print(f"Your precision for seed {seed} using the HMM: {p}")
    print(f"Your recall for seed {seed} using the HMM: {r}")
    print(f"Your F1 for seed {seed} using the HMM: {f1}\n")

    unlabeled_data = []
    with open(os.path.join('data', 'markov_models', 'bio_dataset_unlabeled.txt'), encoding='utf-8') as f:
        content = f.readlines()
        for i in range(0, len(content), 2):
            unlabeled_data.append(list(content[i].strip())[1:])

    scores_each_iteration = self_training_hmm(train, dev, unlabeled_data, 5)
    for i in range(5):
        print(f"Iteration {i}: {scores_each_iteration[i]}")


if __name__ == '__main__':
    main()
