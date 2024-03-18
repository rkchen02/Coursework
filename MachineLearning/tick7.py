from utils.markov_models import load_dice_data, print_matrices
import os
from typing import List, Dict, Tuple
from collections import defaultdict

'''
Implement a Hidden Markov Model for a sequence of dice rolls.
The dice can be in one of two hidden states: fair or loaded.
The observed states are the numbers 1 to 6 on the dice.
'''

def get_transition_probs(hidden_sequences: List[List[str]]) -> Dict[Tuple[str, str], float]:
    """
    Calculate the transition probabilities for the hidden dice types using maximum likelihood estimation.
    Counts the number of times each state sequence appears and divides it by the count of all transitions going from that state.
    The table must include probability values for all state-state pairs, even if they are zero.

    @param hidden_sequences: A list of dice type sequences
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
    states = set([state for seq in hidden_sequences for state in seq])
    for s1 in states:
        for s2 in states:
            if (s1, s2) not in transition_probs:
                transition_probs[(s1, s2)] = 0.0

    return transition_probs


def get_emission_probs(hidden_sequences: List[List[str]], observed_sequences: List[List[str]]) -> Dict[Tuple[str, str], float]:
    """
    Calculate the emission probabilities from hidden dice states to observed dice rolls, using maximum likelihood estimation.
    Counts the number of times each dice roll appears for the given state (fair or loaded) and divides it by the count of that state.
    The table must include probability values for all state-observation pairs, even if they are zero.

    @param hidden_sequences: A list of dice type sequences
    @param observed_sequences: A list of dice roll sequences
    @return: A dictionary from a (hidden_state, observed_state) tuple to a probability.
    """
    emission_counts = defaultdict(int)
    state_counts = defaultdict(int)

    # Count number of times each dice roll appears for the given state
    for hidden_seq, observed_seq in zip(hidden_sequences, observed_sequences):
        for hidden_state, observed_state in zip(hidden_seq, observed_seq):
            emission_counts[(hidden_state, observed_state)] += 1
            state_counts[hidden_state] += 1

    # Calculate probabilities
    emission_probs = {
        (h_state, o_state): count / state_counts[h_state]
        for (h_state, o_state), count in emission_counts.items()
    }

    # Introduce special start and end observations that can only be emitted by the start and end hidden states
    all_observations = set([obs for seq in observed_sequences for obs in seq])
    for state in ['B', 'Z']:
        for obs in all_observations:
            emission_probs[(state, obs)] = 0.0
        emission_probs[(state, state)] = 1.0

    # Ensure all emissions have a value in the dictionary
    for h_state in state_counts:
        for o_state in all_observations:
            if (h_state, o_state) not in emission_probs:
                emission_probs[(h_state, o_state)] = 0.0

    return emission_probs


def estimate_hmm(training_data: List[Dict[str, List[str]]]) -> List[Dict[Tuple[str, str], float]]:
    """
    The parameter estimation (training) of the HMM. It consists of the calculation of transition and emission probabilities.
    We use 'B' for the start state and 'Z' for the end state, both for emissions and transitions.

    @param training_data: The dice roll sequence data (visible dice rolls and hidden dice types), encoded as a list of dictionaries with each consisting of the fields 'observed', and 'hidden'.
    @return A list consisting of two dictionaries, the first for the transition probabilities, the second for the emission probabilities.
    """
    start_state = 'B'
    end_state = 'Z'

    observed_sequences = [[start_state] + x['observed'] + [end_state] for x in training_data]
    hidden_sequences = [[start_state] + x['hidden'] + [end_state] for x in training_data]

    transition_probs = get_transition_probs(hidden_sequences)
    emission_probs = get_emission_probs(hidden_sequences, observed_sequences)

    return [transition_probs, emission_probs]


def main():
    """
    Code to check your work locally (run this from the root directory, 'mlrd/')
    """
    dice_data = load_dice_data(os.path.join('data', 'markov_models', 'dice_dataset'))
    transition_probs, emission_probs = estimate_hmm(dice_data)
    print(f"The transition probabilities of the HMM:")
    print_matrices(transition_probs)
    print(f"The emission probabilities of the HMM:")
    print_matrices(emission_probs)

if __name__ == '__main__':
    main()
