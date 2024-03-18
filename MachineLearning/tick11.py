import os
from typing import Dict, Set
from exercises.tick10 import load_graph
from collections import deque

'''
Calculate the betweenness centrality for each node in the graph using Brandes' algorithm.
'''

def get_node_betweenness(graph: Dict[int, Set[int]]) -> Dict[int, float]:
    """
    Use Brandes' algorithm to calculate the betweenness centrality for each node in the graph.

    @param graph: a dictionary mapping each node ID to a set of node IDs it is connected to
    @return: a dictionary mapping each node ID to that node's betweenness centrality
    """
    betweenness = {node: 0.0 for node in graph}

    for node in graph:
        stack = []
        predecessors = {node: [] for node in graph}
        distance = {node: -1 for node in graph}
        sigma = {node: 0 for node in graph}
        distance[node] = 0
        sigma[node] = 1
        queue = deque([node])

        while queue:
            current = queue.popleft()
            stack.append(current)

            # Explore the neighbours
            for neighbour in graph[current]:
                # Path discovery
                if distance[neighbour] < 0:
                    queue.append(neighbour)
                    distance[neighbour] = distance[current] + 1

                # Path counting
                if distance[neighbour] == distance[current] + 1:
                    sigma[neighbour] += sigma[current]
                    predecessors[neighbour].append(current)

        delta = {node: 0 for node in graph}

        # Accumulation
        while stack:
            current = stack.pop()
            for predecessor in predecessors[current]:
                delta[predecessor] += (sigma[predecessor] / sigma[current]) * (1 + delta[current])
            if current != node:
                betweenness[current] += delta[current]

    for node in graph:
        betweenness[node] /= 2

    return betweenness

def main():
    graph = load_graph(os.path.join('data', 'social_networks', 'simple_network.edges'))

    graph = {1: {2, 5}, 2: {3}, 3: {4}, 4: {5}, 5: {}}

    betweenness = get_node_betweenness(graph)
    print(f"Node betweenness values: {betweenness}")


if __name__ == '__main__':
    main()
