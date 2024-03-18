import os
from typing import Dict, Set
from collections import deque

'''
Graph visualisation - given an undirected, unweighted graph,
visualise and calculate various network statistics
'''

def load_graph(filename: str) -> Dict[int, Set[int]]:
    """
    Load the graph file. Each line in the file corresponds to an edge; the first column is the source node and the
    second column is the target. As the graph is undirected, your program should add the source as a neighbour of the
    target as well as the target a neighbour of the source.

    @param filename: The path to the network specification
    @return: a dictionary mapping each node (represented by an integer ID) to a set containing all the nodes it is
        connected to (also represented by integer IDs)
    """
    graph = {}

    with open(filename, 'r') as file:
        for line in file:
            source, target = map(int, line.split())

            if source not in graph:
                graph[source] = set()
            if target not in graph:
                graph[target] = set()

            graph[source].add(target)
            graph[target].add(source)

    return graph


def get_node_degrees(graph: Dict[int, Set[int]]) -> Dict[int, int]:
    """
    Find the number of neighbours of each node in the graph.

    @param graph: a dictionary mappings each node ID to a set of node IDs it is connected to
    @return: a dictionary mapping each node ID to the degree of the node
    """
    degrees = {}

    for node, neighbours in graph.items():
        degrees[node] = len(neighbours)

    return degrees


def get_diameter(graph: Dict[int, Set[int]]) -> int:
    """
    Find the longest shortest path between any two nodes in the network using a breadth-first search.

    @param graph: a dictionary mappings each node ID to a set of node IDs it is connected to
    @return: the length of the longest shortest path between any pair of nodes in the graph
    """
    max_diameter = 0

    for node in graph:
        visited = set()
        queue = deque([(node, 0)])

        while queue:
            current, distance = queue.popleft()
            visited.add(current)

            max_diameter = max(max_diameter, distance)

            for neighbour in graph[current]:
                if neighbour not in visited:
                    queue.append((neighbour, distance + 1))
                    visited.add(neighbour)

    return max_diameter



def main():
    graph = load_graph(os.path.join('data', 'social_networks', 'simple_network.edges'))

    degrees = get_node_degrees(graph)
    print(f"Node degrees: {degrees}")

    diameter = get_diameter(graph)
    print(f"Diameter: {diameter}")


if __name__ == '__main__':
    main()
