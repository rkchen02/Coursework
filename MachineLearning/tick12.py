import os
from typing import Set, Dict, List, Tuple
from exercises.tick10 import load_graph

'''
Implement the Girvan-Newman algorithm, which divides a graph into clusters by 
splitting it into relatively dense components.
'''

def get_number_of_edges(graph: Dict[int, Set[int]]) -> int:
    """
    Find the number of edges in the graph.

    @param graph: a dictionary mapping each node ID to a set of node IDs it is connected to
    @return: the number of edges
    """
    num_edges = 0

    for node, neighbours in graph.items():
        num_edges += len(neighbours)

    return num_edges // 2


def get_components(graph: Dict[int, Set[int]]) -> List[Set[int]]:
    """
    Find the number of components in the graph using a DFS.

    @param graph: a dictionary mappings each node ID to a set of node IDs it is connected to
    @return: list of components for the graph.
    """
    components = []
    visited = set()

    # DFS algorithm to check for connectivity between a pair of nodes
    def dfs(node, component):
        visited.add(node)
        component.add(node)
        for neighbour in graph[node]:
            if neighbour not in visited:
                dfs(neighbour, component)

    for node in graph:
        if node not in visited:
            component = set()
            dfs(node, component)
            components.append(component)

    return components


def get_edge_betweenness(graph: Dict[int, Set[int]]) -> Dict[Tuple[int, int], float]:
    """
    Calculate the edge betweenness.

    @param graph: a dictionary mapping each node ID to a set of node IDs it is connected to
    @return: betweenness for each pair of vertices in the graph connected to each other by an edge
    """
    # Initialize edge betweenness
    edge_betweenness = {}
    for node in graph:
        for neighbour in graph[node]:
            edge_betweenness[(node, neighbour)] = 0.0

    for node in graph:
        stack = []
        predecessors = {node: [] for node in graph}
        distance = {node: -1 for node in graph}
        sigma = {node: 0 for node in graph}
        distance[node] = 0
        sigma[node] = 1
        queue = [node]

        while queue:
            current = queue.pop(0)
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
            w = stack.pop()
            for v in predecessors[w]:
                c = (sigma[v] / sigma[w]) * (1 + delta[w])
                edge_betweenness[(v, w)] += c
                delta[v] += c

    return edge_betweenness


def girvan_newman(graph: Dict[int, Set[int]], min_components: int) -> List[Set[int]]:
    """     * Find the number of edges in the graph.
     *
     * @param graph
     *        {@link Map}<{@link Integer}, {@link Set}<{@link Integer}>> The
     *        loaded graph
     * @return {@link Integer}> Number of edges.
    """
    # While number of connected components less than the specified number of clusters,
    # and number of edges in the graph is greater than 0
    components = get_components(graph)

    while len(components) < min_components and get_number_of_edges(graph) > 0:
        # Calculate edge betweeness for every edge in the graph
        edge_betweenness = get_edge_betweenness(graph)
        max_edge = max(edge_betweenness, key=edge_betweenness.get)

        # Remove the edge with the highest betweenness
        graph[max_edge[0]].remove(max_edge[1])
        graph[max_edge[1]].remove(max_edge[0])

        # Recalculate connected components
        components = get_components(graph)

    return components

def main():
    graph = load_graph(os.path.join('data', 'social_networks', 'facebook_circle.edges'))

    # Graph 1
    graph = {1: {2, 5}, 2: {3}, 3: {4}, 4: {5}, 5: {}}
    # Graph 2
    graph = {1:{2, 3, 4, 5}, 2: {}, 3: {}, 4: {}, 5: {}}

    num_edges = get_number_of_edges(graph)
    print(f"Number of edges: {num_edges}")

    components = get_components(graph)
    print(f"Number of components: {len(components)}")

    edge_betweenness = get_edge_betweenness(graph)
    print(f"Edge betweenness: {edge_betweenness}")

    # clusters = girvan_newman(graph, min_components=20)
    # print(f"Girvan-Newman for 20 clusters: {clusters}")


if __name__ == '__main__':
    main()
