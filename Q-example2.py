import collections

class Node:
    def __init__(self, val=0, neighbors=None):
        self.val = val
        self.neighbors = neighbors if neighbors is not None else []

def cloneGraph(node):
    if not node:
        return None
    nodes = get_all_nodes(node)
    mapping = map_all_nodes(nodes)
    clone_edge(nodes, mapping)
    return mapping[node]

def get_all_nodes(node):
    nodes = []
    queue = collections.deque([(node)])
    visited = set([(node)])

    while queue:
        node = queue.popleft()
        nodes.append(node)
        for n in node.neighbors:
            if n not in visited:
                visited.add(n)
                queue.append(n)
    return nodes

def map_all_nodes(nodes):
    mapping = {}
    for node in nodes:
        mapping[node] = Node(node.val)
    return mapping

def clone_edge(nodes, mapping):
    for node in nodes:
        new_node = mapping[node]
        for n in node.neighbors:
            new_neighbor = mapping[n]
            new_node.neighbors.append(new_neighbor)

if __name__ == '__main__':

    node1 = Node(1)
    node2 = Node(2)
    node3 = Node(3)
    node4 = Node(4)
    node1.neighbors.append(node2)
    node1.neighbors.append(node4)
    node2.neighbors.append(node1)
    node2.neighbors.append(node3)
    node3.neighbors.append(node2)
    node3.neighbors.append(node4)
    node4.neighbors.append(node1)
    node4.neighbors.append(node3)

    print(cloneGraph(node1))
