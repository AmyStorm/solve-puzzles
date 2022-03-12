"""
# Definition for a Node.

"""
import collections


class Node:
    def __init__(self, val=0, neighbors=None):
        self.val = val
        self.neighbors = neighbors if neighbors is not None else []


class Solution:
    def cloneGraph(self, node: 'Node') -> 'Node':
        if not node:
            return None
        nodes = self.get_all_nodes(node)
        mapping = self.map_all_nodes(nodes)
        self.clone_edge(nodes, mapping)
        return mapping[node]

    def get_all_nodes(self, node):
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

    def map_all_nodes(self, nodes):
        mapping = {}
        for node in nodes:
            mapping[node] = Node(node.val)
        return mapping

    def clone_edge(self, nodes, mapping):
        for node in nodes:
            new_node = mapping[node]
            for n in node.neighbors:
                new_neighbor = mapping[n]
                new_node.neighbors.append(new_neighbor)

    def main(self):
        nodes = [[2, 4], [1, 3], [2, 4], [1, 3]]
        nodeMap = {}
        nodecount = 0
        for node in nodes:
            nodeMap[nodecount] = Node(nodecount)
            for line in node:
                if nodeMap.get(line) is not None:
                    nodeMap[nodecount].neighbors.append(nodeMap.get(line))
                else:
                    nodeMap[line] = Node(line)
                    nodeMap[nodecount].neighbors.append(nodeMap.get(line))
            nodecount += 1
        print(str(nodeMap.get(1)))
        print(str(self.cloneGraph(nodeMap.get(1))))


if __name__ == '__main__':
    Solution().main()
