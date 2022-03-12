# Definition for a binary tree node.
# class TreeNode:
#     def __init__(self, val=0, left=None, right=None):
#         self.val = val
#         self.left = left
#         self.right = right
from typing import Optional


class TreeNode:
    def __init__(self, val=0, left=None, right=None):
        self.val = val
        self.left = left
        self.right = right


class Solution:
    def subtreecalc(self, tree: TreeNode, maxValue: int, matchNodes: list):
        if tree is None:
            return 0
        leftTree = tree.left
        rightTree = tree.right
        print(str(tree.val) + '---' + str(maxValue))
        if tree.val >= maxValue:
            maxValue = tree.val
            matchNodes.append(tree.val)
            print(1)
        if leftTree is not None:
            self.subtreecalc(leftTree, maxValue, matchNodes)

        if rightTree is not None:
            self.subtreecalc(rightTree, maxValue, matchNodes)
        return 0

    def goodNodes(self, root: TreeNode) -> int:
        matchNodes = list()
        self.subtreecalc(root, root.val, matchNodes)
        print(matchNodes)
        return len(matchNodes)

    def main(self):
        treeNode = TreeNode(10, TreeNode(5, TreeNode(3, TreeNode(3), TreeNode(-2)), TreeNode(2, None, TreeNode(1))),
                 TreeNode(-3, None, TreeNode(11)))
        self.goodNodes(treeNode)

if __name__ == '__main__':
    Solution().main()
