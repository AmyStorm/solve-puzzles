# Definition for a binary tree node.
# class TreeNode:
#     def __init__(self, val=0, left=None, right=None):
#         self.val = val
#         self.left = left
#         self.right = right
from typing import Optional, List


class TreeNode:
    def __init__(self, val=0, left=None, right=None):
        self.val = val
        self.left = left
        self.right = right


class Solution:
    def backtracking(self, tree: TreeNode, resultlist: list, path: list, remainNum : int):
        if tree is None:
            return

        isLeaf = tree.left is None and tree.right is None
        remainNum -= tree.val
        path.append(tree.val)
        print(str(tree.val) + '-' + str(isLeaf))
        if isLeaf:
            if remainNum == 0:
                resultlist.append(list(path))
                return
            else:
                return

        for x in ['left', 'right']:
            if getattr(tree, x) is not None:
                self.backtracking(getattr(tree, x), resultlist, path, remainNum)
                path.pop(len(path) - 1)


    def pathSum(self, root: Optional[TreeNode], targetSum: int) -> List[List[int]]:
        resultList = list()
        self.backtracking(root, resultList, list(), targetSum)
        print(resultList)
        return resultList

    def main(self):
        treeNode = TreeNode(-2, None, TreeNode(-3))
        # treeNode = TreeNode(10, TreeNode(5, TreeNode(3, TreeNode(3), TreeNode(-2)), TreeNode(2, None, TreeNode(1))),
        #          TreeNode(-3, None, TreeNode(11)))
        # treeNode = TreeNode(1, TreeNode(-2, TreeNode(1, TreeNode(-1), None), TreeNode(3)), TreeNode(-3, TreeNode(-2), None))
        self.pathSum(treeNode, -5)

if __name__ == '__main__':
    Solution().main()
