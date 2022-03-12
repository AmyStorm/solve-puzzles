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
    #找子樹的求和集
    def subtreecalc(self, tree: TreeNode, resultlist: list) -> list:
        if tree is None:
            return list()
        leftTree = tree.left
        rightTree = tree.right

        tmpList = list()
        tmpList.append(tree.val)
        lList = list()
        rList = list()
        if leftTree is not None:
            lList = self.subtreecalc(leftTree, resultlist)

        if rightTree is not None:
            rList = self.subtreecalc(rightTree, resultlist)

        if len(rList) != 0:
            tmpList.append(max(rList) + tree.val)
        if len(lList) != 0:
            tmpList.append(max(lList) + tree.val)
        if len(rList) != 0 and len(lList) != 0:
            resultlist.append(max(lList) + max(rList) + tree.val)
        resultlist += tmpList
        print(str(tree.val) + '-all-' + str(resultlist))
        return tmpList

    def maxPathSum(self, root: Optional[TreeNode]) -> int:
        resultList = list()
        self.subtreecalc(root, resultList)
        print(max(resultList))
        return max(resultList)

    def main(self):
        treeNode = TreeNode(10, TreeNode(5, TreeNode(3, TreeNode(3), TreeNode(-2)), TreeNode(2, None, TreeNode(1))),
                 TreeNode(-3, None, TreeNode(11)))
        self.maxPathSum(treeNode)

if __name__ == '__main__':
    Solution().main()
