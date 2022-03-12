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


class DoubleList:
    def __init__(self, val=0, left=None, right=None):
        self.val = val
        self.left = left
        self.right = right


class Solution:
    #找子樹的路徑集
    def subtreecalc(self, tree: TreeNode, dummy: DoubleList) -> list:
        if tree is None:
            return list()
        leftTree = tree.left
        rightTree = tree.right

        if leftTree is not None:
            tmp = self.subtreecalc(leftTree, resultlist)
            index = 0
            for item in tmp:
                tmp[index] = item + tree.val
                index += 1
            print(str(tree.val) + '-l-' + str(tmp))
            tmpList += tmp
            resultlist += tmp
            # resultlist.append([elem + tree.val for elem in tmp])
        if rightTree is not None:
            tmp = self.subtreecalc(rightTree, resultlist)
            index = 0
            for item in tmp:
                tmp[index] = item + tree.val
                index += 1
            print(str(tree.val) + '-r-' + str(tmp))
            tmpList += tmp
            resultlist += tmp
            # resultlist.append([elem + tree.val for elem in tmp])
        print(str(tree.val) + '-all-' + str(resultlist))
        return tmpList

    def treeToDoubleList(self, root: Optional[TreeNode]) -> int:
        resultList = list()
        self.subtreecalc(root, resultList)
        print(resultList.count(targetSum))
        return resultList.count(targetSum)

    def main(self):
        treeNode = TreeNode(10, TreeNode(5, TreeNode(3, TreeNode(3), TreeNode(-2)), TreeNode(2, None, TreeNode(1))),
                 TreeNode(-3, None, TreeNode(11)))
        self.treeToDoubleList(treeNode, 8)

if __name__ == '__main__':
    Solution().main()
