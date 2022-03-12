from numpy import array

"""
not work
"""
class Solution(object):

    def match(self, a, b):
        a = str(a)
        # return a.replace(b, '-')
        if a.find(b) != -1:
            return 1
        else:
            return 0

    '''
        return the smaller one
    '''

    def compare(self, a, b):
        # lengthA = len(a.replace('-', ''))
        # lengthB = len(b.replace('-', ''))
        # if lengthA < lengthB:
        #     return a
        # else:
        #     return b
        if a > b:
            return a
        else:
            return b

    def findAllConcatenatedWordsInADict(self, words):
        """
        :type words: List[str]
        :rtype: List[str]
        """
        wordSortedList = sorted(words, key=lambda j: len(j), reverse=True)
        length = len(wordSortedList)
        final = []
        result = []
        # word 1 ~ N
        # for wordIndex in range(0, length - 1):
        for wordIndex in range(0, length - 1):
            word = wordSortedList[wordIndex]
            t = []
            dp = [[0 for col in range(len(word))] for row in range(length + 1)]
            # pack start
            # dp = [[word for col in range(len(word))] for row in range(length)]
            for i in range(wordIndex + 1, length):

                wordi = wordSortedList[i]
                lengthi = len(wordi)
                for w in range(0, len(word)):
                    # select
                    # print(str(dp[i][w]) + '_' + str(lengthi) + '_' + str(w))

                    if w - lengthi >= 0:
                        tmp = self.match(word, wordi)
                        dp[i + 1][w] = self.compare(dp[i][w], dp[i][w] + tmp)
                    # not select
                    dp[i + 1][w] = self.compare(dp[i + 1][w], dp[i][w])
                if dp[i + 1][len(word) - 1] >= 2:
                    result.append(word)
                    break
            # if dp[length][len(word) - 1].replace('-', '') == '':
            #     result.append(word)
        for r in words:
            if result.count(r) > 0:
                final.append(r)
        return final
