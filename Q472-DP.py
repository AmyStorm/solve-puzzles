#0-alpha
#1-'-'
class Solution(object):

    def match(self, base, word, wordi, index):
        if wordi == '':
            compiled = word
        else:
            head = word[:index + 1]
            replaced = head.replace(wordi, '-' * len(wordi))
            compiled = replaced + word[index + 1:]
        compiledStrings = list(compiled)
        matches = []
        for csi in range(len(compiledStrings)):
            if compiledStrings[csi] == '-':
                matches.append(csi)
        # matches = [each.start() for each in re.finditer('-', compiled)]
        listCompiled = list(compiled)
        countbase = 0
        countnew = 0
        for i in range(len(base)):
            if base[i] == '-':
                countbase += 1
                if i not in matches:
                    listCompiled[i] = '-'
            if listCompiled[i] == '-':
                countnew += 1
        if countbase > countnew:
            return base
        else:
            return ''.join(listCompiled)


    '''
        return the smaller one
    '''
    def compare(self, a, b):
        lengthA = len(a.replace('-', ''))
        lengthB = len(b.replace('-', ''))
        if lengthA <= lengthB:
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
        result = []
        final = []
        # word 1 ~ N
        #for wordIndex in range(0, length - 1):
        for wordIndex in range(0, length - 1):
            word = wordSortedList[wordIndex]
            if word == '':
                continue
            t = []
            dp = []
            lenword = len(word)
            for col in range(lenword):
                t.append(word)
            dp.append(t.copy())
            #pack start
            #dp = [[word for col in range(lenword)] for row in range(length)]
            for i in range(length - 1, wordIndex, -1):
                dp.append(t.copy())
                si = length - 1 - i
                wordi = wordSortedList[i]
                lengthi = len(wordi)

                for w in range(0, lenword):
                    # select
                    #print(dp[i][w] + '_' + str(len(dp[i][w])) + '_' + str(lengthi) + '_' + str(w))

                    if w - lengthi >= 0:
                        # print('**' + wordi)
                        # tmp = self.match(dp[si][w][:w + 1], wordi)
                        dp[si + 1][w] = self.match(dp[si][w], word, wordi, w)
                        # dp[si + 1][w] = self.compare(dp[si][w], tmp)
                        # if word == 'bilt' and (wordi == 'il' or wordi == 'b' or wordi == 't'):
                        #     print(tmp + '__' + str(dp[si][w][:w + 1]) + '__' + dp[si][w][w + 1:] + '__' + str(
                        #         w) + '__' + wordi + '__' + dp[si + 1][w])
                    # not select
                    dp[si + 1][w] = self.compare(dp[si + 1][w], dp[si][w])
                if dp[si + 1][lenword - 1].replace('-', '') == '':
                    result.append(word)
                    break
            print(dp)
        for r in words:
            if result.count(r) > 0:
                final.append(r)
        return final

    def main(self):
        print(self.findAllConcatenatedWordsInADict(["catad", "cat", "cats", "catsdogcats", "dog", "dogcatsdog", "hippopotamuses", "rat", "ratcatdogcat", "catadogcat"]))


if __name__ == '__main__':
    Solution().main()
