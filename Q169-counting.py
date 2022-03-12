from typing import List


class Solution:
    def majorityElement(self, nums: List[int]) -> int:
        protential = nums[0]
        vote = 1
        for u in nums:
            if u == protential:
                vote += 1
            else:
                vote -= 1

            if vote <= 0:
                protential = u
                vote = 1
        return protential

    def main(self):
        print(self.majorityElement([1, 2, 3, 3, 3]))

if __name__ == '__main__':
    Solution().main()
