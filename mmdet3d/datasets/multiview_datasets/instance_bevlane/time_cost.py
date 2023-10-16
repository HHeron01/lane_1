import time
from pprint import pprint

__all__ = ['TimeCost']


class TimeCost(object):
    """
    计时器类，用于测量代码的时间消耗。
    """
    def __init__(self, close=True):
        """
        初始化计时器对象。
        Args:
            close (bool): 是否启用计时器。默认为True，如果设置为False，将禁用计时器。
        """
        self.close = close
        self.tag_list = []  # 存储时间标签的列表
        self.t0 = time.time()  # 记录初始化时刻的时间戳
        self.add_tag("t0", self.t0)  # 添加初始标签"t0"，记录初始时间

    def clear(self):
        """
        清除所有时间标签和重置初始时间。
        """
        self.tag_list.clear()
        self.t0 = time.time()
        self.add_tag("t0", self.t0)

    def add_tag(self, tag_name, tag_time=None):
        """
        添加一个时间标签。
        Args:
            tag_name (str): 时间标签的名称。
            tag_time (float, optional): 时间标签的时间戳。如果未提供，默认为当前时间。
        Notes:
            如果计时器已关闭（close=True），则不会添加时间标签。
        """
        if self.close:
            return
        if tag_time is None:
            tag_time = time.time()
        self.tag_list.append(
            {
                "tag_name": tag_name,
                "tag_time": tag_time,
            }
        )

    def print_tag(self):
        """
        打印所有时间标签。

        Notes:
            如果计时器已关闭（close=True），则不会打印时间标签。
        """
        if self.close:
            return
        for t in self.tag_list:
            pprint(t)

    def print_time_cost(self):
        """
        打印各时间标签之间的时间消耗。
        Notes:
            如果计时器已关闭（close=True），则不会打印时间消耗。
        """
        if self.close:
            return
        last_t = self.t0
        for i, tag in enumerate(self.tag_list):
            tn = tag["tag_name"]
            tt = tag["tag_time"]
            print("time cost {:0.4f}s, \t tag is {}".format(tt - last_t, tn))
            last_t = tt
        print("-" * 100)
        self.clear()


"""
TEST FUNCTION
"""


def test():
    tc = TimeCost(close=False)
    tc.add_tag('t1')
    for i in range(9999999):
        n = i * i
    tc.add_tag('tttttt2')

    for i in range(99999999):
        n = i * i
    tc.add_tag('t3')

    tc.print_tag()
    tc.print_time_cost()


if __name__ == "__main__":
    test()
