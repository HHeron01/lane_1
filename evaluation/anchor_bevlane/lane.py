from scipy.interpolate import InterpolatedUnivariateSpline  # 导入InterpolatedUnivariateSpline类

"""
提供了获取车道点的x坐标、迭代车道点、检查点边界等功能
"""
class Lane:
    def __init__(self, points=None, invalid_value=-2., metadata=None):
        super(Lane, self).__init__()  # 调用父类的构造函数

        self.curr_iter = 0  # 当前迭代器的位置
        self.points = points  # 表示车道的点的二维数组
        self.invalid_value = invalid_value  # 无效值，默认为-2
        self.function = InterpolatedUnivariateSpline(points[:, 1], points[:, 0], k=min(3, len(points) - 1))
        # 根据车道的点创建一个插值函数，x坐标为点的y坐标，y坐标为点的x坐标
        self.min_y = points[:, 1].min() - 0.01  # 点的y坐标的最小值减去0.01
        self.max_y = points[:, 1].max() + 0.01  # 点的y坐标的最大值加上0.01

        self.metadata = metadata or {}  # 元数据字典，默认为空字典

    def __repr__(self):
        return '[Lane]\n' + str(self.points) + '\n[/Lane]'  # 返回表示车道的点的字符串形式

    def __call__(self, lane_ys):
        lane_xs = self.function(lane_ys)  # 使用插值函数计算给定y坐标时的x坐标

        lane_xs[(lane_ys < self.min_y) | (lane_ys > self.max_y)] = self.invalid_value
        # 将超出y坐标范围的x坐标设置为无效值
        return lane_xs  # 返回计算得到的x坐标数组

    def __iter__(self):
        return self  # 返回迭代器本身

    def __next__(self):
        if self.curr_iter < len(self.points):  # 如果还有点未迭代完
            self.curr_iter += 1  # 更新当前迭代器位置
            return self.points[self.curr_iter - 1]  # 返回当前点
        self.curr_iter = 0  # 重置迭代器位置
        raise StopIteration  # 抛出StopIteration异常，表示迭代结束
