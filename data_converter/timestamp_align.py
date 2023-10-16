from datetime import datetime, timedelta  # 导入datetime和timedelta模块

def timestamp_to_beijing_time(timestamp):  # 将时间戳转换为北京时间字符串的函数
    dt_object = datetime(1970, 1, 1) + timedelta(microseconds=int(timestamp) // 1000)  # 计算datetime对象
    beijing_time = dt_object + timedelta(hours=8)  # 北京时间比UTC时间多8个小时
    beijing_time_str = beijing_time.strftime('%Y%m%d%H%M%S%f')  # 格式化为字符串，包括年月日时分秒微秒
    beijing_time_str = beijing_time_str[0:15]  # 只取前15位，即年月日时分秒
    return beijing_time_str

def convert_first_column(input_file):  # 将文件的第一列时间戳转换为北京时间的函数
    converted_lines = []  # 存储转换后的行
    with open(input_file, 'r') as infile:  # 打开输入文件以读取内容
        for line in infile:  # 遍历文件中的每一行
            data = line.strip().split(',')  # 去除换行符，并根据逗号分割行数据为列表
            if len(data) > 0:  # 如果列表不为空
                timestamp = data[0]  # 获取第一个元素（时间戳）
                try:
                    timestamp_int = int(timestamp)  # 尝试将时间戳转换为整数
                    beijing_time = timestamp_to_beijing_time(timestamp_int)  # 转换为北京时间字符串
                    data[0] = beijing_time  # 使用转换后的北京时间字符串更新第一个元素
                except ValueError:
                    # 处理时间戳无法转换为整数的情况
                    pass
                converted_line = ','.join(data) + '\n'  # 将列表元素连接起来，并添加换行符
                converted_lines.append(converted_line)  # 将转换后的行添加到转换行列表中

    # 将转换后的行写回原始文件
    with open(input_file, 'w') as outfile:
        outfile.writelines(converted_lines)

# 示例用法
input_file = "/home/slj/Documents/workspace/ThomasVision/data/smart_lane/Odometry.txt"  # 输入文件路径
convert_first_column(input_file)  # 调用函数进行转换
