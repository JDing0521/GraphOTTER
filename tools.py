import logging
from logging.handlers import TimedRotatingFileHandler
import pandas as pd
import os

# 日志
def logger_config(logging_name,logging_path):
    '''
    配置log
    :param log_path: 输出log路径
    :param logging_name: 记录中name，可随意
    :return:
    '''
    '''
    logger是日志对象，handler是流处理器，console是控制台输出（没有console也可以，将不会在控制台输出，会在日志文件中输出）
    '''
    # 获取logger对象,取名
    logger = logging.getLogger(logging_name)
    # 输出DEBUG及以上级别的信息，针对所有输出的第一层过滤
    logger.setLevel(level=logging.DEBUG)
    # 获取文件日志句柄并设置日志级别，第二层过滤
    # handler = logging.FileHandler(settings.LOG_PATH, encoding='UTF-8')
    handler = TimedRotatingFileHandler(filename=logging_path, when="D", interval=1, backupCount=7,encoding='utf-8')
    handler.setLevel(logging.INFO)
    # 生成并设置文件日志格式
    formatter = logging.Formatter("%(asctime)s %(pathname)s %(filename)s %(funcName)s %(lineno)s \
      %(levelname)s - %(message)s", "%Y-%m-%d %H:%M:%S")
    handler.setFormatter(formatter)
    # # console相当于控制台输出，handler文件输出。获取流句柄并设置日志级别，第二层过滤
    console = logging.StreamHandler()
    console.setLevel(logging.DEBUG)
    # 为logger对象添加句柄
    logger.addHandler(handler)
    logger.addHandler(console)
    return logger

def table2markdown(table_list):
    header = table_list[0]
    table_head = '| ' + ' | '.join(header) + ' |' + "\n"  # 拼接表头
    # 获取表格主体
    table_body = table_list[1:]
    new_table_body = []
    # 将每一个列表项转换为字符串
    for i in table_body:
        row = []
        for j in i:  # 对这一行的遍历
            row.append(str(j))  # 转换为字符串并加入row列表
        new_table_body.append(row)  # 再将row加入new_table_body
    # 拼接列表主体
    table_body = '\n'.join(['| ' + ' | '.join(i) + ' |' for i in new_table_body])
    # 制作列表分隔符
    table_split = '| --- ' * len(header) + ' |' + "\n"
    # 拼接成table变量
    table = table_head + table_split + table_body

    return table
def table2Tuple(table):
    cells = []
    for r in range(len(table)):
        row_tuple = []
        for c in range(len(table[r])):
            row_tuple.append((r,c,table[r][c]))
        row_tuple = '\t'.join([str(i) for i in row_tuple])
        cells.append(row_tuple)
    cells = '\n'.join(cells)
    return str(cells)


def mkdir(path):
    folder = os.path.exists(path)
    if not folder:  # 判断是否存在文件夹如果不存在则创建为文件夹
        os.makedirs(path)  # makedirs 创建文件时如果路径不存在会创建这个路径
    else:
        pass



