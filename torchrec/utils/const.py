"""全局常量"""
import os
import socket

hostname = socket.gethostname()
print(hostname)

# 工作路径常量
if hostname == 'yanshihongdeMacBook-Pro.local':
    WORK_DIR = '/Users/yansh/Work'  # 工作目录(MacBook Pro)
elif hostname == 'thuir103':
    WORK_DIR = '/work/yansh15'  # 工作目录(cpu103)
else:
    WORK_DIR = '/work/yansh'  # 工作目录(deep/deeper/deepest)

# 数据路径常量
RAW_DATA_DIR = os.path.join(WORK_DIR, 'RawData')  # 原始数据目录
DATASET_DIR = os.path.join(WORK_DIR, 'Dataset')  # 过滤并格式化数据目录(通常包含interaction.csv, user.csv, item.csv)

BASE_INTERACTION_CSV = 'base_interaction.csv'  # 用户ID、物品ID、评分、标签、时间戳信息
BASE_INTERACTION_PKL = 'base_interaction.pkl'  # 用户ID、物品ID、评分、标签、时间戳信息
INTERACTION_CSV = 'interaction.csv'  # 用户ID、物品ID、评分、标签、时间戳、上下文特征（可选）信息
INTERACTION_PKL = 'interaction.pkl'  # 用户ID、物品ID、评分、标签、时间戳、上下文特征（可选）信息
USER_CSV = 'user.csv'  # 用户ID、用户特征（可选）信息
USER_PKL = 'user.pkl'  # 用户ID、用户特征（可选）信息
ITEM_CSV = 'item.csv'  # 物品ID、物品特征（可选）信息
ITEM_PKL = 'item.pkl'  # 物品ID、物品特征（可选）信息
DESCRIPTION_TXT = "description.txt"  # 数据集统计信息
DESCRIPTION_PKL = "description.pkl"  # 数据集统计信息

STATISTIC_DIR = "STATISTIC"

USER_POS_HIS_SET_DICT_PKL = 'user_pos_his_set_dict.pkl'  # 正向交互信息构成集合按 uid 合并文件
USER_NEG_HIS_SET_DICT_PKL = 'user_neg_his_set_dict.pkl'  # 负向交互信息构成集合按 uid 合并文件

HISTORY_DIR = 'HISTORY'

POS_HIS_CSV_TEMPLATE = 'pos_his_%d.csv'  # 正向交互历史信息
POS_HIS_NPY_TEMPLATE = 'pos_his_%d.npy'  # 正向交互历史信息
NEG_HIS_CSV_TEMPLATE = 'neg_his_%d.csv'  # 负向交互历史信息
NEG_HIS_NPY_TEMPLATE = 'neg_his_%d.npy'  # 负向交互历史信息

NEGATIVE_SAMPLE_DIR = "NEGATIVE_SAMPLE"

DEV_NEG_CSV_TEMPLATE = 'dev_neg_%d.csv'  # 验证集负采样文件
DEV_NEG_NPY_TEMPLATE = 'dev_neg_%d.npy'  # 验证集负采样文件
TEST_NEG_CSV_TEMPLATE = 'test_neg_%d.csv'  # 测试集负采样文件
TEST_NEG_NPY_TEMPLATE = 'test_neg_%d.npy'  # 测试集负采样文件

SPLIT_INDEX_DIR = "SPLIT_INDEX"

SEQUENTIAL_SPLIT_NAME_TEMPLATE = "seq_split_%d_%.2f"  # %d：暖用户要求，%.2f：验证/测试集比例
LEAVE_K_OUT_SPLIT_NAME_TEMPLATE = "leave_k_out_%d_%d"  # %d：暖用户要求，%d：留出个数

TRAIN_INDEX_CSV_TEMPLATE = "%s.train_index.csv"  # 训练集对应的interaction_df的index，%s：划分名称
TRAIN_INDEX_NPY_TEMPLATE = "%s.train_index.npy"  # 训练集对应的interaction_df的index，%s：划分名称
DEV_INDEX_CSV_TEMPLATE = "%s.dev_index.csv"  # 验证集集对应的interaction_df的index，%s：划分名称
DEV_INDEX_NPY_TEMPLATE = "%s.dev_index.npy"  # 验证集集对应的interaction_df的index，%s：划分名称
TEST_INDEX_CSV_TEMPLATE = "%s.test_index.csv"  # 测试集对应的interaction_df的index，%s：划分名称
TEST_INDEX_NPY_TEMPLATE = "%s.test_index.npy"  # 测试集对应的interaction_df的index，%s：划分名称

# 数据格式相关常量
SEP = '\t'
SEQ_SEP = ","

# 数据列字符串常量
INDEX = "index"
UID = 'uid'
IID = 'iid'
RATE = "rate"
LABEL = 'label'
TIME = 'time'

# 数据集描述词典常量
# 0.数据集描述
INFO = "info"

# 1.基本数据列/特征列相关
BASE_FEATURES = "base_features"
CONTEXT_FEATURES = "context_features"
USER_FEATURES = "user_features"
ITEM_FEATURES = "item_features"

FEATURE_NAME = "feature_name"
FEATURE_TYPE = "feature_type"

# 特征列类型
NUMERIC_COLUMN = "numeric"
CATEGORICAL_COLUMN = "categorical"
NUMERIC_LIST_COLUMN = "numeric_list"
CATEGORICAL_LIST_COLUMN = "categorical_list"

# 特征列描述常量
BUCKET_BOUNDARIES = "bucket_boundaries"
BUCKET_LOG_BASE = "bucket_log_base"
INT_MAP = "int_map"

# 2.用户交互相关
USER_INTERACTION = "user_interaction"

POSITIVE = "positive"
NEGATIVE = "negative"
ALL = "all"
MIN = "min"
MAX = "max"
MEAN = "mean"
MEDIAN = "median"
STD = "std"
