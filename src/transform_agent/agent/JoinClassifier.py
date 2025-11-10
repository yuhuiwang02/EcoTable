import os
import pandas as pd
import numpy as np
import networkx as nx
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, precision_score, recall_score
from sklearn.preprocessing import StandardScaler
from sentence_transformers import SentenceTransformer
from fuzzywuzzy import fuzz
from scipy.stats import wasserstein_distance
from typing import List, Tuple, Dict, Set, Optional
import time
import traceback
import re
import logging
import argparse
import json
import shutil
import random
from pathlib import Path
from collections import defaultdict
import concurrent.futures
from torch.cuda import amp
import heapq


# 配置日志系统
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler()
    ]
)
logger = logging.getLogger('AutoBI')

# 设备配置：优先使用GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
logger.info(f"使用设备: {device}")
if torch.cuda.is_available():
    logger.info(f"CUDA设备: {torch.cuda.get_device_name(0)}")

# ------------------------------
# 1. 数据结构定义
# ------------------------------

class Table:
    """表示一个数据表，包含表名和列数据"""
    def __init__(self, table_name: str, columns: Dict[str, List]):
        self.table_name = table_name
        self.columns = columns
        self.col_names = list(columns.keys())
        
    def __repr__(self):
        row_count = len(next(iter(self.columns.values()))) if self.columns else 0
        return f"Table({self.table_name}, {len(self.columns)}列, {row_count}行)"
    
    def get_row_count(self) -> int:
        """获取表的行数"""
        if not self.columns:
            return 0
        return len(next(iter(self.columns.values())))

class BIModel:
    """表示一个完整的BI模型，包含多个表和它们之间的连接关系"""
    def __init__(self, tables: List[Table], joins: List[Tuple[Tuple[Table, str], Tuple[Table, str]]]):
        self.tables = tables
        self.joins = joins
        
    def __repr__(self):
        return f"BIModel({len(self.tables)}表, {len(self.joins)}连接)"

# ------------------------------
# 2. 数据加载与预处理（优化版）
# ------------------------------

def load_tables_from_dir(table_dir: str) -> Dict[str, Table]:
    """从目录加载所有CSV表文件（优化版）"""
    tables = {}
    table_path = Path(table_dir)
    
    if not table_path.exists():
        logger.error(f"表目录不存在: {table_path}")
        return tables
        
    # 批量加载所有CSV文件
    csv_files = list(table_path.glob("*.csv"))
    logger.info(f"找到 {len(csv_files)} 个CSV文件")
    
    # 直接处理所有表，不区分大小表
    for filepath in csv_files:
        try:
            # 读取CSV文件
            df = pd.read_csv(filepath)
            table_name = filepath.stem
            
            # 创建表对象
            columns = {col: df[col].tolist() for col in df.columns}
            tables[table_name] = Table(table_name, columns)
            
            logger.info(f"加载表: {table_name} ({len(df)}行, {len(df.columns)}列)")
        except Exception as e:
            logger.error(f"加载表{filepath.name}失败: {e}")
    
    logger.info(f"加载完成: {len(tables)} 张表")
    return tables

def load_links_csv(links_file: str, tables: Dict[str, Table]) -> BIModel:
    """
    加载连接关系CSV文件（增强健壮性）
    支持无列头格式：表1,表2,列1,列2,概率
    """
    try:
        links_path = Path(links_file)
        if not links_path.exists():
            logger.error(f"连接文件不存在: {links_path}")
            return BIModel([], [])
        
        # 尝试不同编码读取文件
        encodings = ['utf-8', 'latin-1', 'iso-8859-1', 'cp1252', 'gbk']
        links_df = None
        
        for encoding in encodings:
            try:
                # 尝试读取前2行检测格式
                df_sample = pd.read_csv(links_path, nrows=2, header=None, encoding=encoding)
                if len(df_sample.columns) >= 4:  # 至少需要4列：表1,表2,列1,列2
                    links_df = pd.read_csv(links_path, header=None, encoding=encoding)
                    logger.info(f"使用编码: {encoding} 成功读取连接文件（无列头）")
                    break
            except (UnicodeDecodeError, pd.errors.ParserError) as e:
                logger.debug(f"编码 {encoding} 失败: {e}")
                continue
        
        if links_df is None:
            try:
                # 最后尝试默认编码
                links_df = pd.read_csv(links_path, header=None)
                logger.info("使用默认编码和无header方式读取连接文件")
            except Exception as e:
                logger.error(f"无法读取连接文件: {e}")
                return BIModel([], [])
        
        # 智能检测列位置
        column_positions = {
            'table1': 0,
            'table2': 1,
            'column1': 2,
            'column2': 3,
            'prob': 4 if len(links_df.columns) > 4 else None
        }
        
        # 收集所有相关表
        tables_in_model = set()
        joins = []
        skipped_lines = 0
        
        # 处理每一行
        for idx, row in links_df.iterrows():
            try:
                # 获取表名（移除.csv后缀）
                table1_name = str(row[column_positions['table1']]).strip()
                table2_name = str(row[column_positions['table2']]).strip()
                
                # 移除可能的.csv后缀
                if table1_name.endswith('.csv'):
                    table1_name = table1_name[:-4]
                if table2_name.endswith('.csv'):
                    table2_name = table2_name[:-4]
                
                # 获取列名
                col1_name = str(row[column_positions['column1']]).strip()
                col2_name = str(row[column_positions['column2']]).strip()
                
                # 获取概率（如果存在）
                prob =  1.0
                
                # 跳过空表名
                if not table1_name or not table2_name:
                    logger.warning(f"跳过空表名行: {idx}")
                    skipped_lines += 1
                    continue
                
                tables_in_model.add(table1_name)
                tables_in_model.add(table2_name)
                
                table1 = tables.get(table1_name)
                table2 = tables.get(table2_name)
                
                if table1 and table2:
                    # 检查列是否存在
                    if col1_name not in table1.columns:
                        logger.warning(f"表 {table1_name} 中不存在列 {col1_name}")
                        col1_name = table1.col_names[0]  # 使用第一列作为备用
                    
                    if col2_name not in table2.columns:
                        logger.warning(f"表 {table2_name} 中不存在列 {col2_name}")
                        col2_name = table2.col_names[0]  # 使用第一列作为备用
                    
                    joins.append(((table1, col1_name), (table2, col2_name)))
                else:
                    missing_tables = []
                    if not table1:
                        missing_tables.append(table1_name)
                    if not table2:
                        missing_tables.append(table2_name)
                    logger.warning(f"跳过无效连接: {', '.join(missing_tables)} 未找到")
                    skipped_lines += 1
            except Exception as e:
                logger.error(f"处理行 {idx} 时出错: {e}")
                skipped_lines += 1
                continue
        
        # 创建表列表
        table_objs = [tables[name] for name in tables_in_model if name in tables]
        
        logger.info(f"加载连接关系: 找到 {len(table_objs)} 张表, {len(joins)} 个连接, 跳过 {skipped_lines} 行")
        return BIModel(table_objs, joins)
    except Exception as e:
        logger.error(f"加载连接关系失败: {e}")
        traceback.print_exc()
        return BIModel([], [])

# ------------------------------
# 3. 本地连接预测（优化版）
# ------------------------------

class JoinDataset(Dataset):
    """PyTorch数据集，用于加载特征和标签"""
    def __init__(self, features, labels):
        self.features = torch.tensor(features, dtype=torch.float32)
        self.labels = torch.tensor(labels, dtype=torch.float32)
    
    def __len__(self):
        return len(self.features)
    
    def __getitem__(self, idx):
        return self.features[idx], self.labels[idx]

class JoinNet(nn.Module):
    """用于连接预测的神经网络（支持GPU）"""
    def __init__(self, input_dim):
        super(JoinNet, self).__init__()
        self.layers = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.BatchNorm1d(128),
            nn.Dropout(0.3),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
            # 移除 Sigmoid 激活函数，因为 BCEWithLogitsLoss 包含它
        )
    
    def forward(self, x):
        return self.layers(x)

class LocalJoinClassifier:
    """本地连接分类器，负责训练和预测列之间的连接关系"""
    def __init__(self, fact_threshold=100):
        # 初始化BERT模型并移至GPU
        self.bert = SentenceTransformer('/data/new_dbt_dataset/JoinClassify/all-mpnet-base-v2-local').to(device)
        
        # 模型和预处理对象
        self.n1_model = None  # N:1连接模型
        self.o1_model = None  # 1:1连接模型
        self.n1_scaler = None
        self.o1_scaler = None
        
        # 配置参数
        self.col_name_cache={}
        self.fact_threshold = fact_threshold  # 事实表阈值（行数）
        self.col_frequency = defaultdict(int)  # 列名频率统计
        self.feature_cache={} #嵌入缓存
        self.tables = []  # 存储所有表引用
        self.logger = logging.getLogger('LocalJoinClassifier')
        self.logger.info(f"初始化本地分类器，事实表阈值={fact_threshold}行")

    def tokenize_name(self, name: str) -> List[str]:
        """将列名/表名拆分为token"""
        name = re.sub(r'[^a-zA-Z0-9]', ' ', name)
        name = re.sub(r'([a-z])([A-Z])', r'\1 \2', name)
        tokens = [token.lower() for token in name.split() if token]
        return tokens

    def _str_similarity(self, s1: str, s2: str) -> Dict:
        """计算字符串相似度特征"""
        try:
            return {
                "jaccard": fuzz.token_sort_ratio(s1, s2) / 100,
                "edit_dist": fuzz.ratio(s1, s2) / 100
            }
        except Exception as e:
            self.logger.error(f"字符串相似度计算错误: {e}")
            return {"jaccard": 0, "edit_dist": 0}
        
    def _sample_frequent_values(self, column: List, max_seq_length=512):
        """
        基于频率采样列值（论文3.2节）
        选择最频繁的单元格值，使总token数不超过max_seq_length
        """
        from collections import Counter
        import math
        
        # 计算值的频率
        value_freq = Counter(column)
        distinct_values = list(value_freq.keys())
        
        # 按频率降序排序
        distinct_values.sort(key=lambda x: value_freq[x], reverse=True)
        
        # 初始化
        sampled_values = []
        current_token_count = 0
        
        # 使用PLM的tokenizer
        tokenizer = self.bert.tokenizer
        
        for value in distinct_values:
            # 将值转换为字符串并tokenize
            value_str = str(value)
            tokens = tokenizer.tokenize(value_str)
            token_count = len(tokens)
            
            # 检查添加后是否超过限制（考虑添加逗号）
            if sampled_values:
                token_count += 1  # 为逗号添加一个token
            
            if current_token_count + token_count <= max_seq_length:
                sampled_values.append(value_str)
                current_token_count += token_count
            else:
                break
                
        return sorted(sampled_values, key=lambda x: tokenizer.tokenize(x))
    
    def _create_text_sequence(self, table: Table, column_name: str, sampled_values: List[str]) -> str:
        """
        创建文本序列（title-colname-stat-col模式）
        格式: "表标题.列名 包含n个值(最大长度,最小长度,平均长度): 值1,值2,..."
        """
        # 获取列数据
        column_data = table.columns[column_name]
        
        # 计算统计信息
        n = len(set(column_data))  # 唯一值数量
        value_lengths = [len(str(x)) for x in column_data]
        max_len = max(value_lengths) if value_lengths else 0
        min_len = min(value_lengths) if value_lengths else 0
        avg_len = sum(value_lengths) / len(value_lengths) if value_lengths else 0
        
        # 构建文本序列
        text_sequence = (
            f"{table.table_name}.{column_name} contains {n} values "
            f"({max_len},{min_len},{avg_len:.1f}): "
            f"{', '.join(sampled_values)}"
        )
        return text_sequence
    
    def get_columns_cosine_similarity(
        self, 
        table1: Table, col1_name: str, 
        table2: Table, col2_name: str
    ) -> float:
        """
        计算两个列之间的余弦相似度（使用title-colname-stat-col模式）
        
        步骤：
        1. 分别获取两个列的文本序列表示
        2. 生成各自的嵌入向量
        3. 计算余弦相似度
        
        参数:
            table1: 第一个表对象
            col1_name: 第一个列名
            table2: 第二个表对象
            col2_name: 第二个列名
            
        返回:
            两列嵌入向量的余弦相似度（float）
        """
        cache_key1 = (table1.table_name, col1_name)
        cache_key2 = (table2.table_name, col2_name)
        if cache_key1 in self.feature_cache:
            if cache_key2 in self.feature_cache:
                emb1,emb2=self.feature_cache[cache_key1],self.feature_cache[cache_key2]
            else:
                emb1=self.feature_cache[cache_key1]
                col2_data = table2.columns[col2_name]
                sampled_values2 = self._sample_frequent_values(col2_data)
                text_seq2 = self._create_text_sequence(table2, col2_name, sampled_values2)
                emb2=self.bert.encode(text_seq2,convert_to_tensor=True,show_progress_bar=False)
                self.feature_cache[cache_key2]=emb2
            cos_sim = torch.nn.functional.cosine_similarity(
            emb1.unsqueeze(0), 
            emb2.unsqueeze(0), 
            dim=1
            )
            return cos_sim.item()  # 返回Python float类型
        elif cache_key2 in self.feature_cache:
            emb2=self.feature_cache[cache_key2]
            col1_data = table1.columns[col1_name]
            sampled_values1 = self._sample_frequent_values(col1_data)
            text_seq1 = self._create_text_sequence(table1, col1_name, sampled_values1)
            emb1=self.bert.encode(text_seq1,convert_to_tensor=True,show_progress_bar=False)
            self.feature_cache[cache_key1]=emb1
            cos_sim = torch.nn.functional.cosine_similarity(
            emb1.unsqueeze(0), 
            emb2.unsqueeze(0), 
            dim=1
            )
            return cos_sim.item()

        # 1. 获取列数据
        col1_data = table1.columns[col1_name]
        col2_data = table2.columns[col2_name]
        
        # 2. 高频采样（优化性能）
        sampled_values1 = self._sample_frequent_values(col1_data)
        sampled_values2 = self._sample_frequent_values(col2_data)
        
        # 3. 创建文本序列（title-colname-stat-col模式）
        text_seq1 = self._create_text_sequence(table1, col1_name, sampled_values1)
        text_seq2 = self._create_text_sequence(table2, col2_name, sampled_values2)
        
        # 4. 生成嵌入向量（GPU加速）
        with torch.no_grad():
            # 批量处理提高效率
            embeddings = self.bert.encode(
                [text_seq1, text_seq2], 
                convert_to_tensor=True,
                show_progress_bar=False
            )
            emb1, emb2 = embeddings[0], embeddings[1]
        self.feature_cache[cache_key1]=emb1
        self.feature_cache[cache_key2]=emb2
        # 5. 计算余弦相似度（PyTorch原生函数）
        cos_sim = torch.nn.functional.cosine_similarity(
            emb1.unsqueeze(0), 
            emb2.unsqueeze(0), 
            dim=1
        )
        return cos_sim.item()  # 返回Python float类型

    def _embedding_sim(self, s1: str, s2: str) -> float:
        """使用Sentence-BERT计算嵌入相似度"""
        try:
            if s1 not in self.col_name_cache:
                self.col_name_cache[s1] = self.bert.encode(s1, convert_to_tensor=True)
            if s2 not in self.col_name_cache:
                self.col_name_cache[s2] = self.bert.encode(s2, convert_to_tensor=True)
                
            emb1 = self.col_name_cache[s1]
            emb2 = self.col_name_cache[s2]
            return torch.dot(emb1, emb2) / (torch.norm(emb1) * torch.norm(emb2)).item()
        except Exception as e:
            self.logger.error(f"嵌入相似度计算错误: {e}")
            return 0.0

    def _data_features(self, col1: List, col2: List) -> Dict:
        """基于数据内容计算特征"""
        features = {
            "contain1": 0, "contain2": 0, "emd": 0, "distinct1": 0, 
            "distinct2": 0, "range_overlap": 0, "value_length1": 0, 
            "value_length2": 0, "value_type1": 0, "value_type2": 0,
            "col_emb_sim": 0.0
        }
        
        try:
            # 1. 包含度和唯一度特征
            set1, set2 = set(col1), set(col2)
            overlap = len(set1 & set2)
            
            features["contain1"] = overlap / len(set1) if set1 else 0
            features["contain2"] = overlap / len(set2) if set2 else 0
            features["distinct1"] = len(set1) / len(col1) if col1 else 0
            features["distinct2"] = len(set2) / len(col2) if col2 else 0
            
            # 2. EMD距离（仅适用于数值列）
            if all(isinstance(x, (int, float)) for x in col1 + col2):
                try:
                    features["emd"] = wasserstein_distance(col1, col2)
                except:
                    features["emd"] = 0
                
            # 3. 范围重叠（仅适用于数值列）
            if all(isinstance(x, (int, float)) for x in col1 + col2) and col1 and col2:
                min1, max1 = min(col1), max(col1)
                min2, max2 = min(col2), max(col2)
                left = max(min1, min2)
                right = min(max1, max2)
                if left <= right:
                    overlap_length = right - left
                    total_range = max(max1, max2) - min(min1, min2)
                    features["range_overlap"] = overlap_length / total_range if total_range > 0 else 0
                    
            # 4. 值长度特征（平均值）
            def avg_value_length(values):
                if not values: return 0
                total_length = sum(len(str(x)) for x in values)
                return total_length / len(values)
                
            features["value_length1"] = avg_value_length(col1)
            features["value_length2"] = avg_value_length(col2)
            
            # 5. 值类型特征（简单分类）
            def get_value_type(values):
                if not values: return "unknown"
                sample = values[0]
                if isinstance(sample, (int, float)):
                    return "numeric"
                elif isinstance(sample, str):
                    return "string"
                elif isinstance(sample, bool):
                    return "boolean"
                else:
                    return "other"
                    
            features["value_type1"] = 1 if get_value_type(col1) == "numeric" else 0
            features["value_type2"] = 1 if get_value_type(col2) == "numeric" else 0
            for key, value in features.items():
                if np.isnan(value) or np.isinf(value):
                    features[key] = 0
        except Exception as e:
            self.logger.error(f"数据特征提取出错: {e}")
            traceback.print_exc()

        return features

    def extract_features(self, c1: Tuple[Table, str], c2: Tuple[Table, str], is_1to1: bool) -> Dict:
        """提取连接特征"""
        try:
            t1, col1_name = c1
            t2, col2_name = c2
            # 确保列存在
            if col1_name not in t1.columns:
                raise ValueError(f"列 '{col1_name}' 不在表 '{t1.table_name}' 中")
            if col2_name not in t2.columns:
                raise ValueError(f"列 '{col2_name}' 不在表 '{t2.table_name}' 中")
            
            data1 = t1.columns[col1_name]
            data2 = t2.columns[col2_name]
            
            # 更新列名频率统计
            self.col_frequency[col1_name] += 1
            self.col_frequency[col2_name] += 1
            
            # 获取列位置
            pos1 = t1.col_names.index(col1_name) / len(t1.col_names) if col1_name in t1.col_names else 0
            pos2 = t2.col_names.index(col2_name) / len(t2.col_names) if col2_name in t2.col_names else 0

            # 1. 元数据特征
            meta_feat = {}
            
            # 字符串相似度特征
            str_feat = self._str_similarity(col1_name, col2_name)
            meta_feat.update(str_feat)
            
            # 嵌入相似度特征
            emb_feat = {
                "col_name_emb_sim": self._embedding_sim(col1_name, col2_name),
                "table_emb_sim": self._embedding_sim(t1.table_name, t2.table_name),
                "col_emb_sim": self.get_columns_cosine_similarity(t1,col1_name,t2,col2_name)
            }
            meta_feat.update(emb_feat)
            
            # 列名长度特征
            meta_feat["token_count1"] = len(self.tokenize_name(col1_name))
            meta_feat["token_count2"] = len(self.tokenize_name(col2_name))
            meta_feat["char_count1"] = len(col1_name)
            meta_feat["char_count2"] = len(col2_name)
            
            # 列名频率特征
            meta_feat["col_freq1"] = self.col_frequency[col1_name]
            meta_feat["col_freq2"] = self.col_frequency[col2_name]
            
            # 位置特征
            meta_feat["pos1"] = pos1
            meta_feat["pos2"] = pos2
            
            # 2. 数据特征
            data_feat = self._data_features(data1, data2)
            
            # 3. 特定于连接类型的特征
            if not is_1to1:  # N:1连接
                row_ratio = len(data1) / len(data2) if data2 and len(data2) > 0 else 0
                features = {
                    **meta_feat, 
                    **data_feat,
                    "row_ratio": row_ratio,
                    "max_contain": max(data_feat["contain1"], data_feat["contain2"])
                }
            else:  # 1:1连接
                # 表头Jaccard相似度
                header_jaccard = len(set(t1.col_names) & set(t2.col_names)) 
                header_jaccard /= len(set(t1.col_names) | set(t2.col_names)) if (t1.col_names and t2.col_names) else 0
                
                features = {
                    **meta_feat, 
                    **data_feat,
                    "min_contain": min(data_feat["contain1"], data_feat["contain2"]),
                    "header_jaccard": header_jaccard
                }
            
            # 确保所有特征值都是Python原生类型
            for key in list(features.keys()):
                value = features[key]
                if torch.is_tensor(value):
                    features[key] = value.item() if value.numel() == 1 else value.cpu().numpy()
                elif hasattr(value, 'item') and callable(getattr(value, 'item')):
                    features[key] = value.item()
            return features
                
        except Exception as e:
            self.logger.error(f"提取特征时出错 - {t1.table_name}.{col1_name} vs {t2.table_name}.{col2_name}: {e}")
            traceback.print_exc()
            return {}


    def _build_join_graph(self, samples: List[Tuple[Tuple, Tuple, int]]) -> nx.Graph:
        """构建连接图用于传递性分析"""
        graph = nx.Graph()
        # 添加所有正样本连接
        for (c1, c2, label) in samples:
            if label == 1:
                t1, col1_name = c1
                t2, col2_name = c2
                # 使用表名和列名作为键
                key1 = (t1.table_name, col1_name)
                key2 = (t2.table_name, col2_name)
                graph.add_edge(key1, key2)
        return graph

    def apply_transitivity(self, samples: List[Tuple[Tuple, Tuple, int]]) -> List:
        """应用标签传递性"""
        try:
            graph = self._build_join_graph(samples)
            new_samples = []
            visited = set()
            
            # 遍历所有连通分量
            for comp in nx.connected_components(graph):
                comp_list = list(comp)
                
                # 为分量中的每对列添加连接
                for i in range(len(comp_list)):
                    for j in range(i+1, len(comp_list)):
                        col1_info = comp_list[i]
                        col2_info = comp_list[j]
                        
                        # 避免重复添加
                        if (col1_info, col2_info) in visited or (col2_info, col1_info) in visited:
                            continue
                            
                        t1_name, col1_name = col1_info
                        t2_name, col2_name = col2_info
                        
                        # 查找对应的表对象
                        t1 = next((t for t in self.tables if t.table_name == t1_name), None)
                        t2 = next((t for t in self.tables if t.table_name == t2_name), None)
                        
                        if t1 and t2 and col1_name in t1.columns and col2_name in t2.columns:
                            new_samples.append(((t1, col1_name), (t2, col2_name), 1))
                            visited.add((col1_info, col2_info))
            
            self.logger.info(f"应用传递性新增 {len(new_samples)} 个样本")
            return samples + new_samples
            
        except Exception as e:
            self.logger.error(f"应用传递性时出错: {e}")
            traceback.print_exc()
            return samples

    def separate_join_types(self, samples: List[Tuple[Tuple, Tuple, int]]) -> Tuple[List, List]:
        """分离N:1和1:1连接"""
        n1_samples = []
        o1_samples = []
        
        for (c1, c2, label) in samples:
            t1, _ = c1
            t2, _ = c2
            size1 = t1.get_row_count()
            size2 = t2.get_row_count()
            
            # 事实表识别：行数大于等于阈值
            is_fact1 = size1 >= self.fact_threshold
            is_fact2 = size2 >= self.fact_threshold
            
            # 分离连接类型
            if (is_fact1 and not is_fact2) or (not is_fact1 and is_fact2):
                n1_samples.append((c1, c2, label))
            else:
                o1_samples.append((c1, c2, label))
                
        self.logger.info(f"分离连接类型: {len(n1_samples)} 个N:1样本, {len(o1_samples)} 个1:1样本")
        return n1_samples, o1_samples

    def generate_negative_samples(self, bm: BIModel, negative_ratio=0.75, max_negative=10000):
        """高效生成负样本"""
        # 1. 创建快速查找结构
        join_set = set()
        for (t1, col1), (t2, col2) in bm.joins:
            # 规范化连接键（排序表名）
            key = (min(t1.table_name, t2.table_name), max(t1.table_name, t2.table_name), col1, col2)
            join_set.add(key)
        
        # 2. 计算最大负样本数
        num_negative = int(len(bm.joins) * negative_ratio)
        
        # 3. 生成候选负样本
        negative_samples = []
        attempts = 0
        max_attempts = num_negative * 10  # 防止无限循环
        
        while len(negative_samples) < num_negative and attempts < max_attempts:
            attempts += 1
            
            # 随机选择两个不同的表
            t1, t2 = random.sample(bm.tables, 2)
            if t1 == t2:
                continue
                
            # 随机选择列
            col1 = random.choice(t1.col_names)
            col2 = random.choice(t2.col_names)
            
            # 创建规范化键
            key = (min(t1.table_name, t2.table_name), max(t1.table_name, t2.table_name), col1, col2)
            
            # 检查是否真实连接
            if key not in join_set:
                negative_samples.append(((t1, col1), (t2, col2), 0))
        
        return negative_samples

    def batch_extract_features(self, pairs: List[Tuple], is_1to1: bool) -> List[Dict]:
        """批量提取特征（优化版）"""
        features_list = []
        
        # 1. 批量处理元数据特征
        col1_names = [f"{t1.table_name}_{col1}" for (t1, col1), _ in pairs]
        col2_names = [f"{t2.table_name}_{col2}" for _, (t2, col2) in pairs]
        
        # 批量计算列名嵌入相似度
        with torch.no_grad():
            emb1 = self.bert.encode(col1_names, convert_to_tensor=True, batch_size=128)
            emb2 = self.bert.encode(col2_names, convert_to_tensor=True, batch_size=128)
            col_name_sims = torch.cosine_similarity(emb1, emb2)
        
        # 2. 批量提取特征
        for i, ((t1, col1), (t2, col2)) in enumerate(pairs):
            try:
                # 获取列数据
                data1 = t1.columns.get(col1, [])
                data2 = t2.columns.get(col2, [])
                
                # 3. 核心：计算列内容嵌入相似度
                col_content_sim = self.get_columns_cosine_similarity(t1, col1, t2, col2)
                
                # 4. 元数据特征
                meta_feat = {
                    "col_name_sim": col_name_sims[i].item(),
                    "table_sim": self._embedding_sim(t1.table_name, t2.table_name),
                    "col_content_sim": col_content_sim,
                    "token_count1": len(self.tokenize_name(col1)),
                    "token_count2": len(self.tokenize_name(col2)),
                    "char_count1": len(col1),
                    "char_count2": len(col2),
                    "col_freq1": self.col_frequency.get(col1, 0),
                    "col_freq2": self.col_frequency.get(col2, 0),
                    "pos1": t1.col_names.index(col1)/len(t1.col_names) if col1 in t1.col_names else 0,
                    "pos2": t2.col_names.index(col2)/len(t2.col_names) if col2 in t2.col_names else 0
                }
                
                # 5. 数据特征
                data_feat = self._data_features(data1, data2)
                
                # 6. 连接类型特定特征
                if not is_1to1:  # N:1连接
                    row_ratio = len(data1)/len(data2) if data2 else 0
                    features = {
                        **meta_feat,
                        **data_feat,
                        "row_ratio": row_ratio,
                        "max_contain": max(data_feat.get("contain1", 0), data_feat.get("contain2", 0))
                    }
                else:  # 1:1连接
                    header_jaccard = len(set(t1.col_names) & set(t2.col_names)) 
                    header_jaccard /= len(set(t1.col_names) | set(t2.col_names)) if t1.col_names and t2.col_names else 0
                    features = {
                        **meta_feat,
                        **data_feat,
                        "min_contain": min(data_feat.get("contain1", 0), data_feat.get("contain2", 0)),
                        "header_jaccard": header_jaccard
                    }
                
                # 7. 确保特征值类型正确
                for key in list(features.keys()):
                    value = features[key]
                    if torch.is_tensor(value):
                        features[key] = value.item() if value.numel() == 1 else value.cpu().numpy()
                    elif hasattr(value, 'item') and callable(getattr(value, 'item')):
                        features[key] = value.item()
                
                features_list.append(features)
                
            except Exception as e:
                self.logger.error(f"处理列对 {t1.table_name}.{col1} ↔ {t2.table_name}.{col2} 时出错: {e}")
                features_list.append({})
        
        return features_list
    def _plot_learning_curves(self, train_losses, val_losses, 
                         train_f1_scores, val_f1_scores,
                         train_precisions, val_precisions,
                         train_recalls, val_recalls,
                         model_type):
        """绘制学习曲线"""
        try:
            import matplotlib.pyplot as plt
            
            epochs = range(1, len(train_losses) + 1)
            
            # 创建2x2的子图
            fig, axes = plt.subplots(2, 2, figsize=(15, 10))
            
            # 绘制损失曲线
            axes[0, 0].plot(epochs, train_losses, 'b-', label='训练损失')
            axes[0, 0].plot(epochs, val_losses, 'r-', label='验证损失')
            axes[0, 0].set_title('损失曲线')
            axes[0, 0].set_xlabel('Epoch')
            axes[0, 0].set_ylabel('Loss')
            axes[0, 0].legend()
            axes[0, 0].grid(True)
            
            # 绘制F1分数曲线
            axes[0, 1].plot(epochs, train_f1_scores, 'b-', label='训练F1')
            axes[0, 1].plot(epochs, val_f1_scores, 'r-', label='验证F1')
            axes[0, 1].set_title('F1分数曲线')
            axes[0, 1].set_xlabel('Epoch')
            axes[0, 1].set_ylabel('F1 Score')
            axes[0, 1].legend()
            axes[0, 1].grid(True)
            
            # 绘制精确率曲线
            axes[1, 0].plot(epochs, train_precisions, 'b-', label='训练精确率')
            axes[1, 0].plot(epochs, val_precisions, 'r-', label='验证精确率')
            axes[1, 0].set_title('精确率曲线')
            axes[1, 0].set_xlabel('Epoch')
            axes[1, 0].set_ylabel('Precision')
            axes[1, 0].legend()
            axes[1, 0].grid(True)
            
            # 绘制召回率曲线
            axes[1, 1].plot(epochs, train_recalls, 'b-', label='训练召回率')
            axes[1, 1].plot(epochs, val_recalls, 'r-', label='验证召回率')
            axes[1, 1].set_title('召回率曲线')
            axes[1, 1].set_xlabel('Epoch')
            axes[1, 1].set_ylabel('Recall')
            axes[1, 1].legend()
            axes[1, 1].grid(True)
            
            plt.tight_layout()
            
            # 保存图像
            os.makedirs("learning_curves", exist_ok=True)
            curve_path = os.path.join("learning_curves", f"{model_type}_learning_curves.png")
            plt.savefig(curve_path)
            plt.close()
            
            self.logger.info(f"学习曲线已保存至: {curve_path}")
            
        except ImportError:
            self.logger.warning("无法导入matplotlib，跳过学习曲线绘制")
        except Exception as e:
            self.logger.error(f"绘制学习曲线时出错: {e}")
    def _is_n1_sample(self, c1: Tuple, c2: Tuple) -> bool:
        """判断样本是否为N:1类型"""
        t1, _ = c1
        t2, _ = c2
        size1 = t1.get_row_count()
        size2 = t2.get_row_count()
        is_fact1 = size1 >= self.fact_threshold
        is_fact2 = size2 >= self.fact_threshold
        return (is_fact1 and not is_fact2) or (not is_fact1 and is_fact2)
    def _evaluate_samples(self, samples, model, scaler, is_1to1):
        """评估模型在样本上的性能"""
        X = []
        y = []
        
        for (c1, c2, label) in samples:
            feat = self.extract_features(c1, c2, is_1to1)
            if feat:
                feat_values = [v.item() if torch.is_tensor(v) else v for v in feat.values()]
                X.append(feat_values)
                y.append(label)
        
        if not X:
            return {"f1": 0, "precision": 0, "recall": 0}
        
        X = np.array(X, dtype=np.float32)
        y = np.array(y, dtype=np.float32)
        
        # 使用训练集的标准化参数
        X_scaled = scaler.transform(X)
        
        # 预测
        model.eval()
        with torch.no_grad():
            X_tensor = torch.tensor(X_scaled, dtype=torch.float32).to(device)
            outputs = model(X_tensor).squeeze()
            probs = torch.sigmoid(outputs).cpu().numpy()
            y_pred = (probs > 0.5).astype(int)
        
        # 计算指标
        f1 = f1_score(y, y_pred, zero_division=0)
        precision = precision_score(y, y_pred, zero_division=0)
        recall = recall_score(y, y_pred, zero_division=0)
        
        return {"f1": f1, "precision": precision, "recall": recall}
    
    def train(self, bimodels: List[BIModel], batch_size=32, epochs=50):
        """训练本地分类器"""
        # 收集所有表的引用
        self.tables = []
        for bm in bimodels:
            self.tables.extend(bm.tables)
        
        # 生成训练样本
        all_samples = []
        
        self.logger.info("\n=== 分析训练数据 ===")
        for bm_idx, bm in enumerate(bimodels):
            self.logger.info(f"BI模型 #{bm_idx+1}: {len(bm.tables)}表, {len(bm.joins)}连接")
            
            # 添加正样本（真实连接）
            for join in bm.joins:
                all_samples.append((join[0], join[1], 1))
            
            # 添加负样本（使用优化后的方法）
            self.logger.info(f"生成负样本...")
            negative_samples = self.generate_negative_samples(bm)
            all_samples.extend(negative_samples)
            self.logger.info(f"生成 {len(negative_samples)} 个负样本")
        
        self.logger.info(f"总样本数: {len(all_samples)} (含负采样)")
        
        # 应用标签传递性
        #all_samples = self.apply_transitivity(all_samples)
        
        # 划分训练集和测试集
        train_samples, test_samples = train_test_split(
            all_samples, 
            test_size=0.2,
            random_state=42,
            stratify=[label for _, _, label in all_samples]
        )
        self.logger.info(f"数据集划分: 训练集 {len(train_samples)} 样本, 测试集 {len(test_samples)} 样本")
        
        # 分离N:1和1:1连接（只对训练集）
        n1_samples, o1_samples = self.separate_join_types(train_samples)

        def _train_model(samples, is_1to1, model_type):
            """训练单个分类器的内部函数"""
            if len(samples) == 0:
                self.logger.warning(f"警告：没有找到{'1:1' if is_1to1 else 'N:1'}连接样本，跳过训练")
                return None, None, None
            
            # 提取特征并转换为张量
            X = []
            y = []
            valid_samples = 0
            skipped_samples = 0
            
            self.logger.info(f"提取{'1:1' if is_1to1 else 'N:1'}模型的特征...")
            for (c1, c2, label) in samples:
                feat = self.extract_features(c1, c2, is_1to1)
                if feat:  # 确保特征提取成功
                    # 确保所有特征值都是Python标量
                    feat_values = [v.item() if torch.is_tensor(v) else v for v in feat.values()]
                    X.append(feat_values)
                    y.append(label)
                    valid_samples += 1
                else:
                    skipped_samples += 1
            
            if valid_samples == 0:
                self.logger.error(f"错误：没有有效的{'1:1' if is_1to1 else 'N:1'}样本")
                return None, None, None
                
            self.logger.info(f"使用 {valid_samples} 个有效样本进行训练，跳过 {skipped_samples} 个无效样本")
            
            X = np.array(X, dtype=np.float32)
            y = np.array(y, dtype=np.float32)

            # 标准化特征
            feature_scaler = StandardScaler()
            X = feature_scaler.fit_transform(X)

            # 划分数据集
            X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)
            train_dataset = JoinDataset(X_train, y_train)
            val_dataset = JoinDataset(X_val, y_val)
            train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
            val_loader = DataLoader(val_dataset, batch_size=batch_size)

            # 初始化模型
            model = JoinNet(input_dim=X.shape[1]).to(device)
            criterion = nn.BCEWithLogitsLoss()  # 替换 nn.BCELoss()
            optimizer = optim.Adam(model.parameters(), lr=0.0001, weight_decay=1e-5)  # 添加权重衰减

            # 创建梯度缩放器
            grad_scaler = torch.amp.GradScaler()
            
            # 初始化记录列表
            train_losses = []
            val_losses = []
            train_f1_scores = []
            val_f1_scores = []
            train_precisions = []
            val_precisions = []
            train_recalls = []
            val_recalls = []
            
            # 训练循环
            model.train()
            self.logger.info(f"开始训练{'1:1' if is_1to1 else 'N:1'}模型...")
            best_f1 = 0
            best_model_state = None
            
            for epoch in range(epochs):
                # 训练阶段
                model.train()
                total_train_loss = 0
                train_preds = []
                train_true = []
                start_time = time.time()
                
                for feats, labels in train_loader:
                    feats, labels = feats.to(device), labels.to(device)
                
                    # 检查输入数据
                    if torch.isnan(feats).any() or torch.isinf(feats).any():
                        self.logger.error("输入数据包含NaN或Inf值，跳过该批次")
                        continue
                        
                    optimizer.zero_grad()
                    
                    # 混合精度训练
                    with torch.amp.autocast(device_type=device.type):
                        outputs = model(feats).squeeze()
                        loss = criterion(outputs, labels)
                    
                    # 反向传播
                    grad_scaler.scale(loss).backward()
                    
                    # 梯度裁剪
                    grad_scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                    
                    # 更新参数
                    grad_scaler.step(optimizer)
                    grad_scaler.update()
                    
                    total_train_loss += loss.item()
                    
                    # 记录训练集预测结果
                    with torch.no_grad():
                        probs = torch.sigmoid(outputs).cpu().numpy()
                        preds = (probs > 0.5).astype(int)
                        train_preds.extend(preds)
                        train_true.extend(labels.cpu().numpy())
                
                # 计算训练集指标
                avg_train_loss = total_train_loss / len(train_loader)
                train_losses.append(avg_train_loss)
                
                if len(train_true) > 0:
                    train_f1 = f1_score(train_true, train_preds, zero_division=0)
                    train_precision = precision_score(train_true, train_preds, zero_division=0)
                    train_recall = recall_score(train_true, train_preds, zero_division=0)
                    
                    train_f1_scores.append(train_f1)
                    train_precisions.append(train_precision)
                    train_recalls.append(train_recall)
                else:
                    train_f1 = 0
                    train_precision = 0
                    train_recall = 0
                
                # 验证阶段
                model.eval()
                total_val_loss = 0
                val_preds = []
                val_true = []
                
                with torch.no_grad():
                    for feats, labels in val_loader:
                        feats = feats.to(device)
                        labels = labels.to(device)
                        outputs = model(feats).squeeze()
                        
                        # 计算验证损失
                        loss = criterion(outputs, labels)
                        total_val_loss += loss.item()
                        
                        # 应用 sigmoid 获取概率
                        probs = torch.sigmoid(outputs).cpu().numpy()
                        preds = (probs > 0.5).astype(int)
                        
                        val_preds.extend(preds)
                        val_true.extend(labels.cpu().numpy())
                
                # 计算验证集指标
                avg_val_loss = total_val_loss / len(val_loader)
                val_losses.append(avg_val_loss)
                
                if len(val_true) > 0:
                    val_f1 = f1_score(val_true, val_preds, zero_division=0)
                    val_precision = precision_score(val_true, val_preds, zero_division=0)
                    val_recall = recall_score(val_true, val_preds, zero_division=0)
                    
                    val_f1_scores.append(val_f1)
                    val_precisions.append(val_precision)
                    val_recalls.append(val_recall)
                    
                    # 更新最佳模型
                    if val_f1 > best_f1:
                        best_f1 = val_f1
                        best_model_state = model.state_dict().copy()
                else:
                    val_f1 = 0
                    val_precision = 0
                    val_recall = 0
                
                # 定期打印进度
                if (epoch + 1) % 5 == 0 or epoch == 0 or epoch == epochs-1:
                    self.logger.info(f"Epoch {epoch+1}/{epochs}:")
                    self.logger.info(f"  训练集 - Loss: {avg_train_loss:.6f}, F1: {train_f1:.4f}, Precision: {train_precision:.4f}, Recall: {train_recall:.4f}")
                    self.logger.info(f"  验证集 - Loss: {avg_val_loss:.6f}, F1: {val_f1:.4f}, Precision: {val_precision:.4f}, Recall: {val_recall:.4f}")
                
                # 打印时间和资源使用情况
                epoch_time = time.time() - start_time
                if torch.cuda.is_available():
                    mem_usage = torch.cuda.memory_allocated() / (1024**2)
                    self.logger.info(f"  用时: {epoch_time:.2f}s, GPU内存: {mem_usage:.2f}MB")
            
            # 训练结束后绘制学习曲线
            self._plot_learning_curves(
                train_losses, val_losses, 
                train_f1_scores, val_f1_scores,
                train_precisions, val_precisions,
                train_recalls, val_recalls,
                model_type
            )
            
            # 保存最佳模型
            if best_model_state is not None:
                model.load_state_dict(best_model_state)
                self.logger.info(f"最佳模型验证集 F1={best_f1:.4f}")
            
            return model, feature_scaler, list(feat.keys()) if feat else []
        
        # 训练N:1模型
        if n1_samples:
            self.logger.info("\n=== 训练N:1模型 ===")
            self.n1_model, self.n1_scaler, self.n1_feat_names = _train_model(n1_samples, False, "N:1")
        else:
            self.logger.warning("=== 跳过N:1模型训练（无样本） ===")
            self.n1_model = None
        
        # 训练1:1模型
        if o1_samples:
            self.logger.info("\n=== 训练1:1模型 ===")
            self.o1_model, self.o1_scaler, self.o1_feat_names = _train_model(o1_samples, True, "1:1")
        else:
            self.logger.warning("=== 跳过1:1模型训练（无样本） ===")
            self.o1_model = None
        
        # 在测试集上评估模型
        self.logger.info("\n=== 测试集评估 ===")
        
        # 评估N:1模型
        if self.n1_model:
            n1_test_samples = [s for s in test_samples if self._is_n1_sample(s[0], s[1])]
            if n1_test_samples:
                n1_test_metrics = self._evaluate_samples(n1_test_samples, self.n1_model, self.n1_scaler, False)
                self.logger.info(f"N:1模型测试集性能: F1={n1_test_metrics['f1']:.4f}, Precision={n1_test_metrics['precision']:.4f}, Recall={n1_test_metrics['recall']:.4f}")
            else:
                self.logger.warning("没有N:1测试样本")
        
        # 评估1:1模型
        if self.o1_model:
            o1_test_samples = [s for s in test_samples if not self._is_n1_sample(s[0], s[1])]
            if o1_test_samples:
                o1_test_metrics = self._evaluate_samples(o1_test_samples, self.o1_model, self.o1_scaler, True)
                self.logger.info(f"1:1模型测试集性能: F1={o1_test_metrics['f1']:.4f}, Precision={o1_test_metrics['precision']:.4f}, Recall={o1_test_metrics['recall']:.4f}")
            else:
                self.logger.warning("没有1:1测试样本")

    
    def predict_prob(self, c1: Tuple[Table, str], c2: Tuple[Table, str]) -> float:
        """预测两个列之间的连接概率"""
        if self.n1_model is None and self.o1_model is None:
            self.logger.warning("警告：没有可用的训练模型，返回默认概率0.0")
            return 0.0
        
        try:
            t1, col1_name = c1
            t2, col2_name = c2
            size1 = t1.get_row_count()
            size2 = t2.get_row_count()
            if size1 == 0 or size2 == 0:
                return 0.0
            # 事实表识别：行数大于等于阈值
            is_fact1 = size1 >= self.fact_threshold
            is_fact2 = size2 >= self.fact_threshold
            
            # 判断连接类型
            if (is_fact1 and not is_fact2) or (not is_fact1 and is_fact2):
                # N:1连接
                if self.n1_model is not None:
                    # 确保特征提取顺序：事实表到维度表
                    if is_fact1 and not is_fact2:
                        feat = self.extract_features(c1, c2, False)
                    else:
                        feat = self.extract_features(c2, c1, False)
                    
                    if not feat:
                        return 0.0
                        
                    feat_array = np.array([list(feat.values())], dtype=np.float32)
                    feat_scaled = self.n1_scaler.transform(feat_array)
                    feat_tensor = torch.tensor(feat_scaled, dtype=torch.float32).to(device)
                    with torch.no_grad():
                        logits = self.n1_model(feat_tensor)
                        return torch.sigmoid(logits).item()
                else:
                    return 0.0
            else:
                # 1:1连接
                if self.o1_model is not None:
                    feat = self.extract_features(c1, c2, True)
                    if not feat:
                        return 0.0
                        
                    feat_array = np.array([list(feat.values())], dtype=np.float32)
                    feat_scaled = self.o1_scaler.transform(feat_array)
                    feat_tensor = torch.tensor(feat_scaled, dtype=torch.float32).to(device)
                    with torch.no_grad():
                        logits = self.o1_model(feat_tensor)
                        return torch.sigmoid(logits).item()
                else:
                    return 0.0
        except Exception as e:
            self.logger.error(f"预测概率时出错: {e}")
            traceback.print_exc()
            return 0.0

    def save_model(self, model_dir: str):
        """保存模型到指定目录"""
        try:
            if not os.path.exists(model_dir):
                os.makedirs(model_dir)
                self.logger.info(f"创建模型目录: {model_dir}")
            
            # 保存模型配置
            config = {
                "fact_threshold": self.fact_threshold,
                "n1_feat_names": self.n1_feat_names if hasattr(self, 'n1_feat_names') else [],
                "o1_feat_names": self.o1_feat_names if hasattr(self, 'o1_feat_names') else []
            }
            
            with open(os.path.join(model_dir, "config.json"), 'w') as f:
                json.dump(config, f, indent=2)
            
            # 保存N:1模型
            if self.n1_model:
                torch.save({
                    'model_state_dict': self.n1_model.state_dict(),
                    'scaler_mean': self.n1_scaler.mean_,
                    'scaler_scale': self.n1_scaler.scale_
                }, os.path.join(model_dir, "n1_model.pt"))
            
            # 保存1:1模型
            if self.o1_model:
                torch.save({
                    'model_state_dict': self.o1_model.state_dict(),
                    'scaler_mean': self.o1_scaler.mean_,
                    'scaler_scale': self.o1_scaler.scale_
                }, os.path.join(model_dir, "o1_model.pt"))
            
            self.logger.info(f"模型已保存至: {model_dir}")
        except Exception as e:
            self.logger.error(f"保存模型失败: {e}")
            traceback.print_exc()

    def load_model(self, model_dir: str):
        """从指定目录加载模型"""
        try:
            if not os.path.exists(model_dir):
                raise FileNotFoundError(f"模型目录不存在: {model_dir}")
            
            # 加载配置
            config_path = os.path.join(model_dir, "config.json")
            if not os.path.exists(config_path):
                raise FileNotFoundError(f"配置文件不存在: {config_path}")
            
            with open(config_path, 'r') as f:
                config = json.load(f)
            
            self.fact_threshold = config.get("fact_threshold", 100)
            self.n1_feat_names = config.get("n1_feat_names", [])
            self.o1_feat_names = config.get("o1_feat_names", [])
            
            # 加载N:1模型
            n1_path = os.path.join(model_dir, "n1_model.pt")
            if os.path.exists(n1_path):
                checkpoint = torch.load(n1_path, map_location=device, weights_only=False)
                input_dim = len(self.n1_feat_names)
                self.n1_model = JoinNet(input_dim).to(device)
                self.n1_model.load_state_dict(checkpoint['model_state_dict'])
                self.n1_model.eval()
                
                self.n1_scaler = StandardScaler()
                self.n1_scaler.mean_ = checkpoint['scaler_mean']
                self.n1_scaler.scale_ = checkpoint['scaler_scale']
                self.logger.info("N:1模型加载成功")
            else:
                self.logger.warning("未找到N:1模型文件")
                self.n1_model = None
            
            # 加载1:1模型
            o1_path = os.path.join(model_dir, "o1_model.pt")
            if os.path.exists(o1_path):
                checkpoint = torch.load(o1_path, map_location=device, weights_only=False)
                input_dim = len(self.o1_feat_names)
                self.o1_model = JoinNet(input_dim).to(device)
                self.o1_model.load_state_dict(checkpoint['model_state_dict'])
                self.o1_model.eval()
                
                self.o1_scaler = StandardScaler()
                self.o1_scaler.mean_ = checkpoint['scaler_mean']
                self.o1_scaler.scale_ = checkpoint['scaler_scale']
                self.logger.info("1:1模型加载成功")
            else:
                self.logger.warning("未找到1:1模型文件")
                self.o1_model = None
            
            self.logger.info(f"模型已从 {model_dir} 加载")
        except Exception as e:
            self.logger.error(f"加载模型失败: {e}")
            traceback.print_exc()

# ------------------------------
# 4. 全局优化（k-MCA算法）
# ------------------------------

class GlobalOptimizer:
    """全局优化器，负责构建连接图并预测最优BI模型"""
    def __init__(self, local_clf: LocalJoinClassifier, top_k_edges=6): # 增加top_k_edges参数
        self.local_clf = local_clf
        self.p = -np.log(0.5)  # 概率0.5对应的权重
        self.top_k_edges = top_k_edges  # 为每对表保留的Top-K连接数
        self.logger = logging.getLogger('GlobalOptimizer')
        self.logger.info(f"初始化全局优化器，虚拟边权重 p={self.p:.4f}, 每对表Top-K连接数: {self.top_k_edges}")

    def build_graph(self, tables: List[Table]) -> nx.DiGraph:
        """构建全局连接图，为每对表寻找Top-K个连接列对"""
        G = nx.MultiDiGraph()
        self.logger.info("\n=== 构建连接图 ===")
        self.logger.info(f"处理 {len(tables)} 张表，每对表保留Top-{self.top_k_edges}个连接")
        
        # 添加节点（每张表一个节点）
        for i, table in enumerate(tables):
            G.add_node(i, table=table)
            self.logger.info(f"添加节点 {i}: {table.table_name} (行数: {table.get_row_count()})")
        
        # 添加边（连接关系）保证表12间只进行一次判断
        total_edges_added = 0
        for i, t1 in enumerate(tables):
            for j, t2 in enumerate(tables):
                if i >= j:
                    continue
                
                # 使用最小堆来维护Top-K个概率最大的列对 (堆中存储元组: (-prob, col1, col2))
                # 使用负概率是因为heapq默认是最小堆，我们需要的是最大概率
                top_k_heap = [] 
                
                # 查找所有可能的列对并计算概率
                for col1 in t1.col_names:
                    for col2 in t2.col_names:
                        prob = self.local_clf.predict_prob((t1, col1), (t2, col2))
                        if prob > 0:  # 设置一个最低概率阈值，避免太多无意义计算
                            # 如果堆未满，直接加入
                            if len(top_k_heap) < self.top_k_edges:
                                heapq.heappush(top_k_heap, (prob, col1, col2))
                            else:
                                # 如果堆已满，且当前概率大于堆顶（最小）概率，则替换
                                if prob > top_k_heap[0][0]:
                                    heapq.heapreplace(top_k_heap, (prob, col1, col2))
                
                # 将堆中的Top-K列对添加到图中
                for prob, col1, col2 in top_k_heap:
                    weight = -np.log(prob) if prob > 0 else 100  # 避免log(0)
                    # 边的属性中存储概率、列名等信息
                    G.add_edge(i, j, weight=weight, prob=prob, col1=col1, col2=col2, 
                              description=f"{t1.table_name}.{col1} ↔ {t2.table_name}.{col2}")
                    total_edges_added += 1
                    self.logger.info(f"  边 {i}→{j}: {t1.table_name}.{col1} ↔ {t2.table_name}.{col2}, 概率={prob:.4f}, 权重={weight:.4f}")
        
        self.logger.info(f"共添加 {total_edges_added} 条边 (基于Top-{self.top_k_edges}策略)")
        return G

    def solve_1mca(self, G: nx.DiGraph) -> nx.DiGraph:
        """解决1-MCA问题（最小成本树形图）"""
        try:
            # 使用Edmonds算法
            edmonds = nx.algorithms.tree.branchings.Edmonds(G)
            arborescence = edmonds.find_optimum(attr='weight', kind='arborescence', preserve_attrs=True)
            self.logger.info("1-MCA求解完成")
            return arborescence
        except nx.NetworkXException as e:
            self.logger.error(f"求解1-MCA失败: {e}")
            return nx.DiGraph()

    def solve_kmca(self, G: nx.DiGraph) -> nx.DiGraph:
        """解决k-MCA问题"""
        try:
            # 创建带虚拟根节点的新图
            G_prime = G.copy()
            root = 'virtual_root'
            G_prime.add_node(root)
            
            # 添加从虚拟根节点到所有节点的边，权重为p
            for node in G.nodes:
                G_prime.add_edge(root, node, weight=self.p)
            
            # 在新图上求解1-MCA
            arborescence = self.solve_1mca(G_prime)
            
            # 移除虚拟根节点及其相连的边
            arborescence.remove_node(root)
            
            self.logger.info("k-MCA求解完成")
            return arborescence
        except Exception as e:
            self.logger.error(f"求解k-MCA失败: {e}")
            return nx.DiGraph()

    def solve_kmca_cc(self, G: nx.DiGraph) -> nx.DiGraph:
        """解决带约束的k-MCA问题（FK-once约束）"""
        try:
            # 先求解k-MCA
            kmca = self.solve_kmca(G)
            
            # 应用FK-once约束：对于同一个源列，只保留权重最小的边
            for node in kmca.nodes:
                out_edges = list(kmca.out_edges(node))
                if len(out_edges) > 1:
                    # 按权重排序
                    out_edges.sort(key=lambda e: kmca.edges[e]['weight'])
                    
                    # 找出所有源列相同的边
                    col_edges = defaultdict(list)
                    for e in out_edges:
                        src_col = kmca.edges[e]['cols'][0]
                        col_edges[src_col].append(e)
                    
                    # 对于每个源列，只保留权重最小的边，移除其他边
                    for src_col, edges in col_edges.items():
                        if len(edges) > 1:
                            # 按权重排序
                            edges.sort(key=lambda e: kmca.edges[e]['weight'])
                            # 保留权重最小的边
                            for e in edges[1:]:
                                kmca.remove_edge(*e)
                                self.logger.info(f"移除冲突边: {e[0]}→{e[1]} (源列: {src_col})")
            
            self.logger.info("k-MCA-CC求解完成")
            return kmca
        except Exception as e:
            self.logger.error(f"求解k-MCA-CC失败: {e}")
            return nx.DiGraph()

    def predict_bi_model(self, tables: List[Table]) -> List[Tuple[Table, Table, str, str, float]]:
        """预测BI模型"""
        # 1. 构建全局图
        G = self.build_graph(tables)
        
        # 2. 解决k-MCA-CC问题（precision模式）
        # kmca_cc = self.solve_kmca_cc(G)
        
        # 3. 添加高概率边（recall模式）
        # recall_edges_added = 0
        # for u, v, data in G.edges(data=True):
        #     if (u, v) not in kmca_cc.edges and data['prob'] > 0.7:  # 阈值设为0.7
        #         kmca_cc.add_edge(u, v, **data)
        #         recall_edges_added += 1
        #         self.logger.info(f"添加额外边（recall模式）: {u}→{v} (概率: {data['prob']:.4f})")
        # self.logger.info(f"在recall模式中添加了 {recall_edges_added} 条边")
        
        # 4. 转换为连接关系
        joins = []
        for u, v, data in G.edges(data=True):
            t1 = G.nodes[u]['table']
            t2 = G.nodes[v]['table']
            col1 = data['col1']
            col2 = data['col2']
            prob = data['prob']
            joins.append((t1, t2, col1, col2, prob))
        
        return joins

# ------------------------------
# 5. 主流程（支持训练和推理模式）
# ------------------------------

def ensure_directory_exists(path: str):
    """确保目录存在，如果不存在则创建"""
    path = Path(path)
    if not path.exists():
        path.mkdir(parents=True, exist_ok=True)
        logger.info(f"创建目录: {path}")
    return path

def train_mode(links_file: str, table_dir: str, model_dir: str):
    """训练模式：训练模型并保存"""
    logger.info("=== 训练模式 ===")
    start_time = time.time()
    
    # 确保输出目录存在
    model_dir = ensure_directory_exists(model_dir)
    
    # 加载数据
    logger.info(f"加载表数据: {table_dir}")
    tables_dict = load_tables_from_dir(table_dir)

    logger.info(f"加载连接关系: {links_file}")
    bimodel = load_links_csv(links_file, tables_dict)
    
    if not bimodel.tables or not bimodel.joins:
        logger.error("数据加载失败，退出")
        return
    
    
    logger.info(f"加载完成: {len(bimodel.tables)}张表, {len(bimodel.joins)}个连接关系")
    
    # 训练本地连接分类器
    logger.info("\n=== 训练本地连接分类器 ===")
    train_start = time.time()
    local_clf = LocalJoinClassifier(fact_threshold=100)
    local_clf.train([bimodel], epochs=30)
    logger.info(f"训练完成，耗时: {time.time()-train_start:.2f}秒")
    
    # 保存模型
    logger.info("\n=== 保存模型 ===")
    local_clf.save_model(model_dir)
    
    total_time = time.time() - start_time
    logger.info(f"=== 训练完成，总耗时: {total_time:.2f}秒 ===")

def predict_mode(table_dir: str, model_dir: str, output_file: str, top_k: int = 6):
    """推理模式：加载模型并预测连接关系"""
    logger.info("=== 推理模式 ===")
    start_time = time.time()
    
    # 确保输出目录存在
    output_dir = Path(output_file).parent
    if output_dir:
        ensure_directory_exists(output_dir)
    
    # 加载模型
    logger.info(f"加载模型: {model_dir}")
    local_clf = LocalJoinClassifier()
    try:
        local_clf.load_model(model_dir)
    except Exception as e:
        logger.error(f"加载模型失败: {e}")
        return
    
    # 加载表数据
    logger.info(f"加载表数据: {table_dir}")
    tables_dict = load_tables_from_dir(table_dir)
    tables = list(tables_dict.values())
    
    if not tables:
        logger.error("未加载到任何表数据，退出")
        return
    
    logger.info(f"加载完成: {len(tables)}张表")

    # 全局优化预测
    logger.info("\n=== 全局优化预测 ===")
    predict_start = time.time()
    optimizer = GlobalOptimizer(local_clf, top_k_edges=top_k)
    predicted_joins = optimizer.predict_bi_model(tables)
    logger.info(f"预测完成，耗时: {time.time()-predict_start:.2f}秒")
    k = 10  # 取前k个结果
# 按概率从高到低排序
    predicted_joins_sorted = sorted(predicted_joins, key=lambda x: x[-1], reverse=True)

    logger.info(f"预测完成，耗时: {time.time()-predict_start:.2f}秒")

    logger.info("\n=== 预测结果 ===")
    results = []
    for t1, t2, col1, col2, prob in predicted_joins_sorted[:top_k]:
        logger.info(f"{t1.table_name}.{col1} ↔ {t2.table_name}.{col2} (概率: {prob:.4f})")
        results.append({
            "table1": t1.table_name,
            "table2": t2.table_name,
            "column1": col1,
            "column2": col2,
            "prob": prob
        })
    
    # 保存到文件
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)
    logger.info(f"结果已保存至: {output_file}")
    
    total_time = time.time() - start_time
    logger.info(f"=== 推理完成，总耗时: {total_time:.2f}秒 ===")

def Pair_join(pair: str, model_dir: str, tables_dir: str) -> float:
    """
    输入连接对，模型路径和表路径，输出连接对的概率

    参数:
        pair (str): 连接对，格式为 'table_name.col1 ↔ table_name.col2'
        model_dir (str): 训练好的模型保存路径
        tables_dir (str): 存储表数据的目录路径

    返回:
        float: 连接对的预测概率（0.0 到 1.0 之间）
    """
    try:
        # 解析连接对字符串
        match = re.match(r'([^.]+)\.([^-]+)\s*↔\s*([^.]+)\.([^\s]+)', pair)
        if not match:
            raise ValueError(f"连接对格式不正确: {pair}")
        
        t1_name, col1_name, t2_name, col2_name = match.groups()

        # 加载表数据
        tables = load_tables_from_dir(tables_dir)
        t1 = tables.get(t1_name)
        t2 = tables.get(t2_name)

        if not t1 or not t2:
            raise ValueError(f"表 {t1_name} 或 {t2_name} 未找到")
        
        # 加载训练好的模型
        local_clf = LocalJoinClassifier()
        local_clf.load_model(model_dir)

        # 预测连接概率
        prob=local_clf.predict_prob((t1,col1_name),(t2,col2_name))

        return prob

    except Exception as e:
        logger.error(f"计算连接对概率时出错: {e}")
        return 0.0
    
def Max_pair_join(t1_name: str, t2_name: str, model_dir: str, tables_dir: str) -> tuple:
    """
    输入两个表名，模型路径和表路径，输出这两表之间最大连接概率的两列及其概率

    参数:
        t1_name (str): 第一个表的名称
        t2_name (str): 第二个表的名称
        model_dir (str): 训练好的模型保存路径
        tables_dir (str): 存储表数据的目录路径

    返回:
        tuple: 包含最大连接概率的两列及其概率，例如 (('table1.col1', 'table2.col2'), 0.85)
    """
    try:
        # 加载表数据
        tables = load_tables_from_dir(tables_dir)
        t1 = tables.get(t1_name)
        t2 = tables.get(t2_name)

        if not t1 or not t2:
            raise ValueError(f"表 {t1_name} 或 {t2_name} 未找到")
        
        # 获取所有列名
        t1_columns = t1.columns
        t2_columns = t2.columns

        # 加载训练好的模型
        local_clf = LocalJoinClassifier()
        local_clf.load_model(model_dir)

        # 初始化最大概率及列组合
        max_prob = 0.0
        best_pair = None

        # 遍历所有列组合，计算连接概率
        for col1 in t1_columns:
            for col2 in t2_columns:
                prob = local_clf.predict_prob((t1, col1), (t2, col2))
                if prob > max_prob:
                    max_prob = prob
                    best_pair = (f"{t1_name}.{col1}", f"{t2_name}.{col2}")

        return best_pair, max_prob

    except Exception as e:
        logger.error(f"计算最大连接概率时出错: {e}")
        return None, 0.0
    
def JoinClassify(
    mode: str,
    tables: str,
    model_dir: str,
    links: Optional[str] = None,
    output: Optional[str] = None,
    top_k: int = 10
) -> None:
    """
    Auto-BI系统主入口函数
    
    Args:
        mode: 运行模式 'train' 或 'predict'
        tables: 表数据目录路径
        model_dir: 模型保存/加载目录路径
        links: 训练模式所需的连接关系CSV文件路径
        output: 推理模式所需的输出JSON文件路径
        top_k: 推理模式中为每对表保留的Top-K连接数量
    """
    if mode == 'train':
        if not links:
            logger.error("训练模式需要指定 links 参数")
            return
        train_mode(links, tables, model_dir)
    elif mode == 'predict':
        if not output:
            logger.error("推理模式需要指定 output 参数")
            return
        predict_mode(tables, model_dir, output, top_k)
    else:
        logger.error(f"不支持的模式: {mode}，请选择 'train' 或 'predict'")

def main():
    """保留原有的命令行接口"""
    parser = argparse.ArgumentParser(description='Auto-BI系统')
    parser.add_argument('--mode', type=str, required=True, choices=['train', 'predict'], 
                        help='运行模式: train(训练) 或 predict(推理)')
    
    # 训练模式参数
    parser.add_argument('--links', type=str, help='训练模式: 连接关系CSV文件路径')
    parser.add_argument('--tables', type=str, required=True, help='表数据目录')
    parser.add_argument('--model_dir', type=str, required=True, help='模型保存/加载目录')
    parser.add_argument('--top_k', type=int, default=10,help='推理模式: 为每对表保留的Top-K连接数量 (默认: 3)')
    
    # 推理模式参数
    parser.add_argument('--output', type=str, help='推理模式: 输出JSON文件路径')
    
    args = parser.parse_args()
    
    JoinClassify(
        mode=args.mode,
        tables=args.tables,
        model_dir=args.model_dir,
        links=args.links,
        output=args.output,
        top_k=args.top_k
    )

if __name__ == "__main__":
    main()
    # python JoinClassifier.py --mode train --links cleaned_data.csv --tables practice  --model_dir models/dbtv1/
    # python JoinClassifier.py --mode predict --tables t/ --model_dir models/dbtv1/ --output predictions.json