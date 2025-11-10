import os
import pandas as pd
import numpy as np
import networkx as nx
os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
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

# ------------------------------
# 日志配置：同时输出到控制台和文件
# ------------------------------
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

set_seed(42)
torch.use_deterministic_algorithms(True)
# log_dir = "./logs"
# os.makedirs(log_dir, exist_ok=True)
# log_file = os.path.join(log_dir, f"AutoBI_{time.strftime('%Y%m%d_%H%M%S')}.log")

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),                   # 控制台输出
        # logging.FileHandler(log_file, encoding='utf-8')  # 写入文件
    ]
)
logger = logging.getLogger('JoinClassify')
# logger.info(f"日志将写入文件: {log_file}")

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
# 2. 数据加载与预处理
# ------------------------------

def load_tables_from_dir(table_dir: str) -> Dict[str, Table]:
    tables = {}
    table_path = Path(table_dir)
    if not table_path.exists():
        logger.error(f"表目录不存在: {table_path}")
        return tables
    csv_files = list(table_path.glob("*.csv"))
    logger.info(f"找到 {len(csv_files)} 个CSV文件")
    for filepath in csv_files:
        try:
            df = pd.read_csv(filepath)
            table_name = filepath.stem
            columns = {col: df[col].tolist() for col in df.columns}
            tables[table_name] = Table(table_name, columns)
            logger.info(f"加载表: {table_name} ({len(df)}行, {len(df.columns)}列)")
        except Exception as e:
            logger.error(f"加载表{filepath.name}失败: {e}")
    logger.info(f"加载完成: {len(tables)} 张表")
    return tables

def load_links_csv(links_file: str, tables: Dict[str, Table]) -> BIModel:
    try:
        links_path = Path(links_file)
        if not links_path.exists():
            logger.error(f"连接文件不存在: {links_path}")
            return BIModel([], [])
        encodings = ['utf-8', 'latin-1', 'iso-8859-1', 'cp1252', 'gbk']
        links_df = None
        for encoding in encodings:
            try:
                df_sample = pd.read_csv(links_path, nrows=2, header=None, encoding=encoding)
                if len(df_sample.columns) >= 4:
                    links_df = pd.read_csv(links_path, header=None, encoding=encoding)
                    logger.info(f"使用编码: {encoding} 成功读取连接文件（无列头）")
                    break
            except (UnicodeDecodeError, pd.errors.ParserError):
                continue
        if links_df is None:
            try:
                links_df = pd.read_csv(links_path, header=None)
                logger.info("使用默认编码和无header方式读取连接文件")
            except Exception as e:
                logger.error(f"无法读取连接文件: {e}")
                return BIModel([], [])
        column_positions = {'table1': 0, 'table2': 1, 'column1': 2, 'column2': 3, 'prob': 4 if len(links_df.columns) > 4 else None}
        tables_in_model = set()
        joins = []
        skipped = 0
        for idx, row in links_df.iterrows():
            try:
                t1n = str(row[column_positions['table1']]).strip()
                t2n = str(row[column_positions['table2']]).strip()
                if t1n.endswith('.csv'): t1n = t1n[:-4]
                if t2n.endswith('.csv'): t2n = t2n[:-4]
                c1n = str(row[column_positions['column1']]).strip()
                c2n = str(row[column_positions['column2']]).strip()
                if not t1n or not t2n:
                    skipped += 1
                    continue
                tables_in_model.update([t1n, t2n])
                t1 = tables.get(t1n); t2 = tables.get(t2n)
                if t1 and t2 and c1n in t1.columns and c2n in t2.columns:
                    joins.append(((t1, c1n), (t2, c2n)))
                else:
                    skipped += 1
            except Exception as e:
                logger.error(f"处理行 {idx} 时出错: {e}")
                skipped += 1
        table_objs = [tables[name] for name in tables_in_model if name in tables]
        logger.info(f"加载连接关系: 找到 {len(table_objs)} 张表, {len(joins)} 个连接, 跳过 {skipped} 行")
        return BIModel(table_objs, joins)
    except Exception as e:
        logger.error(f"加载连接关系失败: {e}")
        traceback.print_exc()
        return BIModel([], [])

def load_links_neg_csv(links_file: str, tables: Dict[str, Table]):
    links_path = Path(links_file)
    if not links_path.exists():
        logger.error(f"连接文件不存在: {links_path}")
        return BIModel([], [])
    enc = ['utf-8', 'latin-1', 'iso-8859-1', 'cp1252', 'gbk']
    links_df = None
    for e in enc:
        try:
            df_sample = pd.read_csv(links_path, nrows=2, header=None, encoding=e)
            if len(df_sample.columns) >= 4:
                links_df = pd.read_csv(links_path, header=None, encoding=e)
                logger.info(f"使用编码: {e} 成功读取连接文件（无列头）")
                break
        except (UnicodeDecodeError, pd.errors.ParserError):
            continue
    if links_df is None:
        try:
            links_df = pd.read_csv(links_path, header=None)
            logger.info("使用默认编码和无header方式读取连接文件")
        except Exception as e:
            logger.error(f"无法读取连接文件: {e}")
            return BIModel([], [])
    column_positions = {'table1': 0, 'table2': 1, 'column1': 2, 'column2': 3, 'prob': 4 if len(links_df.columns) > 4 else None}
    joins = []
    for idx, row in links_df.iterrows():
        t1n = str(row[column_positions['table1']]).strip()
        t2n = str(row[column_positions['table2']]).strip()
        if t1n.endswith('.csv'): t1n = t1n[:-4]
        if t2n.endswith('.csv'): t2n = t2n[:-4]
        c1n = str(row[column_positions['column1']]).strip()
        c2n = str(row[column_positions['column2']]).strip()
        t1 = tables.get(t1n); t2 = tables.get(t2n)
        if t1 and t2 and c1n in t1.columns and c2n in t2.columns:
            joins.append(((t1, c1n), (t2, c2n), 0))
    logger.info(f"加载负样例连接关系: 找到 {len(joins)} 个连接")
    return joins

# ------------------------------
# 3. 单一 1:1 模型（含样本加权训练）
# ------------------------------

class JoinDataset(Dataset):
    def __init__(self, features, labels):
        self.features = torch.tensor(features, dtype=torch.float32)
        self.labels = torch.tensor(labels, dtype=torch.float32)
    def __len__(self): return len(self.features)
    def __getitem__(self, idx): return self.features[idx], self.labels[idx]

class JoinNet(nn.Module):
    def __init__(self, input_dim):
        super(JoinNet, self).__init__()
        self.layers = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.LayerNorm(128),       # ← 替换 BatchNorm1d(128)
            nn.Dropout(0.3),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.LayerNorm(64),        # ← 替换 BatchNorm1d(64)（如果你有的话）
            nn.Linear(64, 1)
        )
    def forward(self, x):
        return self.layers(x)

class LocalJoinClassifier:
    """
    仅训练/使用 1:1 模型。
    1:1 特征中包含：
      - min_contain（最小包含度）
      - max_contain（最大包含度）  <-- 新增
      - 其它原有 1:1 特征
    训练时继续使用“办法A”样本加权（按 min_contain 加权）。
    """
    def __init__(self, fact_threshold=1000):
        self.bert = SentenceTransformer('/data/new_dbt_dataset/JoinClassify/all-mpnet-base-v2-local').to(device)
        torch.use_deterministic_algorithms(True)
        # 单一模型
        self.model = None
        self.scaler = None
        self.feat_names = None

        self.col_name_cache={}
        self.fact_threshold = fact_threshold
        self.feature_cache={}
        self.tables = []
        self.logger = logging.getLogger('LocalJoinClassifier')
        self.logger.info(f"初始化本地分类器（单一1:1），事实表阈值={fact_threshold}行")

    def tokenize_name(self, name: str) -> List[str]:
        name = re.sub(r'[^a-zA-Z0-9]', ' ', name)
        name = re.sub(r'([a-z])([A-Z])', r'\1 \2', name)
        return [t.lower() for t in name.split() if t]

    def _str_similarity(self, s1: str, s2: str) -> Dict:
        try:
            return {
                "jaccard": fuzz.token_sort_ratio(s1, s2) / 100,
                "edit_dist": fuzz.ratio(s1, s2) / 100
            }
        except Exception as e:
            self.logger.error(f"字符串相似度计算错误: {e}")
            return {"jaccard": 0, "edit_dist": 0}
        
    def _sample_frequent_values(self, column: List, max_seq_length=512):
        from collections import Counter
        value_freq = Counter(column)
        distinct_values = list(value_freq.keys())
        distinct_values.sort(key=lambda x: value_freq[x], reverse=True)
        sampled_values = []
        current = 0
        tokenizer = self.bert.tokenizer
        for value in distinct_values:
            value_str = str(value)
            tokens = tokenizer.tokenize(value_str)
            t = len(tokens)
            if sampled_values: t += 1
            if current + t <= max_seq_length:
                sampled_values.append(value_str)
                current += t
            else:
                break
        return sorted(sampled_values, key=lambda x: tokenizer.tokenize(x))
    
    def _create_text_sequence(self, table: Table, column_name: str, sampled_values: List[str]) -> str:
        column_data = table.columns[column_name]
        n = len(set(column_data))
        value_lengths = [len(str(x)) for x in column_data]
        max_len = max(value_lengths) if value_lengths else 0
        min_len = min(value_lengths) if value_lengths else 0
        avg_len = sum(value_lengths) / len(value_lengths) if value_lengths else 0
        return (f"{table.table_name}.{column_name} contains {n} values "
                f"({max_len},{min_len},{avg_len:.1f}): "
                f"{', '.join(sampled_values)}")
    
    def get_columns_cosine_similarity(self, table1: Table, col1_name: str, table2: Table, col2_name: str) -> float:
        key1 = (table1.table_name, col1_name)
        key2 = (table2.table_name, col2_name)
        try:
            if key1 in self.feature_cache and key2 in self.feature_cache:
                emb1, emb2 = self.feature_cache[key1], self.feature_cache[key2]
            else:
                if key1 not in self.feature_cache:
                    v1 = self._sample_frequent_values(table1.columns[col1_name])
                    t1 = self._create_text_sequence(table1, col1_name, v1)
                    self.feature_cache[key1] = self.bert.encode(t1, convert_to_tensor=True, show_progress_bar=False)
                if key2 not in self.feature_cache:
                    v2 = self._sample_frequent_values(table2.columns[col2_name])
                    t2 = self._create_text_sequence(table2, col2_name, v2)
                    self.feature_cache[key2] = self.bert.encode(t2, convert_to_tensor=True, show_progress_bar=False)
                emb1, emb2 = self.feature_cache[key1], self.feature_cache[key2]
            sim = F.cosine_similarity(emb1.unsqueeze(0), emb2.unsqueeze(0), dim=1).item()
            if np.isnan(sim) or np.isinf(sim): return 0.0
            return float(sim)
        except Exception as e:
            self.logger.error(f"列内容相似度计算错误: {e}")
            return 0.0

    def _embedding_sim(self, s1: str, s2: str) -> float:
        try:
            if s1 not in self.col_name_cache:
                self.col_name_cache[s1] = self.bert.encode(s1, convert_to_tensor=True, show_progress_bar=False)
            if s2 not in self.col_name_cache:
                self.col_name_cache[s2] = self.bert.encode(s2, convert_to_tensor=True, show_progress_bar=False)
            e1, e2 = self.col_name_cache[s1], self.col_name_cache[s2]
            sim = F.cosine_similarity(e1.unsqueeze(0), e2.unsqueeze(0)).item()
            if np.isnan(sim) or np.isinf(sim): return 0.0
            return float(sim)
        except Exception as e:
            self.logger.error(f"嵌入相似度计算错误: {e}")
            return 0.0

    def _as_numeric_array(self, values: List) -> np.ndarray:
        try:
            arr = pd.to_numeric(pd.Series(values), errors='coerce').dropna().to_numpy(dtype=float)
            return arr
        except Exception:
            return np.array([], dtype=float)

    def _data_features(self, col1: List, col2: List) -> Dict:
        features = {
            "contain1": 0, "contain2": 0, "emd": 0, "distinct1": 0, 
            "distinct2": 0, "range_overlap": 0, "value_length1": 0, 
            "value_length2": 0, "value_type1": 0, "value_type2": 0,
            "col_emb_sim": 0.0
        }
        try:
            set1, set2 = set(col1), set(col2)
            overlap = len(set1 & set2)
            features["contain1"] = overlap / len(set1) if set1 else 0
            features["contain2"] = overlap / len(set2) if set2 else 0
            features["distinct1"] = len(set1) / len(col1) if col1 else 0
            features["distinct2"] = len(set2) / len(col2) if col2 else 0

            num1 = self._as_numeric_array(col1)
            num2 = self._as_numeric_array(col2)
            if len(num1) and len(num2):
                try:
                    features["emd"] = float(wasserstein_distance(num1, num2))
                except Exception:
                    features["emd"] = 0.0
                min1, max1 = float(num1.min()), float(num1.max())
                min2, max2 = float(num2.min()), float(num2.max())
                left = max(min1, min2); right = min(max1, max2)
                if left <= right:
                    overlap_length = right - left
                    total_range = max(max1, max2) - min(min1, min2)
                    features["range_overlap"] = (overlap_length / total_range) if total_range > 0 else 0

            def avg_value_length(values):
                if not values: return 0
                total_length = sum(len(str(x)) for x in values)
                return total_length / len(values)
            features["value_length1"] = avg_value_length(col1)
            features["value_length2"] = avg_value_length(col2)

            def get_value_type(values):
                if not values: return "unknown"
                s = values[0]
                if isinstance(s, (int, float, np.integer, np.floating)):
                    return "numeric"
                elif isinstance(s, str):
                    return "string"
                elif isinstance(s, bool):
                    return "boolean"
                else:
                    return "other"
            features["value_type1"] = 1 if get_value_type(col1) == "numeric" else 0
            features["value_type2"] = 1 if get_value_type(col2) == "numeric" else 0

            for k, v in list(features.items()):
                if isinstance(v, (np.floating, np.integer)):
                    v = float(v)
                if isinstance(v, (int, float)) and (np.isnan(v) or np.isinf(v)):
                    features[k] = 0.0
        except Exception as e:
            self.logger.error(f"数据特征提取出错: {e}")
            traceback.print_exc()

        return features

    def extract_features(self, c1: Tuple[Table, str], c2: Tuple[Table, str], is_1to1: bool) -> Dict:
        """保持签名不变，但训练/预测一律走 1:1 分支；并在 1:1 中加入 max_contain"""
        try:
            t1, col1_name = c1
            t2, col2_name = c2
            if col1_name not in t1.columns:
                raise ValueError(f"列 '{col1_name}' 不在表 '{t1.table_name}' 中")
            if col2_name not in t2.columns:
                raise ValueError(f"列 '{col2_name}' 不在表 '{t2.table_name}' 中")
            
            data1 = t1.columns[col1_name]
            data2 = t2.columns[col2_name]
            table_col1=f"{t1.table_name}.{col1_name}"
            table_col2=f"{t2.table_name}.{col2_name}"

            
            pos1 = t1.col_names.index(col1_name) / len(t1.col_names) if col1_name in t1.col_names else 0
            pos2 = t2.col_names.index(col2_name) / len(t2.col_names) if col2_name in t2.col_names else 0

            meta_feat = {}
            str_feat = self._str_similarity(col1_name, col2_name)
            meta_feat.update(str_feat)
            emb_feat = {
                "col_name_emb_sim": self._embedding_sim(table_col1, table_col2),
                "table_emb_sim": self._embedding_sim(t1.table_name, t2.table_name),
                "col_emb_sim": self.get_columns_cosine_similarity(t1,col1_name,t2,col2_name)
            } 
            meta_feat.update(emb_feat)
            meta_feat["token_count1"] = len(self.tokenize_name(col1_name))
            meta_feat["token_count2"] = len(self.tokenize_name(col2_name))
            meta_feat["char_count1"] = len(col1_name)
            meta_feat["char_count2"] = len(col2_name)
            meta_feat["pos1"] = pos1
            meta_feat["pos2"] = pos2
            
            data_feat = self._data_features(data1, data2)

            # ---- 单一 1:1 分支（新增 max_contain，同时保留 min_contain） ----
            header_jaccard = len(set(t1.col_names) & set(t2.col_names)) 
            header_jaccard /= len(set(t1.col_names) | set(t2.col_names)) if (t1.col_names and t2.col_names) else 0
            min_contain = min(data_feat["contain1"], data_feat["contain2"])
            max_contain = max(data_feat["contain1"], data_feat["contain2"])  # 新增
            features = {
                **meta_feat, 
                **data_feat,
                "max_contain": max_contain,   # 新增进特征
                "header_jaccard": header_jaccard
            }
            # self.logger.info(f"{t1.table_name}.{col1_name}-{t2.table_name}.{col2_name}")
            # self.logger.info(f"min_cont:{min_contain},max_cont:{max_contain}")
            # self.logger.info(f"表列名嵌入余弦相似度：{self._embedding_sim(table_col1, table_col2)}")
            # self.logger.info(f"列嵌入余弦相似度：{self.get_columns_cosine_similarity(t1,col1_name,t2,col2_name)}")
            # -------------------------------------------------------------

            for key in list(features.keys()):
                value = features[key]
                if torch.is_tensor(value):
                    features[key] = value.item() if value.numel() == 1 else value.cpu().numpy()
                elif hasattr(value, 'item') and callable(getattr(value, 'item')):
                    features[key] = value.item()
                if isinstance(features[key], (int, float)) and (np.isnan(features[key]) or np.isinf(features[key])):
                    features[key] = 0.0
            return features
        except Exception as e:
            # 使用局部变量名安全输出
            try:
                msg = f"提取特征时出错 - {t1.table_name}.{col1_name} vs {t2.table_name}.{col2_name}: {e}"
            except Exception:
                msg = f"提取特征时出错: {e}"
            self.logger.error(msg)
            traceback.print_exc()
            return {}

    def _build_join_graph(self, samples: List[Tuple[Tuple, Tuple, int]]) -> nx.Graph:
        graph = nx.Graph()
        for (c1, c2, label) in samples:
            if label == 1:
                t1, col1_name = c1
                t2, col2_name = c2
                key1 = (t1.table_name, col1_name)
                key2 = (t2.table_name, col2_name)
                graph.add_edge(key1, key2)
        return graph

    def apply_transitivity(self, samples: List[Tuple[Tuple, Tuple, int]]) -> List:
        try:
            graph = self._build_join_graph(samples)
            new_samples = []
            visited = set()
            for comp in nx.connected_components(graph):
                comp_list = list(comp)
                for i in range(len(comp_list)):
                    for j in range(i+1, len(comp_list)):
                        col1_info = comp_list[i]
                        col2_info = comp_list[j]
                        if (col1_info, col2_info) in visited or (col2_info, col1_info) in visited:
                            continue
                        t1_name, col1_name = col1_info
                        t2_name, col2_name = col2_info
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

    def generate_negative_samples(self, bm: BIModel, negative_ratio=0.75, max_negative=10000):
        join_set = set()
        for (t1, col1), (t2, col2) in bm.joins:
            key = (min(t1.table_name, t2.table_name), max(t1.table_name, t2.table_name), col1, col2)
            join_set.add(key)
        num_negative = min(int(len(bm.joins) * negative_ratio), max_negative)
        negative_samples = []
        attempts = 0
        max_attempts = num_negative * 10
        while len(negative_samples) < num_negative and attempts < max_attempts:
            attempts += 1
            t1, t2 = random.sample(bm.tables, 2)
            if t1 == t2: continue
            col1 = random.choice(t1.col_names)
            col2 = random.choice(t2.col_names)
            if col1 == col2: continue
            key = (min(t1.table_name, t2.table_name), max(t1.table_name, t2.table_name), col1, col2)
            if key not in join_set:
                negative_samples.append(((t1, col1), (t2, col2), 0))
        return negative_samples

    def batch_extract_features(self, pairs: List[Tuple], is_1to1: bool) -> List[Dict]:
        """保持签名，内部按 1:1 特征构造，含 min_contain + max_contain"""
        features_list = []
        col1_names = [f"{t1.table_name}.{col1}" for (t1, col1), _ in pairs]
        col2_names = [f"{t2.table_name}.{col2}" for _, (t2, col2) in pairs]
        with torch.no_grad():
            emb1 = self.bert.encode(col1_names, convert_to_tensor=True, batch_size=128, show_progress_bar=False)
            emb2 = self.bert.encode(col2_names, convert_to_tensor=True, batch_size=128, show_progress_bar=False)
            col_name_sims = torch.cosine_similarity(emb1, emb2).cpu().numpy()
        for i, ((t1, col1), (t2, col2)) in enumerate(pairs):
            try:
                data1 = t1.columns.get(col1, [])
                data2 = t2.columns.get(col2, [])
                col_content_sim = self.get_columns_cosine_similarity(t1, col1, t2, col2)
                meta_feat = {
                    "col_content_sim": col_content_sim,
                    "col_name_emb_sim": float(col_name_sims[i]) if i < len(col_name_sims) else 0.0,
                    "table_emb_sim": self._embedding_sim(t1.table_name, t2.table_name),
                    "token_count1": len(self.tokenize_name(col1)),
                    "token_count2": len(self.tokenize_name(col2)),
                    "char_count1": len(col1),
                    "char_count2": len(col2),
                    "pos1": t1.col_names.index(col1)/len(t1.col_names) if col1 in t1.col_names else 0,
                    "pos2": t2.col_names.index(col2)/len(t2.col_names) if col2 in t2.col_names else 0
                }
                data_feat = self._data_features(data1, data2)
                header_jaccard = len(set(t1.col_names) & set(t2.col_names))
                header_jaccard /= len(set(t1.col_names) | set(t2.col_names)) if t1.col_names and t2.col_names else 0
                min_contain = min(data_feat.get("contain1", 0), data_feat.get("contain2", 0))
                max_contain = max(data_feat.get("contain1", 0), data_feat.get("contain2", 0))  # 新增
                features = {
                    **meta_feat,
                    **data_feat,
                    "max_contain": max_contain,    # 新增
                    "header_jaccard": header_jaccard
                }
                for k in list(features.keys()):
                    v = features[k]
                    if torch.is_tensor(v):
                        v = v.item() if v.numel() == 1 else v.cpu().numpy()
                    elif hasattr(v, 'item') and callable(getattr(v, 'item')):
                        v = v.item()
                    if isinstance(v, (int, float)) and (np.isnan(v) or np.isinf(v)):
                        v = 0.0
                    features[k] = v
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
        try:
            import matplotlib.pyplot as plt
            epochs = range(1, len(train_losses) + 1)
            fig, axes = plt.subplots(2, 2, figsize=(15, 10))
            axes[0, 0].plot(epochs, train_losses, label='训练损失')
            axes[0, 0].plot(epochs, val_losses, label='验证损失'); axes[0, 0].legend(); axes[0, 0].grid(True)
            axes[0, 1].plot(epochs, train_f1_scores, label='训练F1')
            axes[0, 1].plot(epochs, val_f1_scores, label='验证F1'); axes[0, 1].legend(); axes[0, 1].grid(True)
            axes[1, 0].plot(epochs, train_precisions, label='训练精确率')
            axes[1, 0].plot(epochs, val_precisions, label='验证精确率'); axes[1, 0].legend(); axes[1, 0].grid(True)
            axes[1, 1].plot(epochs, train_recalls, label='训练召回率')
            axes[1, 1].plot(epochs, val_recalls, label='验证召回率'); axes[1, 1].legend(); axes[1, 1].grid(True)
            plt.tight_layout()
            os.makedirs("learning_curves", exist_ok=True)
            curve_path = os.path.join("learning_curves", f"{model_type}_learning_curves.png")
            plt.savefig(curve_path); plt.close()
            self.logger.info(f"学习曲线已保存至: {curve_path}")
        except ImportError:
            self.logger.warning("无法导入matplotlib，跳过学习曲线绘制")
        except Exception as e:
            self.logger.error(f"绘制学习曲线时出错: {e}")

    def _evaluate_samples(self, samples, model, scaler):
        X, y = [], []
        feat_names = self.feat_names if hasattr(self, 'feat_names') else None

        for (c1, c2, label) in samples:
            feat = self.extract_features(c1, c2, True)  # 统一按 1:1 特征
            if feat:
                X.append([feat.get(k, 0.0) for k in feat_names] if feat_names else list(feat.values()))
                y.append(label)

        if not X:
            return {"f1": 0, "precision": 0, "recall": 0}

        X = np.array(X, dtype=np.float32)
        y = np.array(y, dtype=np.float32)

        X_scaled = scaler.transform(X)

        model.eval()
        with torch.no_grad():
            X_tensor = torch.tensor(X_scaled, dtype=torch.float32, device=device)
            # 关键修改：保持 batch 维度
            logits = model(X_tensor).squeeze(-1)   # 等价写法：.view(-1)
            probs  = torch.sigmoid(logits).cpu().numpy()
            y_pred = (probs > 0.5).astype(int)

        f1 = f1_score(y, y_pred, zero_division=0)
        precision = precision_score(y, y_pred, zero_division=0)
        recall = recall_score(y, y_pred, zero_division=0)

        return {"f1": f1, "precision": precision, "recall": recall}

    
    def train(self, bimodels: List[BIModel], neg, batch_size=32, epochs=50):
        """单一 1:1 模型训练（办法A：按 min_contain 做样本加权）"""
        # 收集表
        self.tables = []
        for bm in bimodels:
            self.tables.extend(bm.tables)
        
        # 汇总样本
        all_samples = []
        self.logger.info("\n=== 分析训练数据 ===")
        for bm_idx, bm in enumerate(bimodels):
            self.logger.info(f"BI模型 #{bm_idx+1}: {len(bm.tables)}表, {len(bm.joins)}连接")
            for join in bm.joins:
                all_samples.append((join[0], join[1], 1))
            self.logger.info(f"生成负样本...")
            negative_samples = self.generate_negative_samples(bm)
            all_samples.extend(neg)
            all_samples.extend(negative_samples)
            self.logger.info(f"生成 {len(negative_samples)} 个负样本")
        
        self.logger.info(f"总样本数: {len(all_samples)} (含负采样)")
        # all_samples = self.apply_transitivity(all_samples)  # 可选
        
        train_samples, test_samples = train_test_split(
            all_samples, 
            test_size=0.2,
            random_state=42,
            stratify=[label for _, _, label in all_samples]
        )
        self.logger.info(f"数据集划分: 训练集 {len(train_samples)} 样本, 测试集 {len(test_samples)} 样本")

        # ========= 超参：重合度权重（办法A） =========
        ALPHA = 3.0   # 放大系数（可调 2~5）
        GAMMA = 8.0   # 陡峭度（可调 6~12）
        # ============================================

        # 提取特征（1:1）
        X = []; y = []; self.feat_names = None
        self.logger.info(f"提取 1:1 模型的特征...")
        min_contain_cache = []  # 为权重保留原始 min_contain 的近似（标准化后会近似映射）
        for (c1, c2, label) in train_samples:
            feat = self.extract_features(c1, c2, True)
            if feat:
                if self.feat_names is None:
                    self.feat_names = list(feat.keys())
                X.append([feat.get(k, 0.0) for k in self.feat_names])
                y.append(label)
        if len(X) == 0:
            self.logger.error("错误：没有有效的 1:1 样本")
            return
        
        X = np.array(X, dtype=np.float32)
        y = np.array(y, dtype=np.float32)

        # 先划分，再fit scaler，避免数据泄漏
        X_train, X_val, y_train, y_val = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        self.scaler = StandardScaler()
        X_train = self.scaler.fit_transform(X_train)
        X_val   = self.scaler.transform(X_val)

        train_dataset = JoinDataset(X_train, y_train)
        val_dataset = JoinDataset(X_val, y_val)
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size)

        self.model = JoinNet(input_dim=X.shape[1]).to(device)

        # 类不平衡处理
        pos = float((y_train == 1).sum())
        neg_cnt = float((y_train == 0).sum())
        pos_weight = torch.tensor([neg_cnt / max(pos, 1.0)], device=device)

        optimizer = optim.Adam(self.model.parameters(), lr=0.0001, weight_decay=1e-5)
        use_amp = (device.type == "cuda")
        grad_scaler = torch.amp.GradScaler(enabled=use_amp)

        # 重合度特征列索引（标准化后仍在相同列）
        overlap_key = "max_contain"  # 办法A按最小包含度做样本加权
        try:
            overlap_idx = self.feat_names.index(overlap_key)
        except ValueError:
            overlap_idx = None
            self.logger.warning(f"未找到重合度特征列: {overlap_key}，将不启用样本加权")

        # 记录
        train_losses=[]; val_losses=[]
        train_f1_scores=[]; val_f1_scores=[]
        train_precisions=[]; val_precisions=[]
        train_recalls=[]; val_recalls=[]

        best_f1 = 0.0
        best_state = None
        EARLY_STOP = 8
        patience = 0
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=3, verbose=False)

        for epoch in range(epochs):
            self.model.train()
            total_train_loss = 0.0
            train_preds = []; train_true = []
            start_time = time.time()
            
            for feats, labels in train_loader:
                feats, labels = feats.to(device), labels.to(device)
                if torch.isnan(feats).any() or torch.isinf(feats).any():
                    continue
                optimizer.zero_grad()
                with torch.amp.autocast(device_type="cuda", enabled=use_amp):
                    outputs = self.model(feats).squeeze(-1)
                    # 基础 BCE（逐样本），含 pos_weight
                    base_loss = F.binary_cross_entropy_with_logits(
                        outputs, labels, reduction='none', pos_weight=pos_weight
                    )
                    if overlap_idx is not None:
                        # 从标准化后的特征中取出 min_contain 列，做[-3,3]限幅并线性映射回[0,1]
                        x = feats[:, overlap_idx]
                        x = torch.clamp(x, -3.0, 3.0)
                        r = (x + 3.0) / 6.0
                        w = 1.0 + ALPHA * (r.clamp(0,1) ** GAMMA)
                    else:
                        w = torch.ones_like(base_loss)
                    loss = (base_loss * w).mean()

                grad_scaler.scale(loss).backward()
                grad_scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                grad_scaler.step(optimizer)
                grad_scaler.update()
                total_train_loss += loss.item()

                with torch.no_grad():
                    probs = torch.sigmoid(outputs).detach().cpu().numpy()
                    preds = (probs > 0.5).astype(int)
                    train_preds.extend(preds)
                    train_true.extend(labels.detach().cpu().numpy())

            avg_train_loss = total_train_loss / max(1, len(train_loader))
            train_losses.append(avg_train_loss)
            if len(train_true) > 0:
                train_f1 = f1_score(train_true, train_preds, zero_division=0)
                train_precision = precision_score(train_true, train_preds, zero_division=0)
                train_recall = recall_score(train_true, train_preds, zero_division=0)
            else:
                train_f1 = train_precision = train_recall = 0.0
            train_f1_scores.append(train_f1); train_precisions.append(train_precision); train_recalls.append(train_recall)

            # 验证
            self.model.eval()
            total_val_loss = 0.0
            val_preds = []; val_true = []
            with torch.no_grad():
                for feats, labels in val_loader:
                    feats = feats.to(device); labels = labels.to(device)
                    outputs = self.model(feats).squeeze()
                    val_loss = F.binary_cross_entropy_with_logits(
                        outputs, labels, reduction='mean', pos_weight=pos_weight
                    )
                    total_val_loss += val_loss.item()
                    probs = torch.sigmoid(outputs).detach().cpu().numpy()
                    preds = (probs > 0.5).astype(int)
                    val_preds.extend(preds)
                    val_true.extend(labels.detach().cpu().numpy())

            avg_val_loss = total_val_loss / max(1, len(val_loader))
            val_losses.append(avg_val_loss)
            if len(val_true) > 0:
                val_f1 = f1_score(val_true, val_preds, zero_division=0)
                val_precision = precision_score(val_true, val_preds, zero_division=0)
                val_recall = recall_score(val_true, val_preds, zero_division=0)
            else:
                val_f1 = val_precision = val_recall = 0.0
            val_f1_scores.append(val_f1); val_precisions.append(val_precision); val_recalls.append(val_recall)

            scheduler.step(val_f1)
            if val_f1 > best_f1:
                best_f1 = val_f1
                best_state = {k: v.detach().cpu().clone() for k, v in self.model.state_dict().items()}
                patience = 0
            else:
                patience += 1
                if patience >= EARLY_STOP:
                    self.logger.info("早停触发")
                    break

            if (epoch + 1) % 5 == 0 or epoch == 0 or epoch == epochs-1:
                self.logger.info(
                    f"[1:1] Epoch {epoch+1}/{epochs} | "
                    f"Train Loss {avg_train_loss:.4f} F1 {train_f1:.4f} | "
                    f"Val Loss {avg_val_loss:.4f} F1 {val_f1:.4f}"
                )
            if torch.cuda.is_available():
                mem_usage = torch.cuda.memory_allocated() / (1024**2)
                self.logger.debug(f"GPU内存: {mem_usage:.2f}MB")

        self._plot_learning_curves(
            train_losses, val_losses, 
            train_f1_scores, val_f1_scores,
            train_precisions, val_precisions,
            train_recalls, val_recalls,
            "one_to_one"
        )
        if best_state is not None:
            self.model.load_state_dict(best_state)
            self.logger.info(f"1:1 最佳验证集 F1={best_f1:.4f}")

        # 测试集评估
        self.logger.info("\n=== 测试集评估（单一1:1） ===")
        if len(test_samples):
            metrics = self._evaluate_samples(test_samples, self.model, self.scaler)
            self.logger.info(f"测试集性能: F1={metrics['f1']:.4f}, Precision={metrics['precision']:.4f}, Recall={metrics['recall']:.4f}")
        else:
            self.logger.warning("没有测试样本")

    def predict_prob(self, c1: Tuple[Table, str], c2: Tuple[Table, str]) -> float:
        """预测两个列之间的连接概率（单一 1:1 模型）"""
        if self.model is None:
            self.logger.warning("警告：没有可用的训练模型，返回默认概率0.0")
            return 0.0
        try:
            self.model.eval()
            t1, col1_name = c1
            t2, col2_name = c2
            size1 = t1.get_row_count()
            size2 = t2.get_row_count()
            if size1 == 0 or size2 == 0:
                return 0.0

            feat = self.extract_features(c1, c2, True)
            if not feat:
                return 0.0
            vec = np.array([[feat.get(k, 0.0) for k in self.feat_names]], dtype=np.float32)
            vec = self.scaler.transform(vec)
            with torch.no_grad():
                logits = self.model(torch.tensor(vec, dtype=torch.float32, device=device))
                return float(torch.sigmoid(logits).item())
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
            config = {
                "fact_threshold": self.fact_threshold,
                "feat_names": self.feat_names if hasattr(self, 'feat_names') else []
            }
            with open(os.path.join(model_dir, "config.json"), 'w') as f:
                json.dump(config, f, indent=2)
            if self.model:
                torch.save({
                    'model_state_dict': self.model.state_dict(),
                    'scaler': self.scaler,
                }, os.path.join(model_dir, "model.pt"))
            self.logger.info(f"模型已保存至: {model_dir}")
        except Exception as e:
            self.logger.error(f"保存模型失败: {e}")
            traceback.print_exc()

    def load_model(self, model_dir: str):
        """从指定目录加载模型"""
        try:
            if not os.path.exists(model_dir):
                raise FileNotFoundError(f"模型目录不存在: {model_dir}")
            config_path = os.path.join(model_dir, "config.json")
            if not os.path.exists(config_path):
                raise FileNotFoundError(f"配置文件不存在: {config_path}")
            with open(config_path, 'r') as f:
                config = json.load(f)
            self.fact_threshold = config.get("fact_threshold", 1000)
            self.feat_names = config.get("feat_names", [])
            model_path = os.path.join(model_dir, "model.pt")
            if os.path.exists(model_path):
                checkpoint = torch.load(model_path, map_location=device,weights_only=False)
                input_dim = len(self.feat_names)
                self.model = JoinNet(input_dim).to(device)
                self.model.load_state_dict(checkpoint['model_state_dict'])
                self.model.eval()
                self.scaler = checkpoint['scaler']
                self.logger.info("1:1 模型加载成功")
            else:
                self.logger.warning("未找到模型文件")
                self.model = None
            self.logger.info(f"模型已从 {model_dir} 加载")
        except Exception as e:
            self.logger.error(f"加载模型失败: {e}")
            traceback.print_exc()

# ------------------------------
# 4. 全局优化（k-MCA算法）
# ------------------------------

class GlobalOptimizer:
    """全局优化器，负责构建连接图并预测最优BI模型"""
    def __init__(self, local_clf: LocalJoinClassifier, top_k_edges=6):
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
        
        # 添加节点
        for i, table in enumerate(tables):
            G.add_node(i, table=table)
            self.logger.info(f"添加节点 {i}: {table.table_name} (行数: {table.get_row_count()})")
        
        total_edges_added = 0
        # 仅遍历一次 i<j
        for i, t1 in enumerate(tables):
            for j, t2 in enumerate(tables):
                if i >= j:
                    continue
                top_k_heap = []
                for col1 in t1.col_names:
                    for col2 in t2.col_names:
                        prob = self.local_clf.predict_prob((t1, col1), (t2, col2))
                        if prob > 0:
                            if len(top_k_heap) < self.top_k_edges:
                                heapq.heappush(top_k_heap, (prob, col1, col2))
                            else:
                                if prob > top_k_heap[0][0]:
                                    heapq.heapreplace(top_k_heap, (prob, col1, col2))
                for prob, col1, col2 in top_k_heap:
                    weight = -np.log(prob) if prob > 0 else 100
                    G.add_edge(i, j, weight=weight, prob=prob, col1=col1, col2=col2, 
                              description=f"{t1.table_name}.{col1} ↔ {t2.table_name}.{col2}")
                    total_edges_added += 1
                    self.logger.info(f"  边 {i}→{j}: {t1.table_name}.{col1} ↔ {t2.table_name}.{col2}, 概率={prob:.4f}, 权重={weight:.4f}")
        self.logger.info(f"共添加 {total_edges_added} 条边 (基于Top-{self.top_k_edges}策略)")
        return G

    def solve_1mca(self, G: nx.DiGraph) -> nx.DiGraph:
        """解决1-MCA问题（最小成本树形图）"""
        try:
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
            G_prime = G.copy()
            root = 'virtual_root'
            G_prime.add_node(root)
            for node in G.nodes:
                G_prime.add_edge(root, node, weight=self.p)
            arborescence = self.solve_1mca(G_prime)
            arborescence.remove_node(root)
            self.logger.info("k-MCA求解完成")
            return arborescence
        except Exception as e:
            self.logger.error(f"求解k-MCA失败: {e}")
            return nx.DiGraph()

    def solve_kmca_cc(self, G: nx.DiGraph) -> nx.DiGraph:
        """带约束的k-MCA（保留示例实现）"""
        try:
            kmca = self.solve_kmca(G)
            for node in kmca.nodes:
                out_edges = list(kmca.out_edges(node))
                if len(out_edges) > 1:
                    out_edges.sort(key=lambda e: kmca.edges[e]['weight'])
                    col_edges = defaultdict(list)
                    for e in out_edges:
                        src_col = kmca.edges[e]['cols'][0]
                        col_edges[src_col].append(e)
                    for src_col, edges in col_edges.items():
                        if len(edges) > 1:
                            edges.sort(key=lambda e: kmca.edges[e]['weight'])
                            for e in edges[1:]:
                                kmca.remove_edge(*e)
                                self.logger.info(f"移除冲突边: {e[0]}→{e[1]} (源列: {src_col})")
            self.logger.info("k-MCA-CC求解完成")
            return kmca
        except Exception as e:
            self.logger.error(f"求解k-MCA-CC失败: {e}")
            return nx.DiGraph()

    def predict_bi_model(self, tables: List[Table]) -> List[Tuple[Table, Table, str, str, float]]:
        """预测BI模型：返回 (t1, t2, col1, col2, prob) 列表"""
        G = self.build_graph(tables)
        joins = []
        for u, v, data in G.edges(data=True):
            t1 = G.nodes[u]['table']
            t2 = G.nodes[v]['table']
            joins.append((t1, t2, data['col1'], data['col2'], data['prob']))
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

def train_mode(links_file: str, neglinks_file: str, table_dir: str, model_dir: str):
    """训练模式：训练模型并保存（单一 1:1）"""
    logger.info("=== 训练模式 ===")
    start_time = time.time()
    model_dir = ensure_directory_exists(model_dir)

    logger.info(f"加载表数据: {table_dir}")
    tables_dict = load_tables_from_dir(table_dir)

    logger.info(f"加载连接关系: {links_file}")
    bimodel = load_links_csv(links_file, tables_dict)
    nega = load_links_neg_csv(neglinks_file, tables_dict)

    if not bimodel.tables or not bimodel.joins:
        logger.error("数据加载失败，退出")
        return

    logger.info(f"加载完成: {len(bimodel.tables)}张表, {len(bimodel.joins)}个连接关系")

    logger.info("\n=== 训练本地连接分类器（1:1） ===")
    train_start = time.time()
    local_clf = LocalJoinClassifier(fact_threshold=1000)
    local_clf.train([bimodel], nega, epochs=30)
    logger.info(f"训练完成，耗时: {time.time()-train_start:.2f}秒")

    logger.info("\n=== 保存模型 ===")
    local_clf.save_model(model_dir)

    total_time = time.time() - start_time
    logger.info(f"=== 训练完成，总耗时: {total_time:.2f}秒 ===")

def predict_mode(table_dir: str, model_dir: str, output_file: str, top_k: int = 6):
    """推理模式：加载模型并预测连接关系"""
    logger.info("=== 推理模式 ===")
    start_time = time.time()

    output_dir = Path(output_file).parent
    if output_dir:
        ensure_directory_exists(output_dir)

    logger.info(f"加载模型: {model_dir}")
    local_clf = LocalJoinClassifier()
    try:
        local_clf.load_model(model_dir)
    except Exception as e:
        logger.error(f"加载模型失败: {e}")
        return

    logger.info(f"加载表数据: {table_dir}")
    tables_dict = load_tables_from_dir(table_dir)
    tables = list(tables_dict.values())
    if not tables:
        logger.error("未加载到任何表数据，退出")
        return
    logger.info(f"加载完成: {len(tables)}张表")

    logger.info("\n=== 全局优化预测 ===")
    predict_start = time.time()
    optimizer = GlobalOptimizer(local_clf, top_k_edges=top_k)
    predicted_joins = optimizer.predict_bi_model(tables)
    logger.info(f"预测完成，耗时: {time.time()-predict_start:.2f}秒")

    logger.info("\n=== 预测结果 ===")
    results = []
    for t1, t2, col1, col2, prob in predicted_joins:
        # if prob < 0.1:
        #     continue
        logger.info(f"{t1.table_name}.{col1} ↔ {t2.table_name}.{col2} (概率: {prob:.4f})")
        results.append({
            "table1": t1.table_name,
            "table2": t2.table_name,
            "column1": col1,
            "column2": col2,
            "prob": prob
        })

    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)
    logger.info(f"结果已保存至: {output_file}")

    total_time = time.time() - start_time
    logger.info(f"=== 推理完成，总耗时: {total_time:.2f}秒 ===")

def JoinClassify(
    mode: str,
    tables: str,
    model_dir: str,
    neglinks: Optional[str] = None,
    links: Optional[str] = None,
    output: Optional[str] = None,
    top_k: int = 6
) -> None:
    """
    Auto-BI系统主入口函数
    """
    if mode == 'train':
        if not links:
            logger.error("训练模式需要指定 links 参数")
            return
        train_mode(links, neglinks, tables, model_dir)
    elif mode == 'predict':
        if not output:
            logger.error("推理模式需要指定 output 参数")
            return
        predict_mode(tables, model_dir, output, top_k)
    else:
        logger.error(f"不支持的模式: {mode}，请选择 'train' 或 'predict'")

def main():
    """命令行接口"""
    parser = argparse.ArgumentParser(description='Auto-BI系统（单一1:1模型）')
    parser.add_argument('--mode', type=str, required=True, choices=['train', 'predict'], help='运行模式: train(训练) 或 predict(推理)')
    parser.add_argument('--links', type=str, help='训练模式: 连接关系CSV文件路径')
    parser.add_argument('--neglinks', type=str, help='训练模式: 负样例连接关系CSV文件路径')
    parser.add_argument('--tables', type=str, required=True, help='表数据目录')
    parser.add_argument('--model_dir', type=str, required=True, help='模型保存/加载目录')
    parser.add_argument('--top_k', type=int, default=6, help='推理模式: 为每对表保留的Top-K连接数量 (默认: 6)')
    parser.add_argument('--output', type=str, help='推理模式: 输出JSON文件路径')
    args = parser.parse_args()

    JoinClassify(
        mode=args.mode,
        tables=args.tables,
        model_dir=args.model_dir,
        links=args.links,
        neglinks=args.neglinks,
        output=args.output,
        top_k=args.top_k
    )

if __name__ == "__main__":
    main()
