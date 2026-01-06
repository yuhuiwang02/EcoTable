import json
import os
import shutil
import random
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Set, Tuple
import re

class EdgeBasedNoiseInjector:
    def __init__(self, path_a: str, path_b: str):
        self.path_a = Path(path_a)
        self.path_b = Path(path_b)
        self.json_file = None
        self.data = None
        self.noise_records = []
        
    def copy_data(self):
        if self.path_b.exists():
            shutil.rmtree(self.path_b)
        shutil.copytree(self.path_a, self.path_b)
        print(f"✓ 数据已从 {self.path_a} 复制到 {self.path_b}")
    
    def load_json(self, json_filename: str):
        self.json_file = self.path_b / json_filename
        with open(self.json_file, 'r', encoding='utf-8') as f:
            self.data = json.load(f)
        print(f"✓ 已加载JSON文件: {json_filename}")
    
    def check_and_clean_tables(self):
        removed_count = 0
        for category in ['normal', 'no_join_cols', 'no_overlap']:
            if category not in self.data:
                continue
            valid_tables = []
            for table in self.data[category]:
                table_name = table['table_name']
                folder_name = f"{table_name}_{category}"
                folder_path = self.path_b / folder_name
                if folder_path.exists():
                    valid_tables.append(table)
                else:
                    print(f"  ⚠ 文件夹不存在，删除表格: {folder_name}")
                    removed_count += 1
            self.data[category] = valid_tables
        
        with open(self.json_file, 'w', encoding='utf-8') as f:
            json.dump(self.data, f, ensure_ascii=False, indent=2)
        print(f"✓ 文件夹检查完成，删除了 {removed_count} 个无效表格")
    
    def get_all_join_columns_for_table(self, table_name: str) -> Set[str]:
        all_join_cols = set()
        for table in self.data.get('normal', []):
            if table['table_name'] == table_name:
                edge_join_columns = table.get('edge_join_columns', {})
                for edge_key, edge_cols in edge_join_columns.items():
                    all_join_cols.update(edge_cols)
                break
        return all_join_cols
    
    def get_edge_unique_id(self, edge_info: Dict) -> str:
        """生成边的唯一标识符：table_name + edge_key"""
        return f"{edge_info['table_name']}::{edge_info['edge_key']}"
    
    def check_has_non_join_columns(self, edge_info: Dict) -> bool:
        """检查边的至少一个节点是否存在额外的非连接列，用于列级别加噪"""
        table_name = edge_info['table_name']
        all_join_cols_in_table = self.get_all_join_columns_for_table(table_name)
        
        for node_info in [edge_info['node1_info'], edge_info['node2_info']]:
            all_columns = node_info.get('columns', [])
            non_join_cols = [col for col in all_columns if col not in all_join_cols_in_table]
            if len(non_join_cols) > 0:
                return True
        return False
    
    def extract_edge_candidates(self) -> Tuple[List[Dict], List[Dict]]:
        cell_level_candidates = []
        column_level_candidates = []
        
        for table in self.data.get('normal', []):
            table_name = table['table_name']
            node_mapping = table.get('node_mapping', {})
            edge_join_columns = table.get('edge_join_columns', {})
            original_shape = table.get('original_shape', [0, 0])
            
            for edge_key, edge_join_cols in edge_join_columns.items():
                nodes = edge_key.split('-')
                if len(nodes) != 2:
                    continue
                node1_id, node2_id = nodes
                if node1_id not in node_mapping or node2_id not in node_mapping:
                    continue
                
                node1_info = node_mapping[node1_id]
                node2_info = node_mapping[node2_id]
                edge_num_join_cols = len(edge_join_cols)
                edge_has_lat_lon = any('latitude' in col.lower() or 'longitude' in col.lower() 
                                       for col in edge_join_cols)
                
                edge_candidate = {
                    'table_name': table_name,
                    'edge_key': edge_key,
                    'edge_join_cols': edge_join_cols,
                    'num_join_cols': edge_num_join_cols,
                    'has_lat_lon': edge_has_lat_lon,
                    'node1_id': node1_id,
                    'node1_info': node1_info,
                    'node2_id': node2_id,
                    'node2_info': node2_info,
                    'original_shape': original_shape
                }
                
                if edge_num_join_cols < 2:
                    cell_level_candidates.append(edge_candidate)
                elif edge_num_join_cols >= 2 and edge_has_lat_lon:
                    column_level_candidates.append(edge_candidate)
        
        return cell_level_candidates, column_level_candidates
    
    def select_edge_for_noise(self, candidates: List[Dict], count: int) -> List[Dict]:
        count = min(count, len(candidates))
        return random.sample(candidates, count)
    
    def detect_column_type(self, series: pd.Series) -> str:
        non_null = series.dropna()
        if len(non_null) == 0:
            return 'string'
        if pd.api.types.is_numeric_dtype(series):
            return 'numeric'
        sample = str(non_null.iloc[0])
        if re.match(r'^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}\.\d{3}$', sample):
            return 'datetime'
        return 'string'
    
    # ========== 数值类型加噪方法 ==========
    def apply_scientific_notation(self, value) -> Tuple[str, str]:
        if pd.isna(value):
            return str(value), 'no_change'
        try:
            float_val = float(value)
            sci_notation = f"{float_val:.2e}"
            return sci_notation, 'scientific_notation'
        except:
            return str(value), 'no_change'
    
    def apply_percentage_format(self, value) -> Tuple[str, str]:
        if pd.isna(value):
            return str(value), 'no_change'
        try:
            float_val = float(value)
            percentage = f"{float_val * 100:.2f}%"
            return percentage, 'percentage'
        except:
            return str(value), 'no_change'
    
    # ========== 时间类型加噪方法 ==========
    def apply_datetime_format_change(self, value) -> Tuple[str, str]:
        if pd.isna(value):
            return str(value), 'no_change'
        try:
            dt = datetime.strptime(str(value), '%Y-%m-%dT%H:%M:%S.%f')
            new_format = dt.strftime('%Y/%m/%d %H:%M')
            return new_format, 'datetime_format_change'
        except:
            return str(value), 'no_change'
    
    # ========== 字符串类型加噪方法 ==========
    def apply_keyboard_mistake(self, value: str) -> Tuple[str, str]:
        if pd.isna(value):
            return str(value), 'no_change'
        value = str(value)
        if len(value) < 2:
            return value, 'no_change'
        
        keyboard_map = {
            'a': ['s', 'q', 'w'], 'b': ['v', 'g', 'h', 'n'], 'c': ['x', 'd', 'f', 'v'],
            'd': ['s', 'e', 'r', 'f', 'c', 'x'], 'e': ['w', 'r', 'd', 's'], 
            'f': ['d', 'r', 't', 'g', 'v', 'c'], 'g': ['f', 't', 'y', 'h', 'b', 'v'],
            'h': ['g', 'y', 'u', 'j', 'n', 'b'], 'i': ['u', 'o', 'k', 'j'],
            'j': ['h', 'u', 'i', 'k', 'm', 'n'], 'k': ['j', 'i', 'o', 'l', 'm'],
            'l': ['k', 'o', 'p'], 'm': ['n', 'j', 'k'], 'n': ['b', 'h', 'j', 'm'],
            'o': ['i', 'p', 'l', 'k'], 'p': ['o', 'l'], 'q': ['w', 'a'],
            'r': ['e', 't', 'f', 'd'], 's': ['a', 'w', 'e', 'd', 'x', 'z'],
            't': ['r', 'y', 'g', 'f'], 'u': ['y', 'i', 'j', 'h'], 
            'v': ['c', 'f', 'g', 'b'], 'w': ['q', 'e', 's', 'a'],
            'x': ['z', 's', 'd', 'c'], 'y': ['t', 'u', 'h', 'g'],
            'z': ['a', 's', 'x']
        }
        
        value_list = list(value)
        pos = random.randint(0, len(value_list) - 1)
        char = value_list[pos].lower()
        
        if char in keyboard_map:
            value_list[pos] = random.choice(keyboard_map[char])
            return ''.join(value_list), 'keyboard_mistake'
        return value, 'no_change'
    
    def apply_lowercase(self, value: str) -> Tuple[str, str]:
        if pd.isna(value):
            return str(value), 'no_change'
        return str(value).lower(), 'lowercase'
    
    def apply_extra_space(self, value: str) -> Tuple[str, str]:
        if pd.isna(value):
            return str(value), 'no_change'
        value_str = str(value)
        if len(value_str) < 2:
            return value_str, 'no_change'
        pos = random.randint(1, len(value_str) - 1)
        return value_str[:pos] + ' ' + value_str[pos:], 'extra_space'
    
    def apply_special_char(self, value: str) -> Tuple[str, str]:
        if pd.isna(value):
            return str(value), 'no_change'
        special_chars = ['!', '@', '#', '$', '%', '&', '*', '-', '_']
        char = random.choice(special_chars)
        if random.choice([True, False]):
            return char + str(value), 'special_char_prefix'
        else:
            return str(value) + char, 'special_char_suffix'
    
    def apply_substring(self, value: str) -> Tuple[str, str]:
        if pd.isna(value):
            return str(value), 'no_change'
        value_str = str(value)
        if len(value_str) < 3:
            return value_str, 'no_change'
        min_len = max(2, len(value_str) // 2)
        start = random.randint(0, len(value_str) - min_len)
        end = random.randint(start + min_len, len(value_str))
        return value_str[start:end], 'substring'
    
    def apply_abbreviation(self, value: str) -> Tuple[str, str]:
        if pd.isna(value):
            return str(value), 'no_change'
        value_str = str(value)
        words = value_str.split()
        if len(words) > 1:
            abbr = ''.join([w[0].upper() for w in words if len(w) > 0])
            return abbr, 'abbreviation'
        elif len(value_str) > 3:
            return value_str[:3].upper(), 'abbreviation'
        return value_str, 'no_change'
    
    def apply_cell_level_noise_to_single_node(self, df: pd.DataFrame, edge_info: Dict, 
                                               csv_file_name: str, node_id: str,
                                               noise_methods: List, noise_rows: List[int]) -> pd.DataFrame:
        """对单个节点的DataFrame应用单元格级别加噪，每行随机选择加噪方法"""
        table_name = edge_info['table_name']
        edge_key = edge_info['edge_key']
        edge_join_cols = edge_info['edge_join_cols']
        
        if len(edge_join_cols) == 0:
            return df
        
        col = edge_join_cols[0]
        if col not in df.columns:
            return df
        
        col_type = self.detect_column_type(df[col])
        
        for row_idx in noise_rows:
            if row_idx >= len(df):
                continue
            original_value = df.at[row_idx, col]
            
            # 每行随机选择一个加噪方法
            noise_method_func = random.choice(noise_methods)
            new_value, noise_type = noise_method_func(original_value)
            
            if noise_type != 'no_change':
                df.at[row_idx, col] = new_value
                self.noise_records.append({
                    'table_name': table_name, 'csv_file': csv_file_name,
                    'node_id': node_id, 'edge_key': edge_key,
                    'noise_level': 'cell', 'noise_type': noise_type,
                    'column': col, 'column_type': col_type,
                    'original_index': int(row_idx),
                    'original_value': str(original_value), 'new_value': str(new_value)
                })
        return df
    
    def apply_cell_level_noise(self, df: pd.DataFrame, edge_info: Dict, 
                            csv_file_name: str, selected_node_id: str) -> pd.DataFrame:
        """原有的单节点加噪方法，保留用于兼容"""
        table_name = edge_info['table_name']
        edge_key = edge_info['edge_key']
        edge_join_cols = edge_info['edge_join_cols']
        original_shape = edge_info.get('original_shape', [0, 0])
        
        if len(edge_join_cols) == 0:
            print(f"    ⚠ 该边没有连接列，跳过")
            return df
        
        col = edge_join_cols[0]
        if col not in df.columns:
            print(f"    ⚠ 列 {col} 不存在，跳过")
            return df
        
        col_type = self.detect_column_type(df[col])
        
        if col_type == 'numeric':
            noise_methods = [self.apply_scientific_notation, self.apply_percentage_format]
        elif col_type == 'datetime':
            noise_methods = [self.apply_datetime_format_change]
        elif col_type == 'string':
            noise_methods = [
                self.apply_keyboard_mistake, self.apply_lowercase,
                self.apply_extra_space, self.apply_special_char,
                self.apply_substring, self.apply_abbreviation
            ]
        else:
            print(f"    ⚠ 列 {col} 类型未知，跳过")
            return df
        
        base_table_rows = original_shape[0]
        num_rows_to_noise = max(1, int(base_table_rows * 0.01))
        num_rows = len(df)
        num_rows_to_noise = min(num_rows_to_noise, num_rows)
        noised_rows = list(range(num_rows_to_noise))
        
        print(f"    单元格加噪：列 '{col}' ({col_type})，基表行数 {base_table_rows}，子表行数 {num_rows}，加噪 {num_rows_to_noise} 行（基表的1%）")
        
        for row_idx in noised_rows:
            original_value = df.at[row_idx, col]
            method = random.choice(noise_methods)
            new_value, noise_type = method(original_value)
            
            if noise_type != 'no_change':
                df.at[row_idx, col] = new_value
                self.noise_records.append({
                    'table_name': table_name, 'csv_file': csv_file_name,
                    'node_id': selected_node_id, 'edge_key': edge_key,
                    'noise_level': 'cell', 'noise_type': noise_type,
                    'column': col, 'column_type': col_type,
                    'original_index': int(row_idx),
                    'original_value': str(original_value), 'new_value': str(new_value)
                })
        return df
    
    def apply_column_level_noise(self, df: pd.DataFrame, edge_info: Dict, 
                                  csv_file_name: str, selected_node_id: str) -> pd.DataFrame:
        table_name = edge_info['table_name']
        edge_key = edge_info['edge_key']
        edge_join_cols = edge_info['edge_join_cols']
        
        selected_node = (edge_info['node1_info'] if selected_node_id == edge_info['node1_id'] 
                        else edge_info['node2_info'])
        all_columns = selected_node.get('columns', [])
        all_join_cols_in_table = self.get_all_join_columns_for_table(table_name)
        non_join_cols = [col for col in all_columns 
                        if col not in all_join_cols_in_table and col in df.columns]
        
        if not non_join_cols:
            print(f"  ⚠ 没有可用的非连接列进行合并")
            return df
        
        selected_join_col = random.choice(edge_join_cols)
        selected_non_join_col = random.choice(non_join_cols)
        new_col_name = f"{selected_join_col}-{selected_non_join_col}"
        
        df[new_col_name] = df[selected_join_col].astype(str) + '-' + df[selected_non_join_col].astype(str)
        df = df.drop([selected_join_col, selected_non_join_col], axis=1)
        
        self.noise_records.append({
            'table_name': table_name, 'csv_file': csv_file_name,
            'node_id': selected_node_id, 'edge_key': edge_key,
            'noise_level': 'column', 'noise_type': 'column_merge',
            'join_column': selected_join_col, 'non_join_column': selected_non_join_col,
            'merged_column': new_col_name,
            'description': f'将连接列 {selected_join_col} 与非连接列 {selected_non_join_col} 合并为 {new_col_name}'
        })
        print(f"  → 列级别加噪：合并列 '{selected_join_col}' 和 '{selected_non_join_col}' -> '{new_col_name}'")
        return df
    
    def apply_column_level_noise_join_col_only(self, df: pd.DataFrame, edge_info: Dict, 
                                                csv_file_name: str, selected_node_id: str) -> pd.DataFrame:
        """专门用于cell_candidates的列级别加噪：只合并连接列和非连接列"""
        table_name = edge_info['table_name']
        edge_key = edge_info['edge_key']
        edge_join_cols = edge_info['edge_join_cols']
        
        selected_node = (edge_info['node1_info'] if selected_node_id == edge_info['node1_id'] 
                        else edge_info['node2_info'])
        all_columns = selected_node.get('columns', [])
        all_join_cols_in_table = self.get_all_join_columns_for_table(table_name)
        non_join_cols = [col for col in all_columns 
                        if col not in all_join_cols_in_table and col in df.columns]
        
        if not non_join_cols:
            print(f"  ⚠ 没有可用的非连接列进行合并")
            return df
        
        if len(edge_join_cols) == 0:
            print(f"  ⚠ 没有连接列可用于合并")
            return df
        
        # 选择连接列和非连接列进行合并
        selected_join_col = edge_join_cols[0]  # cell_candidates只有一个连接列
        selected_non_join_col = random.choice(non_join_cols)
        new_col_name = f"{selected_join_col}-{selected_non_join_col}"
        
        if selected_join_col not in df.columns or selected_non_join_col not in df.columns:
            print(f"  ⚠ 列不存在于DataFrame中")
            return df
        
        df[new_col_name] = df[selected_join_col].astype(str) + '-' + df[selected_non_join_col].astype(str)
        df = df.drop([selected_join_col, selected_non_join_col], axis=1)
        
        self.noise_records.append({
            'table_name': table_name, 'csv_file': csv_file_name,
            'node_id': selected_node_id, 'edge_key': edge_key,
            'noise_level': 'column', 'noise_type': 'column_merge_from_cell_candidate',
            'join_column': selected_join_col, 'non_join_column': selected_non_join_col,
            'merged_column': new_col_name,
            'description': f'将连接列 {selected_join_col} 与非连接列 {selected_non_join_col} 合并为 {new_col_name}'
        })
        print(f"  → 列级别加噪（来自cell候选）：合并列 '{selected_join_col}' 和 '{selected_non_join_col}' -> '{new_col_name}'")
        return df
    
    def find_pivot_column(self, df: pd.DataFrame, edge_info: Dict, selected_node_id: str) -> str:
        """找到一个非数值类型且有2-3个独特值（去除空值）的非连接列，用于pivot操作"""
        table_name = edge_info['table_name']
        edge_join_cols = edge_info['edge_join_cols']
        
        selected_node = (edge_info['node1_info'] if selected_node_id == edge_info['node1_id'] 
                        else edge_info['node2_info'])
        all_columns = selected_node.get('columns', [])
        all_join_cols_in_table = self.get_all_join_columns_for_table(table_name)
        non_join_cols = [col for col in all_columns 
                        if col not in all_join_cols_in_table and col in df.columns]
        
        for col in non_join_cols:
            # 检查是否为非数值类型
            if pd.api.types.is_numeric_dtype(df[col]):
                continue
            
            # 检查独特值数量（去除空值后）
            unique_values = df[col].dropna().nunique()
            if 2 <= unique_values <= 3:
                return col
        return None
    
    def find_unpivot_condition_column(self, df: pd.DataFrame, edge_info: Dict, selected_node_id: str) -> str:
        """找到一个空值占比>85%且独特值<3的非连接列，用于判断是否可以进行unpivot操作
        
        返回: 符合条件的列名（如果找到），否则返回None
        """
        table_name = edge_info['table_name']
        edge_join_cols = edge_info['edge_join_cols']
        
        selected_node = (edge_info['node1_info'] if selected_node_id == edge_info['node1_id'] 
                        else edge_info['node2_info'])
        all_columns = selected_node.get('columns', [])
        all_join_cols_in_table = self.get_all_join_columns_for_table(table_name)
        non_join_cols = [col for col in all_columns 
                        if col not in all_join_cols_in_table and col in df.columns]
        
        for col in non_join_cols:
            total_count = len(df)
            if total_count == 0:
                continue
            
            # 计算空值占比
            null_count = df[col].isna().sum()
            null_ratio = null_count / total_count
            
            # 检查独特值数量（包括空值的情况）
            unique_values = df[col].dropna().nunique()
            
            # 空值占比>85% 且 独特值<3
            if null_ratio > 0.85 and unique_values < 3:
                return col
        return None
    
    def apply_table_level_noise(self, df: pd.DataFrame, edge_info: Dict, 
                                 csv_file_name: str, selected_node_id: str,
                                 pivot_col: str = None) -> pd.DataFrame:
        """表级别加噪：对DataFrame进行pivot操作
        
        Args:
            df: 要加噪的DataFrame
            edge_info: 边信息
            csv_file_name: CSV文件名
            selected_node_id: 选中的节点ID
            pivot_col: 预先确定的pivot列名（如果为None则自动查找）
        """
        table_name = edge_info['table_name']
        edge_key = edge_info['edge_key']
        edge_join_cols = edge_info['edge_join_cols']
        
        # 如果没有预先指定pivot列，则查找
        if pivot_col is None:
            pivot_col = self.find_pivot_column(df, edge_info, selected_node_id)
        
        if pivot_col is None or pivot_col not in df.columns:
            print(f"  ⚠ 未找到适合pivot的列（需要非数值类型且有2-3个独特值）")
            return df
        
        # 使用连接列作为value列
        if len(edge_join_cols) == 0:
            print(f"  ⚠ 没有连接列可用于pivot")
            return df
        
        # 选择第一个连接列作为value列
        value_col = edge_join_cols[0]
        if value_col not in df.columns:
            print(f"  ⚠ 连接列 {value_col} 不存在于DataFrame中")
            return df
        
        # 获取pivot列的唯一值
        pivot_values = df[pivot_col].dropna().unique().tolist()
        
        print(f"  → 表级别加噪：pivot列 '{pivot_col}'，连接列（value列）'{value_col}'")
        print(f"    Pivot列的唯一值: {pivot_values}")
        
        try:
            # 添加行索引用于pivot
            df['_row_id'] = range(len(df))
            
            # 进行pivot操作：用pivot_col的值作为新列名，value_col的值填充
            pivoted = df.pivot(index='_row_id', columns=pivot_col, values=value_col)
            
            # 重命名列："{pivot_value} with {value_col}"
            new_columns = {val: f"{val} with {value_col}" for val in pivoted.columns}
            pivoted = pivoted.rename(columns=new_columns)
            
            # 重置索引
            pivoted = pivoted.reset_index(drop=True)
            
            # 保留其他列（排除pivot列和value列）
            other_cols = [col for col in df.columns 
                         if col not in [pivot_col, value_col, '_row_id', 'original_index']]
            
            # 获取每行的其他列值（需要group by _row_id）
            if other_cols:
                other_data = df.groupby('_row_id')[other_cols].first().reset_index(drop=True)
                result_df = pd.concat([other_data, pivoted], axis=1)
            else:
                result_df = pivoted
            
            # 如果原df有original_index列，保留它
            if 'original_index' in df.columns:
                original_index_map = df.groupby('_row_id')['original_index'].first().reset_index(drop=True)
                result_df.insert(0, 'original_index', original_index_map)
            
            # 将NaN替换为空字符串
            result_df = result_df.fillna('')
            
            self.noise_records.append({
                'table_name': table_name, 'csv_file': csv_file_name,
                'node_id': selected_node_id, 'edge_key': edge_key,
                'noise_level': 'table', 'noise_type': 'pivot',
                'pivot_column': pivot_col, 'join_column_as_value': value_col,
                'pivot_values': pivot_values,
                'new_columns': list(new_columns.values()),
                'description': f'使用 {pivot_col} 列进行pivot，连接列 {value_col} 作为值'
            })
            
            print(f"    ✓ Pivot成功，新列: {list(new_columns.values())}")
            return result_df
            
        except Exception as e:
            print(f"  ⚠ Pivot操作失败: {str(e)}")
            if '_row_id' in df.columns:
                df = df.drop('_row_id', axis=1)
            return df
    
    def apply_unpivot_noise(self, df: pd.DataFrame, edge_info: Dict, 
                             csv_file_name: str, selected_node_id: str,
                             condition_col: str = None) -> pd.DataFrame:
        """表级别加噪（Unpivot）：将宽表转换为长表
        
        将连接列和条件列（空值占比>85%且独特值<3的列）一起unpivot为metric/value形式，
        其他所有列作为id_vars保留并复制到每行。
        
        例如（name是连接列，gender是条件列，id是其他列）：
            gender, name, id
            male, Mike, 1
            female, Jane, 2
            male, Tom, 3
            female, Jenny, 4
        转换为：
            metric, value, id
            name, Mike, 1
            name, Jane, 2
            name, Tom, 3
            name, Jenny, 4
            gender, male, 1
            gender, female, 2
            gender, male, 3
            gender, female, 4
        
        Args:
            df: 要加噪的DataFrame
            edge_info: 边信息
            csv_file_name: CSV文件名
            selected_node_id: 选中的节点ID
            condition_col: 预先确定的条件列名（空值占比>85%且独特值<3的列）
        """
        table_name = edge_info['table_name']
        edge_key = edge_info['edge_key']
        edge_join_cols = edge_info['edge_join_cols']
        
        # 如果没有预先指定条件列，则查找
        if condition_col is None:
            condition_col = self.find_unpivot_condition_column(df, edge_info, selected_node_id)
        
        if condition_col is None:
            print(f"  ⚠ 未找到适合unpivot的条件列（需要空值占比>85%且独特值<3）")
            return df
        
        # 获取连接列
        if len(edge_join_cols) == 0:
            print(f"  ⚠ 没有连接列可用于unpivot")
            return df
        
        join_col = edge_join_cols[0]
        if join_col not in df.columns:
            print(f"  ⚠ 连接列 {join_col} 不存在于DataFrame中")
            return df
        
        if condition_col not in df.columns:
            print(f"  ⚠ 条件列 {condition_col} 不存在于DataFrame中")
            return df
        
        # 要unpivot的列：连接列 + 条件列
        cols_to_unpivot = [join_col, condition_col]
        
        # id_vars：除了要unpivot的列和original_index之外的所有其他列
        id_vars = [col for col in df.columns 
                   if col not in cols_to_unpivot and col != 'original_index']
        
        if not id_vars:
            print(f"  ⚠ 没有其他列可以作为id_vars保留")
            # 如果没有其他列，仍然可以继续，只是结果只有metric和value两列
        
        print(f"  → 表级别加噪（Unpivot）：")
        print(f"    连接列: '{join_col}'")
        print(f"    条件列: '{condition_col}'")
        print(f"    要unpivot的列: {cols_to_unpivot}")
        print(f"    保留的列(id_vars): {id_vars}")
        
        try:
            # 保存original_index（如果存在）
            has_original_index = 'original_index' in df.columns
            
            # 使用pandas melt进行unpivot操作
            # id_vars: 保持不变的列（除了连接列和条件列之外的所有列）
            # value_vars: 要转换的列（连接列 + 条件列）
            # var_name: 新列名（存储原列名）-> 'metric'
            # value_name: 新列名（存储原值）-> 'value'
            
            if id_vars:
                melted = pd.melt(
                    df.drop(columns=['original_index']) if has_original_index else df,
                    id_vars=id_vars,
                    value_vars=cols_to_unpivot,
                    var_name='pivot-metric',
                    value_name='pivot-value'
                )
            else:
                # 如果没有id_vars，只melt value_vars
                melted = pd.melt(
                    df.drop(columns=['original_index']) if has_original_index else df,
                    value_vars=cols_to_unpivot,
                    var_name='pivot-metric',
                    value_name='pivot-value'
                )
            
            # 重新排列列顺序：metric, value, 然后是其他列
            new_col_order = ['pivot-metric', 'pivot-value'] + id_vars
            result_df = melted[new_col_order]
            
            self.noise_records.append({
                'table_name': table_name, 'csv_file': csv_file_name,
                'node_id': selected_node_id, 'edge_key': edge_key,
                'noise_level': 'table', 'noise_type': 'unpivot',
                'join_column': join_col, 'condition_column': condition_col,
                'unpivoted_columns': cols_to_unpivot,
                'id_vars': id_vars,
                'new_columns': new_col_order,
                'description': f'使用unpivot将列 {cols_to_unpivot} 转换为 metric/value 格式，保留列 {id_vars}'
            })
            
            print(f"    ✓ Unpivot成功，原表 {len(df)} 行 -> 新表 {len(result_df)} 行")
            print(f"    新列结构: {new_col_order}")
            return result_df
            
        except Exception as e:
            print(f"  ⚠ Unpivot操作失败: {str(e)}")
            return df
    
    def process_edge_both_nodes(self, edge_info: Dict, noise_type: str):
        """【修改1】对边的两个端点都进行相同的加噪"""
        table_name = edge_info['table_name']
        edge_join_cols = edge_info['edge_join_cols']
        original_shape = edge_info.get('original_shape', [0, 0])
        
        folder_name = f"{table_name}_normal"
        folder_path = self.path_b / folder_name
        if not folder_path.exists():
            print(f"  ⚠ 文件夹不存在: {folder_name}")
            return
        
        # 确定加噪方法
        if len(edge_join_cols) == 0:
            print(f"    ⚠ 该边没有连接列，跳过")
            return
        
        col = edge_join_cols[0]
        
        # 先读取第一个节点的数据来确定列类型和加噪方法
        node1_csv_file = edge_info['node1_info'].get('subtable_file', '')
        if node1_csv_file:
            csv_file_path = folder_path / f"{node1_csv_file}.csv"
            if csv_file_path.exists():
                temp_df = pd.read_csv(csv_file_path)
                if col in temp_df.columns:
                    col_type = self.detect_column_type(temp_df[col])
                else:
                    print(f"    ⚠ 列 {col} 不存在，跳过")
                    return
            else:
                print(f"    ⚠ CSV文件不存在: {node1_csv_file}")
                return
        else:
            print(f"    ⚠ 节点1没有对应的CSV文件")
            return
        
        # 根据列类型选择加噪方法列表
        if col_type == 'numeric':
            noise_methods = [self.apply_scientific_notation, self.apply_percentage_format]
        elif col_type == 'datetime':
            noise_methods = [self.apply_datetime_format_change]
        elif col_type == 'string':
            noise_methods = [
                self.apply_keyboard_mistake, self.apply_lowercase,
                self.apply_extra_space, self.apply_special_char,
                self.apply_substring, self.apply_abbreviation
            ]
        else:
            print(f"    ⚠ 列 {col} 类型未知，跳过")
            return
        
        # 计算加噪行数
        base_table_rows = original_shape[0]
        num_rows_to_noise = max(1, int(base_table_rows * 0.01))
        
        print(f"    单元格加噪：列 '{col}' ({col_type})，可用加噪方法: {[m.__name__ for m in noise_methods]}")
        
        # 对两个节点都进行加噪
        for node_id, node_info in [(edge_info['node1_id'], edge_info['node1_info']), 
                                    (edge_info['node2_id'], edge_info['node2_info'])]:
            csv_file_name = node_info.get('subtable_file', '')
            if not csv_file_name:
                print(f"    ⚠ 节点 {node_id} 没有对应的CSV文件，跳过")
                continue
            
            csv_file = folder_path / f"{csv_file_name}.csv"
            if not csv_file.exists():
                print(f"    ⚠ CSV文件不存在: {csv_file_name}")
                continue
            
            print(f"    处理文件: {csv_file.name} (节点: {node_id})")
            df = pd.read_csv(csv_file)
            df.insert(0, 'original_index', range(len(df)))
            
            # 确定该节点的加噪行数
            num_rows = len(df)
            actual_rows_to_noise = min(num_rows_to_noise, num_rows)
            noised_rows = list(range(actual_rows_to_noise))
            
            print(f"      子表行数 {num_rows}，加噪 {actual_rows_to_noise} 行（每行随机选择加噪方法）")
            
            # 应用加噪（传递方法列表，每行随机选择）
            df = self.apply_cell_level_noise_to_single_node(
                df, edge_info, csv_file.name, node_id, noise_methods, noised_rows
            )
            
            # 打乱行顺序
            df_shuffled = df.sample(frac=1, random_state=random.randint(0, 10000)).reset_index(drop=True)
            
            # 更新记录中的shuffled_index
            for record in self.noise_records:
                if (record['table_name'] == table_name and 
                    record.get('csv_file') == csv_file.name and 
                    'original_index' in record and 'shuffled_index' not in record):
                    orig_idx = record['original_index']
                    shuffled_position = df_shuffled[df_shuffled['original_index'] == orig_idx].index
                    if len(shuffled_position) > 0:
                        record['shuffled_index'] = int(shuffled_position[0])
            
            df_shuffled = df_shuffled.drop('original_index', axis=1)
            df_shuffled.to_csv(csv_file, index=False)
            print(f"      ✓ 已保存: {csv_file.name}")
    
    def process_edge(self, edge_info: Dict, noise_type: str):
        """处理单个节点的加噪（用于列级别和表级别加噪）"""
        table_name = edge_info['table_name']
        selected_node_id = random.choice([edge_info['node1_id'], edge_info['node2_id']])
        selected_node = (edge_info['node1_info'] if selected_node_id == edge_info['node1_id'] 
                        else edge_info['node2_info'])
        
        csv_file_name = selected_node.get('subtable_file', '')
        if not csv_file_name:
            print(f"  ⚠ 选中节点没有对应的CSV文件")
            return
        
        folder_name = f"{table_name}_normal"
        folder_path = self.path_b / folder_name
        if not folder_path.exists():
            print(f"  ⚠ 文件夹不存在: {folder_name}")
            return
        
        csv_file = folder_path / f"{csv_file_name}.csv"
        if not csv_file.exists():
            print(f"  ⚠ CSV文件不存在: {csv_file_name}")
            return
        
        print(f"  处理文件: {csv_file.name} (节点: {selected_node_id})")
        df = pd.read_csv(csv_file)
        df.insert(0, 'original_index', range(len(df)))
        
        if noise_type == 'cell':
            df = self.apply_cell_level_noise(df, edge_info, csv_file.name, selected_node_id)
        elif noise_type == 'column':
            df = self.apply_column_level_noise(df, edge_info, csv_file.name, selected_node_id)
        elif noise_type == 'column_from_cell':
            df = self.apply_column_level_noise_join_col_only(df, edge_info, csv_file.name, selected_node_id)
        elif noise_type == 'table':
            df = self.apply_table_level_noise(df, edge_info, csv_file.name, selected_node_id)
        
        df_shuffled = df.sample(frac=1, random_state=random.randint(0, 10000)).reset_index(drop=True)
        
        if noise_type == 'cell':
            for record in self.noise_records:
                if (record['table_name'] == table_name and 
                    record.get('csv_file') == csv_file.name and 
                    'original_index' in record and 'shuffled_index' not in record):
                    orig_idx = record['original_index']
                    shuffled_position = df_shuffled[df_shuffled['original_index'] == orig_idx].index
                    if len(shuffled_position) > 0:
                        record['shuffled_index'] = int(shuffled_position[0])
        
        if 'original_index' in df_shuffled.columns:
            df_shuffled = df_shuffled.drop('original_index', axis=1)
        df_shuffled.to_csv(csv_file, index=False)
        print(f"    ✓ 已保存: {csv_file.name}")
    
    def process_edge_for_table(self, edge_info: Dict, selected_node_id: str, pivot_col: str):
        """专门用于表级别加噪的处理方法，使用预先确定的节点和pivot列"""
        table_name = edge_info['table_name']
        selected_node = (edge_info['node1_info'] if selected_node_id == edge_info['node1_id'] 
                        else edge_info['node2_info'])
        
        csv_file_name = selected_node.get('subtable_file', '')
        if not csv_file_name:
            print(f"  ⚠ 选中节点没有对应的CSV文件")
            return
        
        folder_name = f"{table_name}_normal"
        folder_path = self.path_b / folder_name
        if not folder_path.exists():
            print(f"  ⚠ 文件夹不存在: {folder_name}")
            return
        
        csv_file = folder_path / f"{csv_file_name}.csv"
        if not csv_file.exists():
            print(f"  ⚠ CSV文件不存在: {csv_file_name}")
            return
        
        print(f"  处理文件: {csv_file.name} (节点: {selected_node_id})")
        df = pd.read_csv(csv_file)
        df.insert(0, 'original_index', range(len(df)))
        
        # 使用预先确定的pivot列进行表级别加噪
        df = self.apply_table_level_noise(df, edge_info, csv_file.name, selected_node_id, pivot_col)
        
        df_shuffled = df.sample(frac=1, random_state=random.randint(0, 10000)).reset_index(drop=True)
        
        if 'original_index' in df_shuffled.columns:
            df_shuffled = df_shuffled.drop('original_index', axis=1)
        df_shuffled.to_csv(csv_file, index=False)
        print(f"    ✓ 已保存: {csv_file.name}")
    
    def process_edge_for_unpivot(self, edge_info: Dict, selected_node_id: str, condition_col: str):
        """专门用于unpivot表级别加噪的处理方法，使用预先确定的节点和条件列"""
        table_name = edge_info['table_name']
        selected_node = (edge_info['node1_info'] if selected_node_id == edge_info['node1_id'] 
                        else edge_info['node2_info'])
        
        csv_file_name = selected_node.get('subtable_file', '')
        if not csv_file_name:
            print(f"  ⚠ 选中节点没有对应的CSV文件")
            return
        
        folder_name = f"{table_name}_normal"
        folder_path = self.path_b / folder_name
        if not folder_path.exists():
            print(f"  ⚠ 文件夹不存在: {folder_name}")
            return
        
        csv_file = folder_path / f"{csv_file_name}.csv"
        if not csv_file.exists():
            print(f"  ⚠ CSV文件不存在: {csv_file_name}")
            return
        
        print(f"  处理文件: {csv_file.name} (节点: {selected_node_id})")
        df = pd.read_csv(csv_file)
        df.insert(0, 'original_index', range(len(df)))
        
        # 使用预先确定的条件列进行unpivot表级别加噪
        df = self.apply_unpivot_noise(df, edge_info, csv_file.name, selected_node_id, condition_col)
        
        df_shuffled = df.sample(frac=1, random_state=random.randint(0, 10000)).reset_index(drop=True)
        
        if 'original_index' in df_shuffled.columns:
            df_shuffled = df_shuffled.drop('original_index', axis=1)
        df_shuffled.to_csv(csv_file, index=False)
        print(f"    ✓ 已保存: {csv_file.name}")
    
    def check_table_level_eligibility(self, edge_info: Dict) -> Tuple[bool, str, str]:
        """
        检查边是否适合进行表级别加噪（需要存在非数值类型且有2-3个独特值的非连接列）
        返回: (是否符合条件, 符合条件的节点ID, 找到的pivot列名)
        """
        table_name = edge_info['table_name']
        folder_name = f"{table_name}_normal"
        folder_path = self.path_b / folder_name
        
        if not folder_path.exists():
            return False, None, None
        
        # 检查两个节点中是否至少有一个满足条件
        for node_id, node_info in [(edge_info['node1_id'], edge_info['node1_info']), 
                                    (edge_info['node2_id'], edge_info['node2_info'])]:
            csv_file_name = node_info.get('subtable_file', '')
            if not csv_file_name:
                continue
            
            csv_file = folder_path / f"{csv_file_name}.csv"
            if not csv_file.exists():
                continue
            
            df = pd.read_csv(csv_file)
            pivot_col = self.find_pivot_column(df, edge_info, node_id)
            if pivot_col is not None:
                return True, node_id, pivot_col
        
        return False, None, None
    
    def check_unpivot_eligibility(self, edge_info: Dict) -> Tuple[bool, str, str]:
        """
        检查边是否适合进行unpivot加噪（需要存在空值占比>85%且独特值<3的非连接列作为条件列）
        返回: (是否符合条件, 符合条件的节点ID, 找到的条件列名)
        """
        table_name = edge_info['table_name']
        folder_name = f"{table_name}_normal"
        folder_path = self.path_b / folder_name
        
        if not folder_path.exists():
            return False, None, None
        
        # 检查两个节点中是否至少有一个满足条件
        for node_id, node_info in [(edge_info['node1_id'], edge_info['node1_info']), 
                                    (edge_info['node2_id'], edge_info['node2_info'])]:
            csv_file_name = node_info.get('subtable_file', '')
            if not csv_file_name:
                continue
            
            csv_file = folder_path / f"{csv_file_name}.csv"
            if not csv_file.exists():
                continue
            
            df = pd.read_csv(csv_file)
            unpivot_col = self.find_unpivot_condition_column(df, edge_info, node_id)
            if unpivot_col is not None:
                return True, node_id, unpivot_col
        
        return False, None, None
    
    def save_noise_records(self):
        records_file = self.path_b / 'noise_records.json'
        cell_count = len([r for r in self.noise_records if r.get('noise_level') == 'cell'])
        column_count = len([r for r in self.noise_records if r.get('noise_level') == 'column'])
        table_count = len([r for r in self.noise_records if r.get('noise_level') == 'table'])
        pivot_count = len([r for r in self.noise_records if r.get('noise_level') == 'table' and r.get('noise_type') == 'pivot'])
        unpivot_count = len([r for r in self.noise_records if r.get('noise_level') == 'table' and r.get('noise_type') == 'unpivot'])
        
        cell_edges = len(set(r.get('edge_key') for r in self.noise_records if r.get('noise_level') == 'cell'))
        column_edges = len(set(r.get('edge_key') for r in self.noise_records if r.get('noise_level') == 'column'))
        table_edges = len(set(r.get('edge_key') for r in self.noise_records if r.get('noise_level') == 'table'))
        pivot_edges = len(set(r.get('edge_key') for r in self.noise_records if r.get('noise_level') == 'table' and r.get('noise_type') == 'pivot'))
        unpivot_edges = len(set(r.get('edge_key') for r in self.noise_records if r.get('noise_level') == 'table' and r.get('noise_type') == 'unpivot'))
        
        summary = {
            'total_records': len(self.noise_records),
            'cell_level_count': cell_count, 'cell_level_edge_count': cell_edges,
            'column_level_count': column_count, 'column_level_edge_count': column_edges,
            'table_level_count': table_count, 'table_level_edge_count': table_edges,
            'table_level_pivot_count': pivot_count, 'table_level_pivot_edge_count': pivot_edges,
            'table_level_unpivot_count': unpivot_count, 'table_level_unpivot_edge_count': unpivot_edges,
            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        }
        output = {'summary': summary, 'noise_records': self.noise_records}
        
        with open(records_file, 'w', encoding='utf-8') as f:
            json.dump(output, f, ensure_ascii=False, indent=2)
        
        print(f"\n✓ 加噪记录已保存到: {records_file}")
        print(f"  - 单元格级别: {cell_count} 条记录 ({cell_edges} 条边)")
        print(f"  - 列级别: {column_count} 条 ({column_edges} 条边)")
        print(f"  - 表级别: {table_count} 条 ({table_edges} 条边)")
        print(f"    - Pivot: {pivot_count} 条 ({pivot_edges} 条边)")
        print(f"    - Unpivot: {unpivot_count} 条 ({unpivot_edges} 条边)")
    
    def run(self, json_filename: str):
        print("="*70)
        print(" "*20 + "基于边的数据加噪流程启动")
        print("="*70)
        
        print("\n【阶段1】数据准备")
        self.copy_data()
        self.load_json(json_filename)
        self.check_and_clean_tables()
        
        print("\n【阶段2】筛选符合条件的边")
        cell_candidates, col_candidates = self.extract_edge_candidates()
        print(f"  符合单元格级别加噪条件: {len(cell_candidates)} 条边 (连接列<2)")
        print(f"  符合列级别加噪条件: {len(col_candidates)} 条边 (连接列≥2且含经纬度)")
        
        print("\n【阶段3】选择加噪边")
        
        # 跟踪已选择的cell_candidates，使用 table_name::edge_key 作为唯一标识符
        selected_cell_edge_ids = set()
        
        # 3.1 选择单元格级别加噪的边
        cell_count = min(random.randint(160, 180), len(cell_candidates))
        cell_edges = self.select_edge_for_noise(cell_candidates, cell_count)
        selected_cell_edge_ids.update(self.get_edge_unique_id(e) for e in cell_edges)
        
        # 3.2 选择列级别加噪的边（原有的col_candidates）
        # 先过滤出有非连接列的边
        col_candidates_with_non_join = [c for c in col_candidates if self.check_has_non_join_columns(c)]
        col_count = min(10, len(col_candidates_with_non_join))
        col_edges = self.select_edge_for_noise(col_candidates_with_non_join, col_count)
        print(f"  列级别候选（原有）: {len(col_candidates)} 条边，有非连接列: {len(col_candidates_with_non_join)} 条边")
        
        # 3.3 【修改2】从未被选中的cell_candidates中选择10个进行列级别加噪
        # 需要检查是否有非连接列
        remaining_cell_for_col = [c for c in cell_candidates 
                                   if self.get_edge_unique_id(c) not in selected_cell_edge_ids]
        # 过滤出有非连接列的边
        remaining_cell_with_non_join = [c for c in remaining_cell_for_col 
                                         if self.check_has_non_join_columns(c)]
        col_from_cell_count = min(10, len(remaining_cell_with_non_join))
        col_from_cell_edges = self.select_edge_for_noise(remaining_cell_with_non_join, col_from_cell_count)
        selected_cell_edge_ids.update(self.get_edge_unique_id(e) for e in col_from_cell_edges)
        print(f"  列级别候选（来自cell）: 剩余 {len(remaining_cell_for_col)} 条边，有非连接列: {len(remaining_cell_with_non_join)} 条边")
        
        # 3.4 【修改3】从未被选中的cell_candidates中选择8个进行表级别加噪
        # 需要检查这些边是否有适合pivot的列（非数值类型且有2-3个独特值）
        remaining_cell_for_table = [c for c in cell_candidates 
                                     if self.get_edge_unique_id(c) not in selected_cell_edge_ids]
        
        # 筛选符合表级别加噪条件的边，同时记录节点ID和pivot列
        table_eligible_candidates = []
        table_pivot_info = {}  # 存储每个边的pivot信息: edge_unique_id -> (node_id, pivot_col)
        for candidate in remaining_cell_for_table:
            is_eligible, node_id, pivot_col = self.check_table_level_eligibility(candidate)
            if is_eligible:
                table_eligible_candidates.append(candidate)
                edge_uid = self.get_edge_unique_id(candidate)
                table_pivot_info[edge_uid] = (node_id, pivot_col)
        
        table_count = min(8, len(table_eligible_candidates))
        table_edges = self.select_edge_for_noise(table_eligible_candidates, table_count)
        selected_cell_edge_ids.update(self.get_edge_unique_id(e) for e in table_edges)
        
        # 3.5 从未被选中的cell_candidates中选择8个进行unpivot表级别加噪
        # 需要检查这些边是否有适合unpivot的条件列（空值占比>85%且独特值<3）
        remaining_cell_for_unpivot = [c for c in cell_candidates 
                                       if self.get_edge_unique_id(c) not in selected_cell_edge_ids]
        
        # 筛选符合unpivot条件的边，同时记录节点ID和条件列
        unpivot_eligible_candidates = []
        unpivot_info = {}  # 存储每个边的unpivot信息: edge_unique_id -> (node_id, condition_col)
        for candidate in remaining_cell_for_unpivot:
            is_eligible, node_id, condition_col = self.check_unpivot_eligibility(candidate)
            if is_eligible:
                unpivot_eligible_candidates.append(candidate)
                edge_uid = self.get_edge_unique_id(candidate)
                unpivot_info[edge_uid] = (node_id, condition_col)
        
        unpivot_count = min(8, len(unpivot_eligible_candidates))
        unpivot_edges = self.select_edge_for_noise(unpivot_eligible_candidates, unpivot_count)
        selected_cell_edge_ids.update(self.get_edge_unique_id(e) for e in unpivot_edges)
        
        print(f"  实际选择:")
        print(f"    - 单元格级别: {len(cell_edges)} 条边")
        print(f"    - 列级别（原有）: {len(col_edges)} 条边")
        print(f"    - 列级别（来自cell候选）: {len(col_from_cell_edges)} 条边")
        print(f"    - 表级别（Pivot）: {len(table_edges)} 条边 (共检查 {len(remaining_cell_for_table)} 条，符合条件 {len(table_eligible_candidates)} 条)")
        print(f"    - 表级别（Unpivot）: {len(unpivot_edges)} 条边 (共检查 {len(remaining_cell_for_unpivot)} 条，符合条件 {len(unpivot_eligible_candidates)} 条)")
        
        print("\n【阶段4】执行加噪操作")
        
        # 4.1 执行单元格级别加噪（两个端点都加噪）
        print("\n--- 4.1 单元格级别加噪（双端点）---")
        for idx, edge_info in enumerate(cell_edges, 1):
            print(f"\n[Cell {idx}/{len(cell_edges)}] 边: {edge_info['edge_key']}")
            print(f"  所属表格: {edge_info['table_name']}")
            print(f"  连接列数: {edge_info['num_join_cols']}")
            print(f"  连接列: {', '.join(edge_info['edge_join_cols'])}")
            print(f"  基表形状: {edge_info['original_shape']}")
            self.process_edge_both_nodes(edge_info, 'cell')
        
        # 4.2 执行列级别加噪（原有的col_candidates）
        print("\n--- 4.2 列级别加噪（原有）---")
        for idx, edge_info in enumerate(col_edges, 1):
            print(f"\n[Col {idx}/{len(col_edges)}] 边: {edge_info['edge_key']}")
            print(f"  所属表格: {edge_info['table_name']}")
            print(f"  连接列数: {edge_info['num_join_cols']}")
            print(f"  连接列: {', '.join(edge_info['edge_join_cols'])}")
            print(f"  基表形状: {edge_info['original_shape']}")
            self.process_edge(edge_info, 'column')
        
        # 4.3 执行列级别加噪（来自cell候选）
        print("\n--- 4.3 列级别加噪（来自cell候选）---")
        for idx, edge_info in enumerate(col_from_cell_edges, 1):
            print(f"\n[ColFromCell {idx}/{len(col_from_cell_edges)}] 边: {edge_info['edge_key']}")
            print(f"  所属表格: {edge_info['table_name']}")
            print(f"  连接列数: {edge_info['num_join_cols']}")
            print(f"  连接列: {', '.join(edge_info['edge_join_cols'])}")
            print(f"  基表形状: {edge_info['original_shape']}")
            self.process_edge(edge_info, 'column_from_cell')
        
        # 4.4 执行表级别加噪（使用预先确定的pivot列和节点）
        print("\n--- 4.4 表级别加噪（Pivot）---")
        for idx, edge_info in enumerate(table_edges, 1):
            edge_uid = self.get_edge_unique_id(edge_info)
            selected_node_id, pivot_col = table_pivot_info[edge_uid]
            print(f"\n[Table {idx}/{len(table_edges)}] 边: {edge_info['edge_key']}")
            print(f"  所属表格: {edge_info['table_name']}")
            print(f"  连接列数: {edge_info['num_join_cols']}")
            print(f"  连接列: {', '.join(edge_info['edge_join_cols'])}")
            print(f"  基表形状: {edge_info['original_shape']}")
            print(f"  预选节点: {selected_node_id}, Pivot列: {pivot_col}")
            self.process_edge_for_table(edge_info, selected_node_id, pivot_col)
        
        # 4.5 执行表级别加噪（Unpivot，使用预先确定的条件列和节点）
        print("\n--- 4.5 表级别加噪（Unpivot）---")
        for idx, edge_info in enumerate(unpivot_edges, 1):
            edge_uid = self.get_edge_unique_id(edge_info)
            selected_node_id, condition_col = unpivot_info[edge_uid]
            print(f"\n[Unpivot {idx}/{len(unpivot_edges)}] 边: {edge_info['edge_key']}")
            print(f"  所属表格: {edge_info['table_name']}")
            print(f"  连接列数: {edge_info['num_join_cols']}")
            print(f"  连接列: {', '.join(edge_info['edge_join_cols'])}")
            print(f"  基表形状: {edge_info['original_shape']}")
            print(f"  预选节点: {selected_node_id}, 条件列: {condition_col}")
            self.process_edge_for_unpivot(edge_info, selected_node_id, condition_col)
        
        print("\n【阶段5】保存加噪记录")
        self.save_noise_records()
        
        print("\n" + "="*70)
        print(" "*25 + "流程完成！")
        print("="*70)


if __name__ == "__main__":
    path_a = "./dataset"
    path_b = "./noisy_dataset_nyc"
    json_filename = "split_statistics.json"
    
    injector = EdgeBasedNoiseInjector(path_a, path_b)
    injector.run(json_filename)