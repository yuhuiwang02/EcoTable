#!/usr/bin/env python3
"""
使用combination + LLM生成SQL查询
流程：
1. 为每个join path生成所有可能的节点combination (2^(N-M)种)
2. 对每个combination使用模板，让LLM填充列
"""

import json
import os
import csv
import re
from typing import List, Dict, Set, Tuple
from collections import defaultdict
import requests
import time

# ================== 配置 ==================
STATISTICS_FILE = '/data2/liujinqi/Revision/SQL_generation/splited_table-250/dataset/split_statistics.json'
TEMPLATE_FILE = '/data2/liujinqi/Revision/SQL_generation/refactored_templates_v2_normal_fixed.json'
OUTPUT_FILE = '/data2/liujinqi/Revision/SQL_generation/llm_filled_queries_with_combinations.json'
DATA_DIR = '/data2/liujinqi/Revision/SQL_generation/splited_table-250/dataset'

K_TEMPLATES = 2  # 每个combination选择2个不同的模板
MAX_TABLES = None  # 处理所有表

# LLM配置
LLM_API_ENDPOINT = "https://www.dmxapi.com/v1/chat/completions"
LLM_API_KEY = ""
LLM_MODEL = "gpt-5.1"


# ================== 辅助函数 ==================

def parse_pattern(pattern: str) -> List[Tuple[str, str]]:
    """解析树结构pattern"""
    edges = []
    for edge_str in pattern.strip().split():
        parent, child = edge_str.split('-')
        edges.append((parent, child))
    return edges


def identify_all_nodes(edges: List[Tuple[str, str]]) -> Set[str]:
    """识别所有节点"""
    nodes = set()
    for parent, child in edges:
        nodes.add(parent)
        nodes.add(child)
    return nodes


def identify_leaf_nodes(edges: List[Tuple[str, str]]) -> Set[str]:
    """识别叶子节点"""
    parents = set()
    children = set()
    for parent, child in edges:
        parents.add(parent)
        children.add(child)
    return children - parents


def is_connected(selected_nodes: Set[str], edges: List[Tuple[str, str]]) -> bool:
    """检查选中的节点是否连通（可以通过边相互到达）"""
    if len(selected_nodes) <= 1:
        return True

    # 只保留选中节点之间的边
    selected_edges = [(p, c) for p, c in edges if p in selected_nodes and c in selected_nodes]

    if not selected_edges:
        return False  # 没有边连接，不连通

    # BFS检查连通性
    visited = set()
    queue = [next(iter(selected_nodes))]  # 从任意节点开始
    visited.add(queue[0])

    # 构建邻接表
    adjacency = defaultdict(list)
    for p, c in selected_edges:
        adjacency[p].append(c)
        adjacency[c].append(p)  # 无向图

    while queue:
        node = queue.pop(0)
        for neighbor in adjacency[node]:
            if neighbor not in visited:
                visited.add(neighbor)
                queue.append(neighbor)

    return len(visited) == len(selected_nodes)


def count_connected_components(selected_nodes: Set[str], edges: List[Tuple[str, str]]) -> int:
    """计算选中节点中有多少个连通分量"""
    if len(selected_nodes) <= 1:
        return len(selected_nodes)

    # 只保留选中节点之间的边
    selected_edges = [(p, c) for p, c in edges if p in selected_nodes and c in selected_nodes]

    # 构建邻接表
    adjacency = defaultdict(list)
    for p, c in selected_edges:
        adjacency[p].append(c)
        adjacency[c].append(p)

    # BFS计数连通分量
    visited = set()
    num_components = 0

    for start_node in selected_nodes:
        if start_node in visited:
            continue

        # 新的连通分量
        num_components += 1
        queue = [start_node]
        visited.add(start_node)

        while queue:
            node = queue.pop(0)
            for neighbor in adjacency[node]:
                if neighbor not in visited:
                    visited.add(neighbor)
                    queue.append(neighbor)

    return num_components


def find_high_frequency_columns(table_data: Dict) -> Dict[str, Set[str]]:
    """
    找出每个节点的高频列：在多个节点中出现的列
    返回：{node_id: set(high_freq_columns)}
    """
    node_mapping = table_data['node_mapping']

    # 统计每列在多少个节点中出现
    col_frequency = defaultdict(int)
    for node_id, node_info in node_mapping.items():
        for col in node_info['columns']:
            col_frequency[col] += 1

    # 为每个节点找出其高频列（出现频率>1的列）
    high_freq_cols = {}
    for node_id, node_info in node_mapping.items():
        node_cols = set(node_info['columns'])
        high_freq = {col for col in node_cols if col_frequency[col] > 1}
        high_freq_cols[node_id] = high_freq

    return high_freq_cols


def find_minimal_connecting_path(selected_nodes: Set[str], edges: List[Tuple[str, str]]) -> Set[str]:
    """
    找到连接selected_nodes的最小路径，可能需要包含中间节点
    返回：包含所有必需节点的集合（selected_nodes + 中间节点）
    """
    if len(selected_nodes) <= 1:
        return selected_nodes

    # 构建完整的图结构
    adjacency = defaultdict(list)
    all_nodes = set()
    for p, c in edges:
        adjacency[p].append(c)
        adjacency[c].append(p)
        all_nodes.add(p)
        all_nodes.add(c)

    # 使用Steiner树的简化版本：找到连接所有selected_nodes的最小子图
    # 方法：从selected_nodes中任选一个作为根，用BFS找到到其他selected_nodes的路径

    nodes_in_path = set()
    selected_list = list(selected_nodes)
    root = selected_list[0]

    # 从root到每个其他selected节点找最短路径
    for target in selected_list[1:]:
        # BFS找最短路径
        queue = [(root, [root])]
        visited = {root}

        while queue:
            node, path = queue.pop(0)

            if node == target:
                # 找到路径，添加路径上的所有节点
                nodes_in_path.update(path)
                break

            for neighbor in adjacency[node]:
                if neighbor not in visited:
                    visited.add(neighbor)
                    queue.append((neighbor, path + [neighbor]))

    return nodes_in_path


def generate_all_combinations(table_data: Dict) -> List[Dict]:
    """
    为一个join path生成所有可能的节点combination
    如果有N个节点、M个叶子节点，生成2^(N-M)种组合
    """
    edges = parse_pattern(table_data['pattern'])
    all_nodes = identify_all_nodes(edges)
    leaf_nodes = identify_leaf_nodes(edges)
    non_leaf_nodes = all_nodes - leaf_nodes

    # 获取高频列
    high_freq_cols = find_high_frequency_columns(table_data)

    combinations = []

    # 生成所有可能的非叶子节点组合（2^|non_leaf_nodes|种）
    for i in range(2 ** len(non_leaf_nodes)):
        # 叶子节点必选
        user_selected_nodes = set(leaf_nodes)

        # 根据二进制位决定是否选择非叶子节点
        non_leaf_list = sorted(non_leaf_nodes)
        for j, node in enumerate(non_leaf_list):
            if i & (1 << j):  # 第j位是1，选择这个节点
                user_selected_nodes.add(node)

        # JOIN中的节点 = join path中的所有节点（固定不变）
        nodes_in_join = all_nodes

        # 确定需要的JOIN列（对所有参与JOIN的节点）
        required_join_columns = {}  # {(node, col): True}

        # 对于选中的每条边，如果两端都在nodes_in_join中，需要JOIN列
        for parent, child in edges:
            if parent in nodes_in_join and child in nodes_in_join:
                edge_key = f"{parent}-{child}"
                join_cols = table_data['edge_join_columns'].get(edge_key, [])
                for col in join_cols:
                    required_join_columns[(parent, col)] = True
                    required_join_columns[(child, col)] = True

        # 对于每个节点，确定可选的列
        available_columns = {}
        for node in nodes_in_join:
            node_info = table_data['node_mapping'][node]
            all_cols = set(node_info['columns'])

            # 这个节点的JOIN列
            join_cols_for_node = set()
            for parent, child in edges:
                if parent == node or child == node:
                    edge_key = f"{parent}-{child}"
                    if edge_key in table_data['edge_join_columns']:
                        join_cols_for_node.update(table_data['edge_join_columns'][edge_key])

            # 非JOIN列
            non_join_cols = all_cols - join_cols_for_node

            # 高频非JOIN列
            high_freq_non_join = high_freq_cols[node] & non_join_cols

            available_columns[node] = {
                'all_columns': list(all_cols),
                'non_join_columns': list(non_join_cols),
                'high_freq_non_join_columns': list(high_freq_non_join),
                'is_leaf': node in leaf_nodes,
                'can_select_columns': node in user_selected_nodes  # 只有用户选择的节点才能选列
            }

        # 构建JOIN clause（包含所有必需的节点）
        join_clause = build_join_clause_for_combination(
            nodes_in_join, edges, table_data['edge_join_columns'], table_data['node_mapping']
        )

        combinations.append({
            'user_selected_nodes': sorted(user_selected_nodes),  # 用户选择的节点（可以选列）
            'nodes_in_join': sorted(nodes_in_join),  # JOIN需要的所有节点（包括中间节点）
            'leaf_nodes': sorted(leaf_nodes),
            'required_join_columns': list(required_join_columns.keys()),
            'available_columns': available_columns,
            'join_clause': join_clause,
            'num_joins': len([1 for p, c in edges if p in nodes_in_join and c in nodes_in_join])
        })

    return combinations


def build_join_clause_for_combination(
    nodes_in_join: Set[str],
    edges: List[Tuple[str, str]],
    edge_join_columns: Dict,
    node_mapping: Dict
) -> str:
    """为一个combination构建JOIN clause（所有nodes_in_join已经连通）"""
    # 只保留这些节点之间的边
    selected_edges = [(p, c) for p, c in edges if p in nodes_in_join and c in nodes_in_join]

    if not selected_edges:
        # 单个节点
        node = next(iter(nodes_in_join))
        table = node_mapping[node]['subtable_file']
        return f"FROM `{table}` t{node}"

    # 找根节点
    children = set(c for _, c in selected_edges)
    parents = set(p for p, _ in selected_edges)
    root_candidates = parents - children

    if not root_candidates:
        root_node = sorted(nodes_in_join)[0]
    else:
        root_node = sorted(root_candidates)[0]

    # 构建FROM
    root_table = node_mapping[root_node]['subtable_file']
    join_clause = f"FROM `{root_table}` t{root_node}"

    # BFS构建JOIN
    processed = {root_node}
    while len(processed) < len(nodes_in_join):
        added = False
        for parent, child in selected_edges:
            if parent in processed and child not in processed:
                child_table = node_mapping[child]['subtable_file']
                edge_key = f"{parent}-{child}"
                join_cols = edge_join_columns.get(edge_key, [])

                conditions = []
                for col in join_cols:
                    needs_quote = any(c in col for c in [':', '-', '@', ' ', '.', '(', ')'])
                    col_escaped = f'`{col}`' if needs_quote else col
                    conditions.append(f"t{parent}.{col_escaped} = t{child}.{col_escaped}")

                join_clause += f"\nINNER JOIN `{child_table}` t{child} ON {' AND '.join(conditions)}"
                processed.add(child)
                added = True

        if not added:
            break

    return join_clause


# ================== 数据采样 ==================

def sample_table_data(csv_path: str, num_rows: int = 2) -> Dict:
    """采样CSV表的前N行"""
    try:
        with open(csv_path, 'r', encoding='utf-8', errors='ignore') as f:
            reader = csv.reader(f)
            rows = []
            for i, row in enumerate(reader):
                rows.append(row)
                if i >= num_rows:
                    break

        if not rows:
            return {'header': [], 'rows': []}

        return {
            'header': rows[0] if rows else [],
            'rows': rows[1:] if len(rows) > 1 else []
        }
    except Exception as e:
        return {'header': [], 'rows': [], 'error': str(e)}


def prepare_table_samples(table_data: Dict, selected_nodes: List[str]) -> Dict[str, Dict]:
    """准备选中节点的样本数据"""
    table_name = table_data['table_name']
    category = table_data.get('category', 'normal')
    node_path = f"{DATA_DIR}/{table_name}_{category}"

    node_mapping = table_data.get('node_mapping', {})

    samples = {}
    for node_id in selected_nodes:
        if node_id not in node_mapping:
            continue

        node_info = node_mapping[node_id]
        subtable_file = node_info['subtable_file']
        csv_path = os.path.join(node_path, f"{subtable_file}.csv")

        if os.path.exists(csv_path):
            sample = sample_table_data(csv_path)
            samples[node_id] = {
                'table_alias': f't{node_id}',
                'table_name': subtable_file,
                'columns': node_info.get('columns', []),
                'sample_data': sample
            }

    return samples


# ================== 模板选择 ==================

def detect_sql_features(template: Dict) -> Set[str]:
    """检测模板的SQL特性"""
    pattern = template.get('original_pattern', template.get('pattern', ''))
    features = set()

    if re.search(r'\b(SUM|COUNT|AVG|MAX|MIN)\s*\(', pattern, re.IGNORECASE):
        features.add('aggregation')

    if re.search(r'\b(UNION|INTERSECT|EXCEPT)\b', pattern, re.IGNORECASE):
        features.add('set_operators')

    if re.search(r'\(SELECT\b', pattern, re.IGNORECASE):
        features.add('subqueries')

    if re.search(r'\bOVER\s*\(', pattern, re.IGNORECASE):
        features.add('window_functions')

    if re.search(r'\bWITH\b.*\bAS\s*\(', pattern, re.IGNORECASE | re.DOTALL):
        features.add('cte')

    return features


def select_templates_for_combination(
    all_templates: List[Dict],
    num_joins: int,
    combination_index: int,
    k: int = K_TEMPLATES
) -> List[Dict]:
    """为一个combination选择k个不同的模板"""
    # 筛选JOIN数量匹配的模板
    matching_templates = []
    for tpl in all_templates:
        if not tpl['join_requirements']['has_joins']:
            continue

        tpl_joins = tpl['join_requirements']['join_count']
        if abs(tpl_joins - num_joins) <= 1:  # 允许±1
            tpl['features_set'] = detect_sql_features(tpl)
            matching_templates.append(tpl)

    if not matching_templates:
        return []

    # 按SQL特性分组
    feature_groups = {
        'cte': [],
        'window_functions': [],
        'subqueries': [],
        'aggregation': [],
        'basic': []
    }

    for tpl in matching_templates:
        features = tpl.get('features_set', set())
        if 'cte' in features:
            feature_groups['cte'].append(tpl)
        elif 'window_functions' in features:
            feature_groups['window_functions'].append(tpl)
        elif 'subqueries' in features:
            feature_groups['subqueries'].append(tpl)
        elif 'aggregation' in features:
            feature_groups['aggregation'].append(tpl)
        else:
            feature_groups['basic'].append(tpl)

    # 按优先级选择k个不同的模板
    priority_order = ['cte', 'window_functions', 'subqueries', 'aggregation', 'basic']

    selected = []
    for feature_type in priority_order:
        templates_in_group = feature_groups[feature_type]
        if templates_in_group and len(selected) < k:
            # 从这个组中选择，基于combination_index轮换
            idx = (combination_index + len(selected)) % len(templates_in_group)
            selected.append(templates_in_group[idx])

    # 如果仍然不够k个，从matching_templates中继续选
    if len(selected) < k and len(matching_templates) > len(selected):
        for i in range(k - len(selected)):
            idx = (combination_index + len(selected) + i) % len(matching_templates)
            candidate = matching_templates[idx]
            if candidate not in selected:
                selected.append(candidate)

    return selected[:k]


# ================== LLM Prompt构建 ==================

def build_combination_prompt(
    combination: Dict,
    table_data: Dict,
    template: Dict,
    table_samples: Dict[str, Dict]
) -> str:
    """为combination构建LLM prompt"""

    table_name = table_data['table_name']
    user_selected_nodes = combination['user_selected_nodes']  # 可以选列的节点
    nodes_in_join = combination['nodes_in_join']  # 参与JOIN的所有节点
    leaf_nodes = combination['leaf_nodes']
    available_columns = combination['available_columns']

    # 找出只用于JOIN的中间节点
    intermediate_nodes = set(nodes_in_join) - set(user_selected_nodes)

    # 找出user_selected_nodes之间的直接连接边及其JOIN列
    edges = parse_pattern(table_data['pattern'])
    edge_join_columns = table_data.get('edge_join_columns', {})

    required_join_cols_in_select = []
    for parent, child in edges:
        # 如果这条边的两端都在user_selected_nodes中，则它们的JOIN列必须被选中
        if parent in user_selected_nodes and child in user_selected_nodes:
            edge_key = f"{parent}-{child}"
            join_cols = edge_join_columns.get(edge_key, [])
            for col in join_cols:
                required_join_cols_in_select.append(f"t{parent}.{col}")
                required_join_cols_in_select.append(f"t{child}.{col}")

    # 去重
    required_join_cols_in_select = list(set(required_join_cols_in_select))

    prompt = f"""You are a SQL query generator. Your task is to fill placeholders in a SQL template by selecting columns from SPECIFIC ALLOWED nodes.

# Database Context

**Table**: {table_name}
**Nodes in JOIN**: {', '.join([f't{n}' for n in nodes_in_join])} (all will appear in FROM/JOIN clause)
**🔴 NODES YOU CAN SELECT COLUMNS FROM**: {', '.join([f't{n}' for n in user_selected_nodes])} (ONLY these!)
**🔴 INTERMEDIATE NODES (JOIN ONLY)**: {', '.join([f't{n}' for n in intermediate_nodes]) if intermediate_nodes else 'None'} (DO NOT select columns from these!)
**Leaf Nodes**: {', '.join([f't{n}' for n in leaf_nodes])} (MUST select at least one non-JOIN column from each)
**🔴 REQUIRED JOIN COLUMNS TO SELECT**: {', '.join(required_join_cols_in_select) if required_join_cols_in_select else 'None'} (MUST include these in SELECT!)

## Node Information and Column Constraints
"""

    # 添加每个节点的信息，区分可选列节点和仅JOIN节点
    for node_id in nodes_in_join:
        node_info = available_columns[node_id]
        is_leaf = node_info['is_leaf']
        can_select = node_info['can_select_columns']

        if can_select:
            prompt += f"\n**Node t{node_id}** ✓ CAN SELECT COLUMNS"
            if is_leaf:
                prompt += " (LEAF - MUST select columns)"
        else:
            prompt += f"\n**Node t{node_id}** ✗ INTERMEDIATE - DO NOT SELECT COLUMNS (JOIN only)"

        if node_id in table_samples and can_select:
            sample = table_samples[node_id]
            sample_data = sample['sample_data']

            # 分析列类型
            numeric_cols = []
            text_cols = []

            if sample_data['rows']:
                header_idx = {col: i for i, col in enumerate(sample_data['header'])}
                for col in node_info['non_join_columns'][:15]:
                    if col in header_idx:
                        col_idx = header_idx[col]
                        is_numeric = True
                        for row in sample_data['rows'][:2]:
                            if col_idx < len(row):
                                val = row[col_idx]
                                if val and val.strip():
                                    try:
                                        float(val.replace(',', ''))
                                    except (ValueError, AttributeError):
                                        is_numeric = False
                                        break

                        if is_numeric:
                            numeric_cols.append(col)
                        else:
                            text_cols.append(col)

            # 标记高频列
            high_freq = set(node_info['high_freq_non_join_columns'])
            numeric_high_freq = [f"{col} (HIGH-FREQ)" if col in high_freq else col for col in numeric_cols[:8]]
            text_high_freq = [f"{col} (HIGH-FREQ)" if col in high_freq else col for col in text_cols[:8]]

            prompt += f"\n- NUMERIC non-JOIN columns: {', '.join(numeric_high_freq) if numeric_high_freq else 'None'}\n"
            prompt += f"- TEXT non-JOIN columns: {', '.join(text_high_freq) if text_high_freq else 'All'}\n"
            if is_leaf or high_freq:
                prompt += f"- 🔴 At least ONE HIGH-FREQ column must be selected from this node\n"

            if sample_data['header']:
                prompt += f"- Sample data (first 2 rows):\n"
                header = sample_data['header'][:5]
                prompt += f"  Header: {' | '.join(header)}\n"
                for i, row in enumerate(sample_data['rows'][:2]):
                    prompt += f"  Row {i+1}: {' | '.join(row[:5])}\n"
        elif not can_select:
            prompt += "\n- (Intermediate node - used for JOIN only, no columns needed)\n"

    # 添加约束
    prompt += f"""
# CRITICAL CONSTRAINTS (MUST FOLLOW)

1. **🔴 MOST IMPORTANT - Node Selection**: ONLY select columns from nodes {', '.join([f't{n}' for n in user_selected_nodes])}
   - DO NOT select columns from intermediate nodes {', '.join([f't{n}' for n in intermediate_nodes]) if intermediate_nodes else '(none)'}
   - These intermediate nodes appear in JOIN clause but you CANNOT select their columns
2. **🔴 REQUIRED JOIN COLUMNS**: MUST include these JOIN columns in SELECT: {', '.join(required_join_cols_in_select) if required_join_cols_in_select else 'None'}
   - These are columns that connect nodes directly in your selected combination
   - They MUST appear in the final SELECT statement
3. **Leaf Node Requirement**: MUST select at least one non-JOIN column from each leaf node ({', '.join([f't{n}' for n in leaf_nodes])})
4. **High-Frequency Columns**: Each selected node must have at least ONE HIGH-FREQ column selected
5. **Column Count**: Final query must have EXACTLY 6-9 columns total (including required JOIN columns)
6. **Data Type Matching**:
   - SUM/AVG/MAX/MIN: ONLY use on NUMERIC columns
   - COALESCE: All arguments must have SAME type
   - CASE WHEN: Compare values of SAME type
   - COUNT(*) or COUNT(column) works on any type
7. **Semantic Coherence**: Selected columns must form a meaningful query answering a real business question

# Template to Fill

**Template Name**: {template['name']}
**SQL Features**: {', '.join(template.get('features_set', set()))}

**Pattern with JOIN already filled**:
```sql
{template['pattern'].replace('{{join_clause}}', combination['join_clause'])}
```

**Placeholders to fill**: {', '.join([p for p in template.get('placeholders', {}).keys() if 'table' not in p.lower()])}

⚠️ PLACEHOLDER FILLING RULES:
- When placeholder requires "at least N columns", provide that many columns separated by commas
- Use format: t1.column_name, t2.column_name (always include table alias)
- 🔴 Remember: ONLY use nodes {', '.join([f't{n}' for n in user_selected_nodes])} for columns!
- FINAL SELECT must have 6-9 columns total

# CTE Column Reference Rules (if template uses WITH)

🔴 CRITICAL - CTE (WITH clause) COLUMN REFERENCES:
- INSIDE the CTE (WITH ... AS (SELECT ...)): Use table aliases like t1.column_name, t2.column_name
- OUTSIDE the CTE (final SELECT ... FROM cte_name): Do NOT use table aliases, only column names
- Example CORRECT:
  WITH base_data AS (SELECT t1.user_id, t1.name, t2.amount FROM ... WHERE t1.status = 'active')
  SELECT user_id, name, SUM(amount) FROM base_data GROUP BY user_id, name

⚠️ DO NOT MODIFY TEMPLATE STRUCTURE:
- DO NOT remove or change SQL keywords (WITH, SELECT, FROM, WHERE, GROUP BY, etc.)
- DO NOT change template structure - if it has WITH clause, output MUST have WITH clause
- ONLY replace {{{{placeholder}}}} values with actual column names

# Output Format

Return ONLY a JSON object:
{{
  "filled_sql": "<complete SQL with all placeholders filled>",
  "selected_columns": ["t1.col1", "t2.col2", ...],
  "reasoning": "<brief explanation of why these columns form a meaningful query>"
}}

🔴 FINAL CHECKLIST:
- [ ] Only selected columns from allowed nodes: {', '.join([f't{n}' for n in user_selected_nodes])}
- [ ] Did NOT select columns from intermediate nodes: {', '.join([f't{n}' for n in intermediate_nodes]) if intermediate_nodes else '(none)'}
- [ ] Selected at least one non-JOIN column from each leaf node
- [ ] Each allowed node has at least one HIGH-FREQ column
- [ ] Final SELECT has 6-9 columns total
- [ ] All data types match (no SUM on text, no mixed-type COALESCE, etc.)
- [ ] Query is semantically meaningful
- [ ] Template structure preserved (WITH clause if present, etc.)
"""

    return prompt


# ================== LLM调用 ==================

def call_llm_single(prompt: str) -> Dict:
    """调用LLM填充模板"""
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {LLM_API_KEY}"
    }

    payload = {
        "model": LLM_MODEL,
        "messages": [
            {"role": "system", "content": "You are an expert SQL query generator."},
            {"role": "user", "content": prompt}
        ],
        "temperature": 0.7,
        "max_tokens": 1500
    }

    try:
        response = requests.post(LLM_API_ENDPOINT, headers=headers, json=payload, timeout=60)
        response.raise_for_status()

        result = response.json()
        content = result['choices'][0]['message']['content']

        # 尝试解析JSON
        json_match = re.search(r'```json\s*(.*?)\s*```', content, re.DOTALL)
        if json_match:
            content = json_match.group(1)

        parsed = json.loads(content)
        return parsed

    except json.JSONDecodeError as e:
        print(f"  ✗ JSON解析失败: {e}")
        return None
    except Exception as e:
        print(f"  ✗ LLM调用失败: {e}")
        return None


# ================== 主处理流程 ==================

def process_single_table(table_data: Dict, all_templates: List[Dict]) -> Dict:
    """处理单个表：生成所有combination并填充SQL"""
    table_name = table_data['table_name']

    print(f"\n{'='*80}")
    print(f"处理表: {table_name[:60]}")

    # 生成所有combination
    print(f"  生成所有节点combination...")
    combinations = generate_all_combinations(table_data)
    print(f"  ✓ 生成了 {len(combinations)} 个combination")

    all_results = []

    for comb_idx, combination in enumerate(combinations):
        print(f"\n  --- Combination {comb_idx+1}/{len(combinations)} ---")
        print(f"      选中节点: {', '.join([f't{n}' for n in combination['user_selected_nodes']])}")
        print(f"      JOIN数: {combination['num_joins']}")

        # 为这个combination选择k个模板
        templates = select_templates_for_combination(
            all_templates,
            combination['num_joins'],
            comb_idx
        )

        if not templates:
            print(f"      ⚠️ 没有匹配的模板")
            continue

        print(f"      选择了 {len(templates)} 个模板: {', '.join([t['name'] for t in templates])}")

        # 准备样本数据
        table_samples = prepare_table_samples(table_data, combination['user_selected_nodes'])

        # 对每个模板调用LLM
        for template_idx, template in enumerate(templates, 1):
            print(f"      [{template_idx}/{len(templates)}] 填充模板: {template['name']}")

            # 构建prompt并调用LLM
            prompt = build_combination_prompt(combination, table_data, template, table_samples)
            result = call_llm_single(prompt)

            if result:
                result['combination_index'] = comb_idx
                result['template_index'] = template_idx - 1
                result['user_selected_nodes'] = combination['user_selected_nodes']
                result['nodes_in_join'] = combination['nodes_in_join']
                result['leaf_nodes'] = combination['leaf_nodes']
                result['template_name'] = template['name']
                result['template_features'] = list(template.get('features_set', set()))
                all_results.append(result)
                print(f"          ✓ 成功生成SQL（{len(result.get('selected_columns', []))} 列）")
            else:
                print(f"          ✗ 失败")

            # 避免API限流
            time.sleep(0.5)

    print(f"\n  ✓ 成功生成 {len(all_results)}/{len(combinations)} 个查询")

    return {
        'table_name': table_name,
        'num_combinations': len(combinations),
        'queries': all_results,
        'success_rate': f"{len(all_results)}/{len(combinations)}"
    }


# ================== 主函数 ==================

def main():
    """主函数"""
    print("=" * 80)
    print("使用Combination + LLM生成SQL查询")
    print(f"数据源: {DATA_DIR}")
    print(f"模板文件: {TEMPLATE_FILE}")
    print("=" * 80)

    # 加载数据
    print("\n加载数据...")
    with open(STATISTICS_FILE, 'r') as f:
        stats = json.load(f)

    with open(TEMPLATE_FILE, 'r') as f:
        all_templates = json.load(f)

    print(f"✓ 加载了 {len(all_templates)} 个模板")

    # 只处理normal类别的表
    normal_tables = stats.get('normal', [])
    print(f"✓ 加载了 {len(normal_tables)} 个normal表")

    # 处理所有表（或前几个测试）
    results = []
    tables_to_process = normal_tables if MAX_TABLES is None else normal_tables[:MAX_TABLES]

    # 只处理有至少2个JOIN的表
    processed_count = 0
    for i, table_data in enumerate(tables_to_process):
        edges = parse_pattern(table_data['pattern'])
        all_nodes = identify_all_nodes(edges)
        leaf_nodes = identify_leaf_nodes(edges)

        num_joins = len(edges)

        if num_joins >= 2:
            print(f"\n{'='*80}")
            print(f"处理表 {processed_count + 1} (进度: {i+1}/{len(tables_to_process)})")
            print(f"节点数: {len(all_nodes)}, 叶子节点数: {len(leaf_nodes)}, JOIN数: {num_joins}")
            print(f"理论combination数: 2^{len(all_nodes) - len(leaf_nodes)} = {2**(len(all_nodes) - len(leaf_nodes))}")

            result = process_single_table(table_data, all_templates)
            if result:
                results.append(result)
                processed_count += 1

            # 避免API限流
            time.sleep(1)

    # 保存结果
    with open(OUTPUT_FILE, 'w') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    print(f"\n{'='*80}")
    print(f"生成完成！")
    print(f"{'='*80}")
    print(f"成功处理了 {len(results)} 个表")

    # 统计总query数
    total_queries = sum(len(r['queries']) for r in results)
    print(f"总共生成了 {total_queries} 个查询")
    print(f"输出文件: {OUTPUT_FILE}")
    print(f"{'='*80}")

    # ========== 自动执行查询并过滤0行结果 ==========
    print(f"\n{'='*80}")
    print("开始执行查询并过滤0行结果...")
    print(f"{'='*80}")

    import duckdb
    conn = duckdb.connect(':memory:')

    queries_to_remove = []  # 记录需要删除的(table_name, comb_idx, template_idx)

    for table_idx, table_data in enumerate(results, 1):
        table_name = table_data['table_name']
        queries = table_data['queries']

        print(f"\n[{table_idx}/{len(results)}] 执行表: {table_name[:50]}")

        # 获取表信息
        table_info = None
        for t in stats.get('normal', []):
            if t['table_name'] == table_name:
                table_info = t
                break

        if not table_info:
            print(f"  ✗ 找不到表信息，跳过")
            continue

        # 注册所有子表到DuckDB
        category = table_info.get('category', 'normal')
        node_path = f"{DATA_DIR}/{table_name}_{category}"
        node_mapping = table_info.get('node_mapping', {})

        for node_id, node_info in node_mapping.items():
            subtable_file = node_info['subtable_file']
            csv_path = os.path.join(node_path, f"{subtable_file}.csv")
            if os.path.exists(csv_path):
                view_name = f'"{subtable_file}"'
                conn.execute(f"CREATE OR REPLACE VIEW {view_name} AS SELECT * FROM read_csv_auto('{csv_path}', header=true, ignore_errors=true)")

        # 执行每个查询
        for query_idx, query_info in enumerate(queries):
            comb_idx = query_info.get('combination_index', -1)
            template_idx = query_info.get('template_index', 0)
            template_name = query_info.get('template_name', 'unknown')

            try:
                sql_for_duckdb = query_info['filled_sql'].replace('`', '"')
                result = conn.execute(sql_for_duckdb).fetchall()
                row_count = len(result)

                if row_count == 0:
                    queries_to_remove.append((table_name, comb_idx, template_idx))
                    print(f"  [{query_idx+1}/{len(queries)}] Comb{comb_idx}-T{template_idx} {template_name[:30]} - 0行 ✗")
                else:
                    print(f"  [{query_idx+1}/{len(queries)}] Comb{comb_idx}-T{template_idx} {template_name[:30]} - {row_count}行 ✓")

            except Exception as e:
                queries_to_remove.append((table_name, comb_idx, template_idx))
                print(f"  [{query_idx+1}/{len(queries)}] Comb{comb_idx}-T{template_idx} {template_name[:30]} - 错误 ✗: {str(e)[:50]}")

    # 过滤掉0行和错误的查询
    print(f"\n{'='*80}")
    print(f"过滤0行和错误查询...")
    print(f"需要删除: {len(queries_to_remove)} 个查询")

    for table in results:
        original_count = len(table['queries'])
        table['queries'] = [
            q for q in table['queries']
            if (table['table_name'], q.get('combination_index'), q.get('template_index', 0)) not in queries_to_remove
        ]
        removed = original_count - len(table['queries'])
        if removed > 0:
            print(f"  {table['table_name'][:50]}: 删除 {removed} 个查询")

    # 保存过滤后的结果
    final_output = OUTPUT_FILE.replace('.json', '_filtered.json')
    with open(final_output, 'w') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    final_query_count = sum(len(r['queries']) for r in results)
    print(f"\n{'='*80}")
    print(f"过滤完成！")
    print(f"{'='*80}")
    print(f"原始查询数: {total_queries}")
    print(f"删除查询数: {len(queries_to_remove)}")
    print(f"最终查询数: {final_query_count}")
    print(f"最终输出文件: {final_output}")
    print(f"{'='*80}")


if __name__ == '__main__':
    main()
