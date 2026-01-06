#!/usr/bin/env python3
"""
重新执行SQL_queries.json中的所有SQL查询
保存结果为CSV文件，并过滤掉0行的结果
生成SQL-执行结果的映射JSON文件
"""

import json
import os
import duckdb
from typing import Dict, Any, List
import time
import pandas as pd

# 路径配置
SQL_QUERIES_FILE = '/data2/liujinqi/Revision/SQL_generation/SQL_queries.json'
SPLIT_TABLES_DIR = '/data2/liujinqi/Revision/SQL_generation/splited_table-250/dataset'
RESULTS_DIR = '/data2/liujinqi/Revision/SQL_generation/query_results_all'
OUTPUT_QUERIES_FILE = '/data2/liujinqi/Revision/SQL_generation/SQL_queries_with_results.json'
SQL_RESULT_MAPPING_FILE = '/data2/liujinqi/Revision/SQL_generation/SQL_result_mapping.json'

# 创建输出目录
os.makedirs(RESULTS_DIR, exist_ok=True)


def convert_to_serializable(obj):
    """将pandas对象转换为JSON可序列化的格式"""
    import numpy as np
    from datetime import time, date, datetime, timedelta

    # 处理字典
    if isinstance(obj, dict):
        return {k: convert_to_serializable(v) for k, v in obj.items()}
    # 处理列表
    elif isinstance(obj, list):
        return [convert_to_serializable(item) for item in obj]
    # 处理pandas/numpy类型
    elif isinstance(obj, (pd.Timestamp, pd.Timedelta)):
        return obj.isoformat()
    elif isinstance(obj, (np.integer, np.floating)):
        return obj.item()
    elif isinstance(obj, (pd.Series, pd.DataFrame)):
        return obj.to_dict()
    # 处理Python datetime类型
    elif isinstance(obj, (datetime, date)):
        return obj.isoformat()
    elif isinstance(obj, time):
        return obj.isoformat()
    elif isinstance(obj, timedelta):
        return str(obj)
    # 处理NaN/NaT
    elif pd.isna(obj):
        return None
    else:
        return obj


def get_actual_table_dir(table_name: str) -> str:
    """
    获取实际的表文件夹路径
    只使用带_normal后缀的文件夹

    Args:
        table_name: 表名（不带后缀）

    Returns:
        实际的文件夹路径
    """
    # 只使用_normal后缀的文件夹
    normal_dir = os.path.join(SPLIT_TABLES_DIR, f"{table_name}_normal")
    return normal_dir


def execute_sql_query(
    conn: duckdb.DuckDBPyConnection,
    sql: str,
    table_name: str,
    nodes_in_join: List[str]
) -> tuple:
    """
    执行SQL查询并返回结果DataFrame

    Returns:
        (success: bool, result_df or error_message, row_count)
    """
    import re
    created_views = []  # 记录创建的视图名

    try:
        # 获取实际的表文件夹路径（带_normal后缀）
        actual_table_dir = get_actual_table_dir(table_name)

        # 从SQL中提取表名到别名的映射
        # 匹配模式：FROM/JOIN `表名` t节点ID
        pattern = r'(?:FROM|JOIN)\s+`([^`]+)`\s+t(\d+)'
        matches = re.findall(pattern, sql, re.IGNORECASE)

        # 创建节点ID到实际表名的映射
        node_to_table_map = {}
        for table_ref, node_id in matches:
            node_to_table_map[node_id] = table_ref

        # 为join中的每个节点创建视图
        for node_id in nodes_in_join:
            # 从SQL中提取的实际表名
            if node_id in node_to_table_map:
                subtable_file = node_to_table_map[node_id]
                csv_path = os.path.join(actual_table_dir, f"{subtable_file}.csv")
            else:
                # 如果SQL中没有找到，回退到原来的逻辑
                csv_path_node = os.path.join(actual_table_dir, f"node_{node_id}.csv")
                csv_path_table = os.path.join(actual_table_dir, f"{table_name}_{node_id}.csv")

                if os.path.exists(csv_path_node):
                    csv_path = csv_path_node
                    subtable_file = f"node_{node_id}"
                elif os.path.exists(csv_path_table):
                    csv_path = csv_path_table
                    subtable_file = f"{table_name}_{node_id}"
                else:
                    return False, f"CSV file not found: tried {csv_path_node} and {csv_path_table}", 0

            if not os.path.exists(csv_path):
                return False, f"CSV file not found: {csv_path}", 0

            # 使用双引号包裹表名以支持特殊字符
            view_name = f'"{subtable_file}"'
            # 转义CSV路径中的单引号（SQL中单引号需要用两个单引号转义）
            csv_path_escaped = csv_path.replace("'", "''")
            conn.execute(f"DROP VIEW IF EXISTS {view_name}")
            conn.execute(f"CREATE VIEW {view_name} AS SELECT * FROM read_csv_auto('{csv_path_escaped}', header=true, ignore_errors=true)")
            created_views.append(view_name)

        # 将SQL中的反引号替换为双引号（DuckDB语法）
        sql_fixed = sql.replace('`', '"')

        # 执行查询
        result = conn.execute(sql_fixed).fetchdf()
        row_count = len(result)

        # 清理视图
        for view_name in created_views:
            conn.execute(f"DROP VIEW IF EXISTS {view_name}")

        return True, result, row_count

    except Exception as e:
        # 清理视图
        try:
            for view_name in created_views:
                conn.execute(f"DROP VIEW IF EXISTS {view_name}")
        except:
            pass

        return False, str(e), 0


def main():
    """主函数"""
    print("=" * 80)
    print("重新执行所有SQL查询并保存结果")
    print("=" * 80)

    # 读取SQL queries
    print("\n加载SQL查询...")
    with open(SQL_QUERIES_FILE, 'r') as f:
        sql_queries = json.load(f)

    total_tables = len(sql_queries)
    total_queries = sum(len(table['queries']) for table in sql_queries)
    print(f"总表数: {total_tables}")
    print(f"总查询数: {total_queries}")

    # 初始化DuckDB连接
    print("\n初始化DuckDB...")
    conn = duckdb.connect(':memory:')
    conn.execute("SET memory_limit='8GB'")

    # 处理所有查询
    print("\n开始执行查询...\n")

    all_results = []
    sql_result_mapping = {}  # SQL查询到结果的映射
    successful = 0
    failed = 0
    empty_results = 0

    for table_idx, table_data in enumerate(sql_queries, 1):
        # 使用不带后缀的表名进行处理和匹配
        table_name = table_data['table_name'].replace('_normal', '')
        queries = table_data['queries']

        print(f"[{table_idx}/{total_tables}] {table_name} ({len(queries)} queries)")

        for query_idx, query_info in enumerate(queries):
            comb_idx = query_info.get('combination_index', 0)
            template_idx = query_info.get('template_index', 0)
            sql_query = query_info['filled_sql']

            # 执行查询（函数内部会自动寻找带_normal后缀的文件夹）
            success, result, row_count = execute_sql_query(
                conn,
                sql_query,
                table_name,
                query_info['nodes_in_join']
            )

            # 创建映射键（使用不带后缀的表名）
            mapping_key = f"{table_name}_comb{comb_idx}_t{template_idx}_q{query_idx}"

            if not success:
                print(f"  ✗ Query {query_idx} (comb{comb_idx}_t{template_idx}): 执行失败 - {result}")
                # 记录失败的查询
                sql_result_mapping[mapping_key] = {
                    'sql': sql_query,
                    'status': 'failed',
                    'error': result,
                    'table_name': table_name,
                    'combination_index': comb_idx,
                    'template_index': template_idx,
                    'query_index': query_idx
                }
                failed += 1
                continue

            if row_count == 0:
                print(f"  ○ Query {query_idx} (comb{comb_idx}_t{template_idx}): 0行 (跳过)")
                # 记录空结果的查询
                sql_result_mapping[mapping_key] = {
                    'sql': sql_query,
                    'status': 'empty',
                    'row_count': 0,
                    'table_name': table_name,
                    'combination_index': comb_idx,
                    'template_index': template_idx,
                    'query_index': query_idx
                }
                empty_results += 1
                continue

            # 保存CSV文件（使用不带后缀的表名）
            csv_filename = f"{table_name}_comb{comb_idx}_t{template_idx}_q{query_idx}.csv"
            csv_path = os.path.join(RESULTS_DIR, csv_filename)
            result.to_csv(csv_path, index=False)

            # 获取结果的列名和前几行数据作为示例
            result_columns = result.columns.tolist()
            result_sample_raw = result.head(5).to_dict('records') if row_count > 0 else []
            # 转换为JSON可序列化格式
            result_sample = convert_to_serializable(result_sample_raw)

            # 记录成功的查询到映射
            sql_result_mapping[mapping_key] = {
                'sql': sql_query,
                'status': 'success',
                'table_name': table_name,
                'combination_index': comb_idx,
                'template_index': template_idx,
                'query_index': query_idx,
                'template_name': query_info.get('template_name', ''),
                'selected_columns': query_info['selected_columns'],
                'user_selected_nodes': query_info.get('user_selected_nodes', []),
                'nodes_in_join': query_info.get('nodes_in_join', []),
                'result': {
                    'csv_file': csv_filename,
                    'csv_path': csv_path,
                    'row_count': row_count,
                    'column_count': len(result_columns),
                    'columns': result_columns,
                    'sample_data': result_sample  # 前5行数据作为示例
                }
            }

            # 记录查询信息（保持原有格式，使用不带后缀的表名）
            query_result = {
                'table_name': table_name,
                'combination_index': comb_idx,
                'template_index': template_idx,
                'query_index_in_json': query_idx,
                'template_name': query_info.get('template_name', ''),
                'sql': sql_query,
                'selected_columns': query_info['selected_columns'],
                'csv_filename': csv_filename,
                'row_count': row_count,
                'user_selected_nodes': query_info.get('user_selected_nodes', []),
                'nodes_in_join': query_info.get('nodes_in_join', [])
            }
            all_results.append(query_result)

            print(f"  ✓ Query {query_idx} (comb{comb_idx}_t{template_idx}): {row_count}行 -> {csv_filename}")
            successful += 1

    # 保存有结果的查询列表
    print(f"\n{'='*80}")
    print("保存查询列表...")
    with open(OUTPUT_QUERIES_FILE, 'w') as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False)

    print(f"✓ 有效查询列表已保存到: {OUTPUT_QUERIES_FILE}")

    # 保存SQL-结果映射
    print("保存SQL-结果映射...")
    with open(SQL_RESULT_MAPPING_FILE, 'w') as f:
        json.dump(sql_result_mapping, f, indent=2, ensure_ascii=False)

    print(f"✓ SQL-结果映射已保存到: {SQL_RESULT_MAPPING_FILE}")
    print(f"✓ CSV结果已保存到: {RESULTS_DIR}/")

    # 打印统计
    print(f"\n{'='*80}")
    print("执行统计:")
    print(f"{'='*80}")
    print(f"总查询数: {total_queries}")
    print(f"成功: {successful} 个查询 (有结果)")
    print(f"空结果: {empty_results} 个查询 (0行)")
    print(f"失败: {failed} 个查询")
    print(f"有效率: {successful/total_queries*100:.1f}%")
    print(f"\n输出文件:")
    print(f"  查询列表: {OUTPUT_QUERIES_FILE}")
    print(f"  SQL映射: {SQL_RESULT_MAPPING_FILE}")
    print(f"  CSV结果: {RESULTS_DIR}/")
    print(f"{'='*80}")


if __name__ == '__main__':
    main()