#!/usr/bin/env python3
"""
测试：将SQL转换为自然语言查询（NL Query）
处理前5个查询进行测试
"""

import json
import os
import pandas as pd
from openai import OpenAI
import time
from typing import Dict, Any

# API配置 - 使用dmxapi
client = OpenAI(
    api_key="sk-VTOqNlWBMvz6Hg5CK6uwUhctIRGpA5TZ0eGJ0KaVIyLtZwTD",  # 请在此填入你的API Key
    base_url="https://www.dmxapi.com/v1"
)
MODEL = "gpt-5.1"

# 路径配置
SQL_QUERIES_FILE = '/data2/liujinqi/Revision/SQL_generation/SQL_queries.json'
RESULTS_DIR = '/data2/liujinqi/Revision/SQL_generation/combination_query_results'
STATISTICS_FILE = '/data2/liujinqi/Revision/SQL_generation/splited_table-250/dataset/split_statistics.json'
OUTPUT_FILE = '/data2/liujinqi/Revision/SQL_generation/nl_queries_all.json'

# 测试参数
TEST_NUM_QUERIES = None  # None表示处理全部查询


def sample_result_data(csv_path: str, num_rows: int = 3) -> str:
    """从CSV结果中采样数据"""
    try:
        df = pd.read_csv(csv_path)

        if len(df) == 0:
            return "Empty result set"

        # 采样前几行
        sample_size = min(num_rows, len(df))
        sample_df = df.head(sample_size)

        # 格式化为字符串
        result = f"Result has {len(df)} rows and {len(df.columns)} columns.\n"
        result += f"Columns: {', '.join(df.columns)}\n"
        result += f"Sample data (first {sample_size} rows):\n"
        result += sample_df.to_string(index=False, max_colwidth=50)

        return result
    except Exception as e:
        return f"Error reading result: {e}"


def sql_to_nl_query(
    sql: str,
    selected_columns: list,
    table_schema: Dict[str, Any]
) -> str:
    """使用LLM将SQL转换为自然语言查询"""

    # 简化列名为通俗英语
    simplified_columns = []
    for col in selected_columns:
        # 去掉表前缀 (如 t1.column -> column)
        if '.' in col:
            col = col.split('.')[1]
        simplified_columns.append(col)

    columns_list = ', '.join(simplified_columns)

    # 分析SQL特征
    sql_upper = sql.upper()
    has_where = 'WHERE' in sql_upper
    has_group_by = 'GROUP BY' in sql_upper
    has_order_by = 'ORDER BY' in sql_upper
    has_limit = 'LIMIT' in sql_upper
    has_aggregation = any(func in sql_upper for func in ['SUM(', 'AVG(', 'COUNT(', 'MAX(', 'MIN('])

    # 构建SQL特征描述
    sql_features = []
    if has_aggregation:
        sql_features.append("with aggregation (sum/avg/count/max/min)")
    if has_where:
        sql_features.append("with filtering conditions")
    if has_group_by:
        sql_features.append("grouped by certain columns")
    if has_order_by:
        sql_features.append("sorted")
    if has_limit:
        sql_features.append("limited results")

    features_desc = ", ".join(sql_features) if sql_features else "simple selection"

    # 简化SQL用于显示（去掉反引号和双引号）
    sql_display = sql.replace('`', '').replace('"', '')

    prompt = f"""Given the SQL query and columns below, write a natural English query.
- Cover all selected columns: {columns_list} but you cannot mention them directly
- Sound like a real person asking for data, not a machine, keep it conversational
- Vary question types, connectors, and sentence structures
- Simplify column names to plain English, if you don't know the meaning, make a reasonable guess based on context
- Avoid mentioning data sources (e.g., "dataset", "file", "second source") - focus on what data is needed, not where it's from

Return ONLY the question.


SQL Query:
{sql_display}

Columns: {columns_list}

SQL features: {features_desc}

"""

    try:
        response = client.chat.completions.create(
            model=MODEL,
            messages=[
                {"role": "system", "content": "You are an expert at writing natural, specific user questions."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.9,
            max_tokens=1000  # 增加到300，确保复杂查询不会被截断
        )

        nl_query = response.choices[0].message.content.strip()
        nl_query = nl_query.strip('"\'')

        return nl_query

    except Exception as e:
        print(f"  Error generating NL query: {e}")
        return f"Error: {e}"


def process_single_query(
    table_name: str,
    query_info: Dict[str, Any],
    query_index: int,
    table_schema: Dict[str, Any]
) -> Dict[str, Any]:
    """处理单个查询，生成NL query"""

    comb_idx = query_info.get('combination_index', 0)
    template_idx = query_info.get('template_index', 0)

    print(f"\n  Query {query_index}: comb{comb_idx}_template{template_idx}")

    # 生成NL query（不需要CSV文件）
    nl_query = sql_to_nl_query(
        query_info['filled_sql'],
        query_info['selected_columns'],
        table_schema
    )

    print(f"    Selected columns: {query_info['selected_columns'][:3]}...")
    print(f"    NL Query: {nl_query}")

    time.sleep(0.3)  # 避免API限流

    # 构建CSV文件名（可能存在也可能不存在）
    csv_filename = f"{table_name}_comb{comb_idx}_q{template_idx}.csv"

    return {
        'table_name': table_name,
        'combination_index': comb_idx,
        'template_index': template_idx,
        'query_index_in_json': query_index,
        'template_name': query_info.get('template_name', ''),
        'sql': query_info['filled_sql'],
        'selected_columns': query_info['selected_columns'],
        'nl_query': nl_query,
        'csv_filename': csv_filename,
        'user_selected_nodes': query_info.get('user_selected_nodes', []),
        'nodes_in_join': query_info.get('nodes_in_join', [])
    }


def main():
    """主函数"""
    print("=" * 80)
    if TEST_NUM_QUERIES is None:
        print(f"SQL转自然语言查询（处理全部查询）")
    else:
        print(f"SQL转自然语言查询（处理前{TEST_NUM_QUERIES}个查询）")
    print("=" * 80)
    print(f"\nAPI: dmxapi ({MODEL})")

    # 读取数据
    print("\n加载数据...")
    with open(SQL_QUERIES_FILE, 'r') as f:
        sql_queries = json.load(f)

    with open(STATISTICS_FILE, 'r') as f:
        statistics = json.load(f)

    # 创建表名到schema的映射
    table_schema_map = {}
    for table_info in statistics.get('normal', []):
        table_schema_map[table_info['table_name']] = table_info

    # 收集所有查询（不需要CSV文件）
    all_queries = []
    for table_data in sql_queries:
        table_name = table_data['table_name']

        # 获取该表的schema
        if table_name not in table_schema_map:
            print(f"⚠️  警告：找不到表 {table_name} 的schema信息")
            continue

        table_schema = table_schema_map[table_name]

        for query_idx, query_info in enumerate(table_data['queries']):
            all_queries.append((table_name, query_idx, query_info, table_schema))

    total_queries = len(all_queries)
    queries_to_process = all_queries if TEST_NUM_QUERIES is None else all_queries[:TEST_NUM_QUERIES]
    num_to_process = len(queries_to_process)

    print(f"总查询数（SQL_queries.json）: {total_queries}")
    if TEST_NUM_QUERIES is None:
        print(f"处理全部 {num_to_process} 个查询\n")
    else:
        print(f"处理前 {num_to_process} 个查询\n")

    # 处理查询
    all_results = []
    successful = 0
    failed = 0

    for i, (table_name, query_idx, query_info, table_schema) in enumerate(queries_to_process):
        print(f"\n{'='*80}")
        print(f"[{i+1}/{num_to_process}] 表: {table_name}")
        print(f"{'='*80}")

        try:
            result = process_single_query(table_name, query_info, query_idx, table_schema)
            if result:
                all_results.append(result)
                successful += 1
        except Exception as e:
            print(f"    错误: {e}")
            import traceback
            traceback.print_exc()
            failed += 1
            continue

    # 保存结果
    print(f"\n{'='*80}")
    print("保存测试结果...")

    with open(OUTPUT_FILE, 'w') as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False)

    print(f"✓ 测试NL查询已保存到: {OUTPUT_FILE}")

    # 打印统计
    print(f"\n{'='*80}")
    print("测试统计:")
    print(f"{'='*80}")
    print(f"成功: {successful} 个查询")
    print(f"失败: {failed} 个查询")
    if successful + failed > 0:
        print(f"成功率: {successful/(successful+failed)*100:.1f}%")
    print(f"\n输出文件: {OUTPUT_FILE}")
    print(f"{'='*80}")


if __name__ == '__main__':
    main()
