import os
import json
from openai import OpenAI
import pandas as pd
import re
from pathlib import Path
import shutil
import sqlparse
from sqlparse.sql import IdentifierList, Identifier
from sqlparse.tokens import Keyword, DML
import argparse
from typing import List

def parse_sql_columns(sql_content, api_key, model_name, output_dir):
    """使用LLM解析SQL文件中的列定义和数据血缘"""
    # 构建提示词
    prompt = f"""
Please perform a comprehensive column-level data lineage analysis for the following SQL query. I need a complete mapping showing how EVERY column in the target table '{join_nodes[0]}' is derived from the source tables and columns.

Known relevant tables (blue nodes): {', '.join(blue_tables)}

SQL query content:
{sql_content}


Please provide the analysis results in strict JSON format as follows:
{{
  "columns": [
    {{
      "target_table": "{join_nodes[0]}",
      "target_column": "target_column_name",
      "source_tables": [blue_tables[i], blue_tables[i], ...],
      "source_columns": ["source_table1.column_name", "source_table2.column_name", ...],
      "transformation": "description of transformation logic"
    }},
    ...
  ]
}}

CRITICAL REQUIREMENTS:
1. Analyze ALL columns that appear in the final SELECT clause of the query, ensuring complete coverage of the target table
2. For each target column, identify ALL contributing source tables and source columns as precisely as possible
3. For complex transformation logic, provide a clear and concise description of the transformation process
4. If the exact source cannot be determined, use "unknown" as the value but make every effort to identify the source
5. Ensure all table and column names exactly match those used in the SQL query
6. Focus on identifying which blue nodes contribute to which columns in the purple table
7. Provide a mapping for EVERY column in the target table - do not omit any columns
8. If a column is derived from multiple sources, list all contributing tables and columns
9. For calculated fields or expressions, describe the calculation logic in detail
10. Pay special attention to JOIN conditions and WHERE clause filters that might affect column derivation
11. IMPORTANT: All source_tables must be selected ONLY from the available tables listed above: {', '.join(blue_tables)}
12. Do not invent table names - use only the table names found in the join_graph nodes

Please ensure your analysis is thorough and complete, leaving no column unmapped.
"""
     # 创建带自定义 base_url 的客户端
    client = OpenAI(
        api_key=api_key,
        base_url="https://aihubmix.com/v1"  # 自定义 API 端点
    )

    try:
        # 使用新版 Chat Completions API
        response = client.chat.completions.create(
            model = model_name,  # 或根据需求使用 "gpt-4o-turbo"
            messages=[
                {"role": "system", "content": "你是一个SQL专家和数据血缘分析专家。请严格按照要求的JSON格式返回分析结果。"},
                {"role": "user", "content": prompt}
            ],
            response_format={"type": "json_object"},  # 要求返回 JSON
            temperature=0.1
        )
        
        # 从响应中提取 JSON 内容
        result = response.choices[0].message.content
        if not result:
            print(f"无法为 {join_nodes[0]} 生成列映射")
            return None
        
        # 保存结果
        output_path = output_dir / "blue_to_join_columns_mapping.json"
        columns_mapping = json.loads(result)
        with open(output_path, 'w') as f:
            json.dump(columns_mapping, f, indent=2)
        
        print(f"列映射已保存: {output_path}")
        return columns_mapping  # 解析为字典

        
    except Exception as e:
        print(f"使用LLM解析SQL时出错: {str(e)}")
        return None

def parse_sql_column_single(sql_content, api_key, model_name, output_dir):
    """使用LLM解析SQL文件中的列定义和数据血缘（从JOIN节点到紫色表）"""
    
    # 构建提示词
    prompt = f"""
Please perform a comprehensive column-level data lineage analysis for the following SQL transformation from JOIN node '{join_nodes[0]}' to purple table '{purple_tables[0]}'. I need a complete mapping showing how EVERY column in the target table '{purple_tables[0]}' is derived from the source table '{join_nodes[0]}' and its columns.

SQL transformation content:
{sql_content}
join

Please provide the analysis results in strict JSON format as follows:
{{
  "columns": [
    {{
      "target_table": "{purple_tables[0]}",
      "target_column": "target_column_name",
      "source_tables": ["{join_nodes[0]}"],
      "source_columns": ["{join_nodes[0]}.column_name"],
      "transformation": "description of transformation logic"
    }},
    ...
  ]
}}

CRITICAL REQUIREMENTS:
1. Analyze ALL columns that appear in the final SELECT clause of the query, ensuring complete coverage of the target table
2. For each target column, identify ALL contributing source columns from the JOIN node table
3. For complex transformation logic, provide a clear and concise description of the transformation process
4. If the exact source cannot be determined, use "unknown" as the value but make every effort to identify the source
5. Ensure all table and column names exactly match those used in the SQL query
6. Focus on identifying how columns from the JOIN node are transformed into columns in the purple table
7. Provide a mapping for EVERY column in the target table - do not omit any columns
8. If a column is derived from multiple sources, list all contributing columns from the JOIN node
9. For calculated fields or expressions, describe the calculation logic in detail
10. Pay special attention to any transformations, filters, or calculations applied to the JOIN node columns

Please ensure your analysis is thorough and complete, leaving no column unmapped.
"""

    # 创建带自定义 base_url 的客户端
    client = OpenAI(
        api_key=api_key,
        base_url="https://aihubmix.com/v1"  # 自定义 API 端点
    )

    try:
        # 使用新版 Chat Completions API
        response = client.chat.completions.create(
            model=model_name,
            messages=[
                {"role": "system", "content": "你是一个SQL专家和数据血缘分析专家。请严格按照要求的JSON格式返回分析结果。"},
                {"role": "user", "content": prompt}
            ],
            response_format={"type": "json_object"},  # 要求返回 JSON
            temperature=0.1
        )
        
        # 从响应中提取 JSON 内容
        result = response.choices[0].message.content
        if not result:
            print(f"无法为 {purple_tables[0]} 生成列映射")
            return None
        
        # 保存结果
        output_path = output_dir / "join_to_purple_columns_mapping.json"
        columns_mapping = json.loads(result)
        with open(output_path, 'w') as f:
            json.dump(columns_mapping, f, indent=2)
        
        print(f"JOIN到紫色表列映射已保存: {output_path}")
        return columns_mapping  # 解析为字典
        
    except Exception as e:
        print(f"使用LLM解析SQL时出错: {str(e)}")
        return None
    
def find_blue_nodes_from_join_graph(join_graph_path):
    """从join_graph.json文件中提取蓝色节点表名"""
    try:
        with open(join_graph_path, 'r') as f:
            join_graph = json.load(f)
        
        # 提取所有节点的表名
        blue_nodes = [node["table"] for node in join_graph.get("nodes", [])]
        return blue_nodes
    except Exception as e:
        print(f"读取join_graph.json时出错: {str(e)}")
        return []

def merge_blue_to_purple_mapping(blue_to_join_mapping, join_to_purple_mapping):
    """
    将蓝色节点到JOIN节点的映射与JOIN节点到紫色表的映射合并，
    创建从蓝色节点直接到紫色表的完整映射
    """
    # 确保输入是有效的字典且包含必要的结构
    if (not isinstance(blue_to_join_mapping, dict) or 
        not isinstance(join_to_purple_mapping, dict) or
        "columns" not in blue_to_join_mapping or
        "columns" not in join_to_purple_mapping):
        print("警告: 输入映射格式不正确")
        return {"columns": []}
    
    # 创建一个映射字典，用于快速查找JOIN节点列到紫色表列的映射
    join_to_purple_dict = {}
    for purple_col_info in join_to_purple_mapping["columns"]:
        # 假设每个JOIN节点列只映射到一个紫色表列
        join_column_key = f"{purple_col_info['source_tables'][0]}.{purple_col_info['source_columns'][0]}"
        join_to_purple_dict[join_column_key] = {
            "target_table": purple_col_info["target_table"],
            "target_column": purple_col_info["target_column"],
            "transformation": purple_col_info["transformation"]
        }
    
    # 创建最终的蓝色节点到紫色表的映射
    blue_to_purple_mapping = {"columns": []}
    
    for blue_col_info in blue_to_join_mapping["columns"]:
        # 获取蓝色节点列映射到的JOIN节点列
        join_column_key = f"{blue_col_info['target_table']}.{blue_col_info['target_column']}"
        
        # 查找这个JOIN节点列映射到的紫色表列
        if join_column_key in join_to_purple_dict:
            purple_info = join_to_purple_dict[join_column_key]
            
            # 创建蓝色节点到紫色表的映射项
            blue_to_purple_mapping["columns"].append({
                "target_table": purple_info["target_table"],
                "target_column": purple_info["target_column"],
                "source_tables": blue_col_info["source_tables"],
                "source_columns": blue_col_info["source_columns"],
                "transformation": f"{blue_col_info['transformation']} -> {purple_info['transformation']}"
            })
        else:
            # 如果没有找到映射，可能是JOIN节点列没有映射到紫色表
            print(f"警告: JOIN节点列 {join_column_key} 没有映射到紫色表")
    
    return blue_to_purple_mapping

# 修改版
def find_table_columns_mapping(purple_table, output_dir, args):
    """找到紫色表的列与蓝色节点的映射关系（两步映射：蓝色节点->JOIN节点->紫色表）"""
    purple_dir = output_dir / purple_table
    
    # 第一步：获取蓝色节点到JOIN节点的映射
    join_to_purple_dir = purple_dir / "sql" / "blue_to_purple" / "join_to_purple"
    
    # 读取蓝色节点到JOIN节点的SQL文件
    blue_to_join_sql_file = list(join_to_purple_dir.glob(f"step_1_*.sql"))[0]
    join_sql_name = blue_to_join_sql_file.stem  # 获取不带扩展名的文件名，格式为 "step_1_{join_name}"
    join_node_name = join_sql_name.replace("step_1_", "")
    
    # 从join_graph.json获取蓝色节点表名
    join_graph_path = purple_dir / "join_graph.json"
    blue_nodes = find_blue_nodes_from_join_graph(join_graph_path)
    
    if not blue_nodes:
        print(f"无法从join_graph.json获取蓝色节点信息")
        return {}
    
    # 设置全局变量
    global blue_tables
    blue_tables = blue_nodes 
    global purple_tables
    purple_tables = [purple_table]
    global join_nodes
    join_nodes = [join_node_name]

    # 解析蓝色节点到JOIN节点的SQL
    blue_to_join_mapping = {}
    sql_content = read_file_content(blue_to_join_sql_file)
    
    # 解析SQL中的列定义
    mapping = parse_sql_columns(
        sql_content, 
        api_key=args.api_key, 
        model_name=args.model, 
        output_dir=purple_dir
    )
        
    if mapping:
        blue_to_join_mapping = mapping
    
    # 第二步：获取JOIN节点到紫色表的映射（如果存在）
    join_to_purple_mapping = {}
    all_sql_files = list(join_to_purple_dir.glob("*.sql"))
    
    if len(all_sql_files) > 1: # 可能不存在
        # 对文件进行排序，确保按正确的顺序处理
        join_to_purple_sql_files = list(join_to_purple_dir.glob("*.sql"))[1:-1]  # 排除第一个文件
        join_to_purple_sql_files.sort()
        
        # 读取所有文件内容并合并
        sql_content = ""
        for sql_file in join_to_purple_sql_files:
            file_content = read_file_content(sql_file)
            sql_content += file_content + "\n\n"  # 添加换行符分隔不同文件的内容
        
        print(f"已合并 {len(join_to_purple_sql_files)} 个 JOIN 到紫色表的 SQL 文件")

        # 解析SQL中的列定义
        join_to_purple_mapping = parse_sql_column_single(
            sql_content, 
            api_key=args.api_key, 
            model_name=args.model, 
            output_dir=purple_dir
        )

    # 合并映射结果 
    final_mapping = blue_to_join_mapping   
    if join_to_purple_mapping:
        final_mapping = merge_blue_to_purple_mapping(blue_to_join_mapping=blue_to_join_mapping, join_to_purple_mapping=join_to_purple_mapping)
        
    # 保存最终的映射结果 还需要加入test.py逻辑
    output_path = purple_dir / "columns_mapping.json"
    with open(output_path, 'w') as f:
        json.dump(final_mapping, f, indent=2)
    
    return final_mapping

def filter_columns_by_node_combination(columns_mapping, node_combination, purple_table):
    """Filter columns based on node combination"""
    filtered_columns = {}
    
    # Extract node table names
    node_tables = [node["table"] for node in node_combination]
    
    # LLM returns data in a different format - adjust processing accordingly
    # columns_mapping is now a list of objects with target_table, source_tables, etc.
    for column_info in columns_mapping["columns"]:
        target_column = column_info["target_column"]
        source_tables = column_info["source_tables"]
        
        # Check if any source table is in the node combination or is the purple table itself
        should_keep = False
        
        # Check if any source table is in the node combination
        for source_table in source_tables:
            if source_table in node_tables or source_table == purple_table:
                should_keep = True
                break
        
        # If not, check if any node table appears in the transformation expression
        if not should_keep:
            transformation = column_info.get("transformation", "")
            for table in node_tables:
                if table in transformation:
                    should_keep = True
                    break
        
        # If the column should be kept, add it to the filtered results
        if should_keep:
            # Convert to the format expected by downstream functions
            filtered_columns[target_column] = {
                "source_table": source_tables[0] if source_tables else "unknown",
                "expression": column_info.get("transformation", "unknown")
            }
    
    return filtered_columns

def create_filtered_csv(purple_table, filtered_columns, node_combination, output_path, output_dir):
    """创建过滤后的CSV文件"""
    purple_csv_path = output_dir / purple_table / "tables" / f"{purple_table}.csv"
    
    if not purple_csv_path.exists():
        print(f"找不到紫色表 {purple_table} 的CSV文件")
        return False
    
    try:
        # 读取原始CSV
        df = pd.read_csv(purple_csv_path)
        
        # 过滤列
        columns_to_keep = list(filtered_columns.keys())
        df_filtered = df[columns_to_keep]
        
        # 保存过滤后的CSV
        df_filtered.to_csv(output_path, index=False)
        print(f"创建过滤后的CSV: {output_path}")
        
        return True
    except Exception as e:
        print(f"创建过滤后的CSV时出错: {str(e)}")
        return False

def create_filtered_schema(purple_table, filtered_columns, node_combination, output_path, output_dir):
    """创建过滤后的schema文件"""
    purple_schema_path = output_dir / purple_table / "schemas" / f"{purple_table}.schema"
    
    if not purple_schema_path.exists():
        print(f"找不到紫色表 {purple_table} 的schema文件")
        return False
    
    try:
        # 读取原始schema
        with open(purple_schema_path, 'r') as f:
            schema_content = f.read()
        
        # 提取列信息
        column_lines = []
        in_columns_section = False
        
        for line in schema_content.split('\n'):
            if line.strip().startswith('Column Name'):
                in_columns_section = True
                column_lines.append(line)
                continue
            
            if in_columns_section:
                if line.strip().startswith('-') or not line.strip():
                    continue
                
                if len(line.split()) < 2:
                    in_columns_section = False
                    continue
                
                col_name = line.split()[0].strip()
                if col_name in filtered_columns:
                    column_lines.append(line)
            else:
                column_lines.append(line)
        
        # 写入过滤后的schema
        with open(output_path, 'w') as f:
            f.write('\n'.join(column_lines))
        
        print(f"创建过滤后的schema: {output_path}")
        return True
    except Exception as e:
        print(f"创建过滤后的schema时出错: {str(e)}")
        return False

def create_filtered_sql(purple_table, filtered_columns, node_combination, output_path, output_dir):
    """创建过滤后的SQL文件"""
    # 有待商榷
    purple_sql_path = output_dir / purple_table / "sql" / "blue_to_purple" / "join_to_purple"
    sql_files = list(purple_sql_path.glob(f"step_1_*.sql"))
    
    if not sql_files:
        print(f"找不到紫色表 {purple_table} 的SQL文件")
        return False
    
    try:
        sql_content = read_file_content(sql_files[0])
        
        # 解析SQL并修改SELECT子句
        parsed = sqlparse.parse(sql_content)
        
        for stmt in parsed:
            if stmt.get_type() == 'SELECT':
                # 查找SELECT部分
                select_idx = -1
                from_idx = -1
                
                for i, token in enumerate(stmt.tokens):
                    if token.ttype is DML and token.value.upper() == 'SELECT':
                        select_idx = i
                    elif token.ttype is Keyword and token.value.upper() == 'FROM':
                        from_idx = i
                        break
                
                if select_idx >= 0 and from_idx > select_idx:
                    # 构建新的SELECT子句
                    select_clause = "SELECT\n    "
                    select_columns = []
                    
                    for col_name in filtered_columns:
                        col_info = filtered_columns[col_name]
                        select_columns.append(f"{col_info['expression']} AS {col_name}")
                    
                    select_clause += ",\n    ".join(select_columns)
                    
                    # 替换SELECT部分
                    stmt.tokens[select_idx + 1] = sqlparse.parse(select_clause)[0].tokens[1]
        
        # 保存修改后的SQL
        with open(output_path, 'w') as f:
            f.write(str(parsed[0]))
        
        print(f"创建过滤后的SQL: {output_path}")
        return True
    except Exception as e:
        print(f"创建过滤后的SQL时出错: {str(e)}")
        return False

def read_file_content(file_path):
    """读取文件内容"""
    try:
        with open(file_path, 'r') as f:
            return f.read()
    except Exception as e:
        print(f"读取文件 {file_path} 时出错: {str(e)}")
        return ""

def column_mapping_check(json_path, schema_path):
    column_mapping = json.load(open(json_path, 'r'))
    target_columns = [col['target_column'] for col in column_mapping['columns']]

    with open(schema_path, 'r', encoding="utf-8") as f:
        lines = f.readlines()
    schemas = [line.split(' ')[0] for line in lines[4:]]

    target_columns = set(target_columns)
    schemas = set(schemas)

    inconsistent_columns = target_columns ^ schemas
    if len(inconsistent_columns) == 0:
        print('target_columns is consistent with schema')
        return set()
    else:
        print("exclusive in target_columns:", inconsistent_columns & target_columns)
        print("exclusive in schemas:", inconsistent_columns & schemas)
        return inconsistent_columns & schemas


def column_mapping_check_dir(base_dir, project_name):
    output = {}
    dir_list = os.listdir(base_dir)
    for d in dir_list:
        if project_name in d:
            json_path = os.path.join(base_dir, d, 'columns_mapping.json')
            schema_path = os.path.join(base_dir, d, 'schemas', d + '.schema')
            print("\n=====================================================================")
            print(f"checking table {d}")
            output[d] = column_mapping_check(json_path, schema_path)
            print("=====================================================================\n")
    return output


def _remove_sql_comments(sql: str) -> str:
    # 去掉块注释
    sql = re.sub(r'/\*.*?\*/', '', sql, flags=re.S)
    # 去掉行尾单行注释
    sql = re.sub(r'--.*?$', '', sql, flags=re.M)
    return sql


# 从单个 SQL 字符串中提取表名（匹配 from "db"."schema"."table" 形式）
def extract_table_names_from_sql(sql: str) -> List[str]:
    sql_clean = _remove_sql_comments(sql)

    # 正则：匹配 from （或 join）后跟 "xxx"."yyy"."zzz" 的模式，捕获最后一段（表名）
    pattern = re.compile(
        r'(?:from|join)\s+'  # from 或 join 后面有空白
        r'"[^"]+"\s*\.\s*'  # "xxx" .
        r'"[^"]+"\s*\.\s*'  # "yyy" .
        r'"(?P<table>[^"]+)"',  # "zzz" -> 捕获表名
        flags=re.IGNORECASE
    )

    matches = [m.group('table') for m in pattern.finditer(sql_clean)]

    # 按出现顺序去重（保留第一次出现）
    seen = set()
    ordered_unique = []
    for t in matches:
        if t not in seen:
            seen.add(t)
            ordered_unique.append(t)

    return ordered_unique


# 从文件读取并提取
def extract_table_names_from_file(filepath: str) -> List[str]:
    p = Path(filepath)
    sql = p.read_text(encoding='utf-8')
    return extract_table_names_from_sql(sql)


def retrieve_columns(schema_dir, table_name):
    output = {}
    for table in table_name:
        schema_file = os.path.join(schema_dir, table + '.schema')
        with open(schema_file, 'r', encoding="utf-8") as f:
            lines = f.readlines()
        schemas = [line.split(' ')[0] for line in lines[4:]]
        output[table] = schemas
    return output


def check_exclusive_columns(schemas, exclusive_columns):
    exclusive_columns = set(exclusive_columns)
    for table_name in schemas:
        schema = schemas[table_name]
        schema = set(schema)
        print('columns in table {}: {}'.format(table_name, schema & exclusive_columns))


def append_json(json_file_path, exclusive_columns, source_tables, target_table):
    mapping_data = json.load(open(json_file_path, 'r'))
    column_mapping = mapping_data['columns']
    for exclusive_column in exclusive_columns:
        col_record = {
            "target_table": target_table,
            "target_column": exclusive_column,
            "source_tables": [],
            "source_columns": [],
            "transformation": "Direct selection from "
        }
        for table in source_tables:
            if exclusive_column in source_tables[table]:
                col_record["source_tables"].append(table)
                col_record["source_columns"].append(f"{table}.{exclusive_column}")
                col_record["transformation"] += f"{table}, "
        col_record["transformation"] = col_record["transformation"][:-2]
        if len(col_record["source_tables"]) > 0:
            column_mapping.append(col_record)
        else:
            print(f'No matched columns found for {exclusive_column}')
    save_path = os.path.join(os.path.dirname(json_file_path), 'columns_mapping.json')
    with open(save_path, "w", encoding="utf-8") as f:
        json.dump(mapping_data, f, ensure_ascii=False, indent=2)


def sql_contains_keywords(sql_path, keywords):
    sql_text = Path(sql_path).read_text(encoding="utf-8")
    not_contained_keywords = []
    for keyword in keywords:
        if keyword in sql_text:
            print(f"\"{keyword}\" found in sql, manual check needed")

def test(base_dir, project_name):
    # 检查是否有columns_mapping.json没有而schemas有的列
    purple_tables = column_mapping_check_dir(base_dir, project_name)
    count = []
    for purple_table in purple_tables:
        count.append(len(purple_tables[purple_table]))
    if sum(count) == 0:
        print('All consistent')
        return
    for i, purple_table in enumerate(purple_tables):
        if count[i] > 0:
            exclusive_columns_list = purple_tables[purple_table]
            print('\n===================================================')
            print(f'Processing {purple_table}...')
            # 从sql中获取所有原始表
            source_table_name = extract_table_names_from_file(
                f'{base_dir}/{purple_table}/sql/blue_to_purple/join_to_purple/step_1_{purple_table}.sql')
            # 检查sql中出现的列名
            sql_contains_keywords(f'{base_dir}/{purple_table}/sql/blue_to_purple/join_to_purple/step_1_{purple_table}.sql', exclusive_columns_list)
            # 获取各原始表列名
            source_table_schemas = retrieve_columns(f'{base_dir}/{purple_table}/schemas', source_table_name)
            append_json(f'{base_dir}/{purple_table}/columns_mapping.json',
                        exclusive_columns_list, source_table_schemas, purple_table)
            print("===================================================\n")

def process_node_combinations(purple_table, combinations_file, output_dir, args):
    """处理所有节点组合"""
    # 读取节点组合文件
    try:
        with open(combinations_file, 'r') as f:
            combinations = json.load(f)
    except Exception as e:
        print(f"读取节点组合文件时出错: {str(e)}")
        return
    
    # 获取紫色表的列映射
    output_path = output_dir / purple_table / "columns_mapping.json"
    columns_mapping = {}
    
    # 检查文件是否存在，不存在时才执行映射
    if args.overwrite == 'True':
        columns_mapping = find_table_columns_mapping(purple_table, output_dir, args)
    elif not output_path.exists():
        columns_mapping = find_table_columns_mapping(purple_table, output_dir, args)
    else:
        print(f"列映射文件已存在，跳过: {output_path}")
        # 如果后续需要用到columns_mapping变量，可以在这里读取已存在的文件
        with open(output_path, 'r') as f:
            columns_mapping = json.load(f)

    if not columns_mapping:
        print(f"无法获取紫色表 {purple_table} 的列映射")
        return
    
    # 处理每个节点组合
    for i, combo in enumerate(combinations):
        if "nodes" not in combo:
            print(f"跳过组合 {i}: 缺少'nodes'字段")
            continue
        
        node_combination = combo["nodes"]
        if not node_combination:
            print(f"跳过组合 {i}: 节点列表为空")
            continue
        
        print(f"\n处理组合 {i+1}/{len(combinations)}: {[node['table'] for node in node_combination]}")
        
        # 过滤列
        filtered_columns = filter_columns_by_node_combination(columns_mapping, node_combination, purple_table)
        
        # 创建输出目录
        combo_path = output_dir / purple_table / "query" 
        combo_path.mkdir(exist_ok=True)
        combo_dir = output_dir / purple_table / "query" / f"combination_{i+1}"
        combo_dir.mkdir(exist_ok=True)
        
        tables_dir = combo_dir / "tables"
        schemas_dir = combo_dir / "schemas"
        
        for dir_path in [tables_dir, schemas_dir]:
            dir_path.mkdir(exist_ok=True, parents=True)
        
        # 创建过滤后的文件
        csv_success = create_filtered_csv(purple_table, filtered_columns, node_combination, 
                                         tables_dir / f"{purple_table}.csv", output_dir)
        schema_success = create_filtered_schema(purple_table, filtered_columns, node_combination, 
                                               schemas_dir / f"{purple_table}.schema", output_dir)
        # sql_success = create_filtered_sql(purple_table, filtered_columns, node_combination, 
        #                                  sql_dir / f"{purple_table}.sql", output_dir)
        
        if csv_success and schema_success: # and sql_success:
            print(f"成功创建组合 {i+1} 的过滤文件")
        else:
            print(f"创建组合 {i+1} 的过滤文件时遇到问题")

def main():
    # 设置命令行参数
    parser = argparse.ArgumentParser(description='根据节点组合过滤紫色表的列')
    parser.add_argument('--db_name', required=True, help='数据库名称')
    parser.add_argument('--combinations_file', help='节点组合JSON文件路径，默认为每个紫色表目录下的subgraph_combinations.json')
    parser.add_argument('--model', default='gpt-4o-mini', help='使用的模型名称，默认为gpt-4')
    parser.add_argument('--api_key', required=True, help='OpenAI API密钥')
    parser.add_argument('--overwrite', required=True, help='是否覆盖已存在的列映射文件，True或False')
    
    args = parser.parse_args()
    
    # 设置输出目录
    output_dir = Path(f"/home/wangyuhui/dbt_dataset_test/{args.db_name}")
    
    # 获取所有紫色表
    purple_tables = []
    for item in output_dir.iterdir():
        if item.is_dir():
            purple_tables.append(item.name)
    
    if not purple_tables:
        print("没有找到紫色表")
        return
    
    print(f"找到紫色表: {', '.join(purple_tables)}")
    
    # 处理每个紫色表
    for purple_table in purple_tables:
        print(f"\n{'='*50}")
        print(f"处理紫色表: {purple_table}")
        print(f"{'='*50}")
        
        # 查找节点组合文件
        if args.combinations_file:
            combinations_file = Path(args.combinations_file)
        else:
            combinations_file = output_dir / purple_table / "subgraph_combinations.json"
        
        if not combinations_file.exists():
            print(f"找不到节点组合文件: {combinations_file}")
            continue
        
        process_node_combinations(purple_table, combinations_file, output_dir, args)
    
    test(base_dir=output_dir, project_name=args.db_name)

if __name__ == "__main__":
    main()

'''
python /home/wangyuhui/Auto-ETL/wyh-dataset/DBT_fivetran/execute_answer.py \
    --db_name hubspot \
    --api_key "sk-clvm8vh2xDLljzWN6eF021AdE6534dF98232F40731559cA6" \
    --model gpt-4o-mini \
    --overwrite True
'''