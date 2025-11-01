from openai import OpenAI
import yaml
import re
import argparse
import os
import json
from typing import List, Dict, Any

def read_file_content(file_path: str) -> str:
    """读取文件内容，处理文件不存在的情况"""
    try:
        with open(file_path, 'r', encoding='utf-8') as file:
            return file.read()
    except FileNotFoundError:
        print(f"错误: 文件 {file_path} 不存在")
        return ""
    except Exception as e:
        print(f"读取文件 {file_path} 时出错: {str(e)}")
        return ""

def read_schema_file(schema_path: str) -> List[Dict]:
    """读取.schema文件并解析为列信息列表"""
    try:
        with open(schema_path, 'r', encoding='utf-8') as file:
            lines = file.readlines()
        
        columns = []
        # 跳过表名和分隔线
        start_processing = False
        
        for line in lines:
            line = line.strip()
            
            # 跳过空行和分隔线
            if not line or line.startswith('='):
                continue
                
            # 跳过表名行
            if line.startswith('Table:'):
                continue
                
            # 检查是否到达列标题行
            if 'Column Name' in line and 'Data Type' in line:
                start_processing = True
                continue
                
            # 处理列数据行
            if start_processing:
                # 使用正则表达式匹配列信息
                # 列名可能包含空格，所以不能简单分割
                # 假设格式为: 列名(可能含空格) + 至少2个空格 + 数据类型 + 至少2个空格 + 长度 + 至少2个空格 + Nullable + 至少2个空格 + Default
                match = re.match(r'^(\S+(?:\s+\S+)*)\s{2,}(\S+)\s{2,}(\S*)\s{2,}(\S+)\s{2,}(\S.*)$', line)
                if match:
                    col_name = match.group(1).strip()
                    data_type = match.group(2).strip()
                    length = match.group(3).strip()
                    nullable = match.group(4).strip()
                    default = match.group(5).strip()
                    
                    columns.append({
                        'name': col_name,
                        'type': data_type,
                        'length': length if length else None,
                        'nullable': nullable,
                        'default': default if default != 'None' else None
                    })
                else:
                    # 如果正则匹配失败，尝试简单的分割方法
                    parts = re.split(r'\s{2,}', line)
                    if len(parts) >= 5:
                        columns.append({
                            'name': parts[0].strip(),
                            'type': parts[1].strip(),
                            'length': parts[2].strip() if parts[2].strip() else None,
                            'nullable': parts[3].strip(),
                            'default': parts[4].strip() if parts[4].strip() != 'None' else None
                        })
        
        return columns
        
    except FileNotFoundError:
        print(f"错误: schema文件 {schema_path} 不存在")
        return []
    except Exception as e:
        print(f"读取schema文件 {schema_path} 时出错: {str(e)}")
        return []

def get_purple_table_directories(project_path: str) -> List[str]:
    """获取项目目录下的所有紫色表目录"""
    try:
        directories = []
        for item in os.listdir(project_path):
            item_path = os.path.join(project_path, item)
            if os.path.isdir(item_path):
                directories.append(item)
        return directories
    
    except Exception as e:
        print(f"读取项目目录 {project_path} 时出错: {str(e)}")
        return []

def get_combination_directories(query_path: str) -> List[str]:
    """获取查询目录下的所有组合目录"""
    try:
        directories = []
        for item in os.listdir(query_path):
            item_path = os.path.join(query_path, item)
            if os.path.isdir(item_path) and item.startswith("combination_"):
                directories.append(item)
        return directories
    except Exception as e:
        print(f"读取查询目录 {query_path} 时出错: {str(e)}")
        return []

def get_schema_files(combination_path: str) -> List[str]:
    """获取组合目录下的所有schema文件"""
    schema_dir = os.path.join(combination_path, "schemas")
    if not os.path.exists(schema_dir):
        return []
    
    schema_files = []
    for file in os.listdir(schema_dir):
        if file.endswith(".schema"):
            schema_files.append(os.path.join(schema_dir, file))
    
    return schema_files

def extract_column_descriptions(data_model: Dict, table_name: str) -> Dict[str, str]:
    """从data_model中提取指定表的列描述"""
    descriptions = {}
    
    if not data_model or 'models' not in data_model:
        return descriptions
    
    for model in data_model['models']:
        if model.get('name') == table_name and 'columns' in model:
            for column in model['columns']:
                if 'name' in column and 'description' in column:
                    descriptions[column['name']] = column['description']
    
    return descriptions

def generate_natural_language_query(
    combination_name: str,
    schema_files: List[str],
    data_model: Dict,
    api_key: str,
    model_name: str = "gpt-4o"
) -> str:
    """
    根据提供的schema信息生成自然语言查询
    
    参数:
    combination_name: 组合名称
    schema_files: schema文件路径列表
    data_model: 数据模型内容
    model_name: 使用的大模型名称
    
    返回:
    自然语言查询字符串
    """
    
    # 读取所有schema文件内容
    schema_data = []
    for schema_file in schema_files:
        table_name = os.path.basename(schema_file).replace('.schema', '')
        columns = read_schema_file(schema_file)
        column_descriptions = extract_column_descriptions(data_model, table_name)
        
        
        # 为每个列添加描述
        for column in columns:
            column_name = column.get('name', '')
            if column_name in column_descriptions:
                column['description'] = column_descriptions[column_name]
        
        schema_data.append({
            'table_name': table_name,
            'columns': columns
        })
    
    if not schema_data:
        return "错误: 没有有效的schema数据"
    
    # 构建prompt
    prompt = f"""
You are a senior data analyst specializing in business intelligence and data modeling. Your task is to generate a distinct natural language business query based on the provided schema information.

Input includes:
- Combination name: {combination_name}
- Schema data: {schema_data}
- Data model descriptions: {data_model}

Please follow these steps to create a professional, business-focused natural language query with distinct characteristics:

1. **Schema Analysis**:
   - Analyze all tables and their columns from the schema data
   - Use column descriptions from the data model to understand the business meaning of each column
   - Identify which columns come from which table and their specific purposes

2. **Table-Specific Column Identification**:
   - For each table in the schema data, identify all key columns that are unique to that table
   - Note the specific combination of tables and how they relate to each other
   - Identify any unique metrics or dimensions that are only available in this specific combination

3. **Business Context Identification**:
   - Analyze column names and descriptions to understand the business domain
   - Identify key business metrics: numerical measures like costs, revenues, conversions, etc.
   - Identify dimensions: categorical fields like accounts, campaigns, dates, statuses, etc.
   - Determine time granularity based on date/time columns
   - Note any status or filter columns that might indicate business conditions

4. **Query Intent Inference**:
   - Based on the identified metrics and dimensions, determine the most likely business question
   - Ensure the query reflects the specific combination of tables and their unique columns
   - Consider how this specific table combination differs from others and what unique insights it can provide
   - Focus on the most distinctive aspects of this combination to create a unique query

5. **Natural Language Query Generation**:
   - Craft a single, concise, natural-sounding business question in English
   - Use business terminology rather than technical column names
   - Ensure the query explicitly incorporates columns from ALL provided tables
   - Make the query specific enough to distinguish it from queries with different table combinations
   - Focus on the most distinctive aspects of this combination
   - Avoid generic terms like "email engagement metrics" - be specific about what metrics are being measured
   - Follow this format: "I need [time granularity] report on [specific metrics from all tables] [broken down by dimensions] [with specific filters]"

Your output should be ONLY the natural language query with no additional explanations, justifications, or formatting.

Important: 
- The query must explicitly include columns/references from ALL tables in the provided combination. 
- Different table combinations should result in distinctly different queries.
- Ensure the query explicitly indicates which business concepts come from which tables, without using technical table names.
- Avoid generic terms and instead use specific metrics and dimensions that are unique to this combination.
- Focus on the most distinctive aspects of this combination to create a truly unique query.

Examples of distinct queries for different table combinations:
- For tables [email_sends, email_opens]: "I need a daily report on email campaign performance, including send-to-open ratios and time-to-open metrics, broken down by campaign type and geographic region, with filters for specific send times and audience segments."
- For tables [email_clicks, conversions]: "I need a weekly analysis of email-driven conversion funnel performance, covering click-through rates, conversion rates, and revenue attribution by email campaign and landing page, with filters for high-value customer segments."
- For tables [email_bounces, unsubscribes]: "Provide a monthly monitoring report on email deliverability issues, including hard bounce rates, soft bounce trends, and unsubscribe reasons by campaign and sender reputation, with alerts for sudden spikes in delivery problems."

Note how each query focuses on the unique combination of tables and avoids generic terminology.
"""
    # 创建带自定义 base_url 的客户端
    client = OpenAI(
        api_key=api_key,
        base_url="https://aihubmix.com/v1"  # 自定义 API 端点
    )

    try:
        # 使用新版 Chat Completions API
        response = client.chat.completions.create(
            model = model_name,
            messages=[
                {"role": "system", "content": "你是一个SQL专家和图形建模专家。"},
                {"role": "user", "content": prompt}
            ],
            temperature=0.1
        )
        
        # 提取并返回生成的查询
        result = response.choices[0].message.content.strip()
        
        # 确保结果是一句话，没有额外解释
        if "\n" in result:
            result = result.split("\n")[0]
        
        # 移除可能的引号
        result = re.sub(r'^["\']|["\']$', '', result)
        
        return result
        
    except Exception as e:
        return f"API调用错误: {str(e)}"

def load_data_models_from_directory(models_dir: str) -> Dict[str, Any]:
    """
    从指定目录下的所有YAML文件中加载数据模型
    
    参数:
    models_dir: 包含YAML文件的目录路径
    
    返回:
    合并后的数据模型字典
    """
    data_models = {"models": []}
    
    if not os.path.exists(models_dir):
        print(f"警告: 模型目录 {models_dir} 不存在")
        return data_models
    
    # 遍历目录下的所有YAML文件
    for file_name in os.listdir(models_dir):
        if file_name.endswith(('.yml', '.yaml')):
            file_path = os.path.join(models_dir, file_name)
            try:
                with open(file_path, 'r', encoding='utf-8') as file:
                    content = yaml.safe_load(file)
                    # 如果文件包含模型定义，则合并到总模型中
                    if content and 'models' in content:
                        data_models['models'].extend(content['models'])
            except Exception as e:
                print(f"读取YAML文件 {file_path} 时出错: {str(e)}")
    
    return data_models


def main():
    # 设置命令行参数解析
    parser = argparse.ArgumentParser(description='生成自然语言查询')
    parser.add_argument('--project', required=True, 
                        help='项目名称，用于确定查询组合的基目录')
    parser.add_argument('--model', default='gpt-4o-mini', help='使用的模型名称，默认为gpt-4o-mini')
    parser.add_argument('--api_key', required=True, help='OpenAI API密钥')
    parser.add_argument('--output', default='queries.txt', help='输出文件路径，默认为queries.txt')
    
    args = parser.parse_args()
    
    # 构建项目基目录路径
    project_base_dir = f"/home/wangyuhui/dbt_dataset_test/{args.project}"
    base_dir = f"/home/wangyuhui/Auto-ETL/wyh-dataset/DBT_fivetran/dbt_{args.project}"

    # 从models目录加载所有YAML文件
    models_dir = os.path.join(base_dir, "models")
    data_model = load_data_models_from_directory(models_dir)
    
    if not data_model or not data_model.get('models'):
        print("警告: 没有找到有效的数据模型定义")
        data_model = {"models": []}
    
    # 获取所有紫色表目录
    purple_tables = get_purple_table_directories(project_base_dir)
    if not purple_tables:
        print(f"在项目目录 {project_base_dir} 中没有找到紫色表目录")
        return
    
    # 准备输出文件
    output_lines = []

    # 处理每个紫色表目录
    for purple_table in purple_tables:
        purple_table_path = os.path.join(project_base_dir, purple_table)
        query_path = os.path.join(purple_table_path, "query")
        
        if not os.path.exists(query_path):
            print(f"跳过紫色表 {purple_table}: 查询目录不存在")
            continue
        
        # 获取查询目录下的所有组合目录
        combinations = get_combination_directories(query_path)
        if not combinations:
            print(f"在查询目录 {query_path} 中没有找到组合目录")
            continue
        
        # 处理每个组合
        for combination in combinations:
            combination_path = os.path.join(query_path, combination)
            schema_files = get_schema_files(combination_path)
            
            if not schema_files:
                print(f"跳过组合 {combination}: 没有找到schema文件")
                continue
            
            print(f"处理紫色表: {purple_table}, 组合: {combination}")
            
            # 生成自然语言查询
            query = generate_natural_language_query(
                combination,
                schema_files,
                data_model,
                args.api_key,
                args.model
            )
            
            # 添加到输出
            output_lines.append(f"Purple Table: {purple_table}")
            output_lines.append(f"Combination: {combination}")
            output_lines.append(f"Query: {query}")
            output_lines.append("")  # 空行分隔
            
            print(f"生成的查询: {query}")
            print("")  # 空行分隔

    # 写入输出文件
    try:
        query_output = os.path.join(project_base_dir, args.output)
        with open(query_output, 'w', encoding='utf-8') as f:
            f.write("\n".join(output_lines))
        print(f"结果已保存到 {args.output}")
    except Exception as e:
        print(f"写入输出文件时出错: {str(e)}")

if __name__ == "__main__":
    main()

# 根据schema找到对应的描述 从而根据表述写出NL query
# combination /home/wangyuhui/dbt_dataset/hubspot/hubspot__email_event_clicks/query/combination_i
# 根据其中./schema中的table.schema文件以及对应的data_model.yml（用户超参输入）中对于相关列的描述
# 生成自然语言查询
# 仅在脚本直接运行时执行
'''
python query_EA.py \
  --project hubspot \
  --api_key "sk-clvm8vh2xDLljzWN6eF021AdE6534dF98232F40731559cA6" \
'''