from pathlib import Path
from openai import OpenAI  # 导入新版 SDK
import json
import argparse
import sys

        
def generate_join_graph_prompt(sql_content):
    """生成调用大模型的提示词"""
    prompt = f"""
你是一个SQL专家和图形建模专家。请分析以下SQL查询中的JOIN关系，并创建一张连接图，其中：
- 节点表示参与JOIN的表
- 边表示表之间的JOIN条件

请严格遵循以下规则：
1. 节点只包含原始输入表（即直接来自数据库的表），不包括任何CTE(Common Table Expressions)或子查询
2. 每个节点应该包含表的名称和别名（如果存在）
3. 每条边应该包含JOIN条件和连接类型（INNER JOIN, LEFT JOIN等）
4. 图应该是完整的，包含SQL中所有参与JOIN的原始表
5. 忽略任何中间表（如CTEs），只关注最终JOIN的原始表
6. json格式中的id必须为1开始自增, edge中增加一个checked字段，初始值为false,不需要修改false值

重要说明：
- CTE（如示例中的deals, deals_agg）不是原始表，不应作为节点
- 即使CTE在查询中被引用，也不应为它们创建节点
- 只包含直接来自数据库的物理表或视图

SQL查询内容：
{sql_content}

请以以下JSON格式返回结果：
{{
  "nodes": [
    {{
      "id": "1",
      "table": "table_name",
      "alias": "table_alias"
    }},
    ...
  ],
  "edges": [
    {{
      "source": "source_node_id",
      "target": "target_node_id",
      "condition": "join_condition",
      "type": "join_type"
      "checked": false
    }},
    ...
  ]
}}
"""
    return prompt

def call_openai_api(prompt):
    """调用OpenAI API获取连接图（新版SDK）"""
    # 创建带自定义 base_url 的客户端
    client = OpenAI(
        api_key="sk-q9Qhk8KvRqkBQ6Ti6d971dBbCbCd491b8b995325918cA471",  # 替换为你的实际 API 密钥
        base_url="https://aihubmix.com/v1"  # 自定义 API 端点
    )
    
    try:
        # 使用新版 Chat Completions API
        response = client.chat.completions.create(
            model="gpt-4o",  # 或根据需求使用 "gpt-4o-turbo"
            messages=[
                {"role": "system", "content": "你是一个SQL专家和图形建模专家。"},
                {"role": "user", "content": prompt}
            ],
            response_format={"type": "json_object"},  # 要求返回 JSON
            temperature=0.1
        )
        
        # 从响应中提取 JSON 内容
        result = response.choices[0].message.content
        return json.loads(result)  # 解析为字典
    
    except Exception as e:
        print(f"调用OpenAI API失败: {e}")
        return None


def generate_join_graph_for_purple_table(purple_table, output_dir):
    """为紫色表生成连接图"""
    # 找到蓝色到紫色链中的紫色表 SQL 文件
    blue_to_purple_dir = output_dir / purple_table / "sql" / "blue_to_purple" / "join_to_purple"
    print(f"查找目录: {blue_to_purple_dir}")
    sql_path = None
    if blue_to_purple_dir.exists():
        # 查找包含join节点的 SQL 文件
        sql_path_list = list(blue_to_purple_dir.glob(f"step_1_*.sql"))
        
        if sql_path_list:
            # 使用找到的第一个文件
            sql_path = sql_path_list[0]
            print(f"找到紫色表join节点的SQL文件: {sql_path}")
    
    # 读取SQL内容
    with open(sql_path, 'r') as f:
        sql_content = f.read()
    
    # 生成提示词
    prompt = generate_join_graph_prompt(sql_content)
    
    # 调用OpenAI API
    join_graph = call_openai_api(prompt)
    
    if not join_graph:
        print(f"无法为 {purple_table} 生成连接图")
        return None
    
    # 保存结果
    output_path = output_dir / purple_table / "join_graph.json"
    with open(output_path, 'w') as f:
        json.dump(join_graph, f, indent=2)
    
    print(f"连接图已保存: {output_path}")
    return join_graph

def process_all_purple_tables(output_dir):
    """处理所有紫色表"""
    # 获取所有紫色表目录
    purple_tables = []
    for dir_path in output_dir.iterdir():
        if dir_path.is_dir():
            purple_tables.append(dir_path.name)
    
    if not purple_tables:
        print(f"在 {output_dir} 中没有找到紫色表")
        return
    
    print(f"找到紫色表: {', '.join(purple_tables)}")
    
    jump_dir = ["datalake_sources","hubspot__daily_ticket_history"] #如果已经确认好了，可以将确认过的紫色表文件夹名写在这
    # 为每个紫色表生成连接图
    for purple_table in purple_tables:
        if purple_table not in jump_dir:
            print(f"\n处理紫色表: {purple_table}")
            generate_join_graph_for_purple_table(purple_table, output_dir)

def process_dbt_project(project_name: str):
    """处理DBT项目"""
    try:
        # 构造路径
        project_dir = Path(f"/home/wangyuhui/Auto-ETL/wyh-dataset/DBT_fivetran/dbt_{project_name}/integration_tests")
        output_dir = Path(f"/home/wangyuhui/dbt_dataset_test/{project_name}")
        
        # 验证路径存在
        if not project_dir.exists():
            raise FileNotFoundError(f"项目目录不存在: {project_dir}")
        
        # 处理紫表
        process_all_purple_tables(output_dir)

        print(f"成功处理项目: {project_name}")
        print(f"项目目录: {project_dir}")
        print(f"输出目录: {output_dir}")
        return True
    
    except Exception as e:
        print(f"处理失败: {e}", file=sys.stderr)
        return False
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="DBT项目处理器")
    parser.add_argument("project", help="项目名称")
    parser.add_argument("--verbose", "-v", action="store_true", help="详细模式")
    
    args = parser.parse_args()
    
    if args.verbose:
        print(f"开始处理项目: {args.project}")
    
    success = process_dbt_project(args.project)
    
    if args.verbose:
        print(f"处理 {'成功' if success else '失败'}")
    
    sys.exit(0 if success else 1)

# python /home/wangyuhui/Auto-ETL/wyh-dataset/DBT_fivetran/query_prepare.py hubspot -v
# 可能会出错，可以看看query_select.py的输出如果为空，则手动调整一下join_graph.json文件