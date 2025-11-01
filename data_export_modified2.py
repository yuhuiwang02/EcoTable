import os
import json
import psycopg2
import networkx as nx
import shutil
from pathlib import Path
import re
from collections import deque

# 从环境变量获取配置信息
db_name = os.getenv("DB_NAME", "hubspot")
user_name = os.getenv("USER_NAME", "wyh")
user_password = os.getenv("USER_PASSWORD", "yuhui.wang.db")
dbt_project_name = os.getenv("DBT_PROJECT_NAME", f"dbt_{db_name}")

# 配置路径
project_dir = Path(f"/home/wangyuhui/Auto-ETL/wyh-dataset/DBT_fivetran/{dbt_project_name}/integration_tests")
compiled_dir = project_dir / "target" / "compiled"
manifest_path = project_dir / "target" / "manifest.json"
catalog_path = project_dir / "target" / "catalog.json"
output_dir = Path(f"/home/wangyuhui/dbt_dataset_test/{db_name}")
output_dir.mkdir(exist_ok=True)

# 数据库配置
db_config = {
    "host": "localhost",
    "port": 5434,
    "user": user_name,
    "password": user_password, 
    "database": db_name
}

# 数据库模式配置
db_schema = f"public_{db_name}"  # 数据库中的模式名

def extract_table_name_from_image(image_path):
    """从图片文件名中提取表名"""
    return image_path.stem

def get_blue_node_upstream_roots(graph, blue_node_id):
    """获取蓝色节点的所有最上游父节点（根节点）"""
    root_nodes = set()
    visited = set()
    stack = [blue_node_id]
    
    while stack:
        current_node = stack.pop()
        
        # 如果已经访问过，跳过
        if current_node in visited:
            continue
        visited.add(current_node)
        
        # 获取当前节点的所有直接上游节点
        upstream_nodes = list(graph.predecessors(current_node))
        
        if not upstream_nodes:  # 没有上游节点，说明是根节点
            root_nodes.add(current_node)
        else:
            # 继续向上游遍历
            stack.extend(upstream_nodes)
    
    return root_nodes

def get_purple_tables(manifest, graph):
    """获取所有符合条件的紫色表名"""
    purple_tables = []

    # 遍历所有模型节点
    for node_id, node_data in manifest["nodes"].items():
        if node_data["resource_type"] != "model":
            continue
            
        table_name = node_data["name"]
        should_keep = True  # 标记是否保留该紫色表
        
        # 获取最近的Join节点
        join_node_id = find_nearest_join_node(graph, node_id)
        if not join_node_id:
            print(f"警告: 紫色表 {table_name} 没有找到合适的JOIN节点，跳过该紫色表")
            continue

        # 获取Join的蓝色节点所有上游根节点
        upstream_nodes = list(graph.predecessors(join_node_id))
        if len(upstream_nodes) >= 3:
            # 找到蓝色节点
            blue_nodes = find_blue_nodes(graph, node_id)
            if blue_nodes:  # 确保有蓝色节点
                # 获取所有蓝色节点的根节点集合
                used_roots = set()
                for blue_node_id in blue_nodes:
                    roots = get_blue_node_upstream_roots(graph, blue_node_id)
                    if not roots.intersection(used_roots):
                        used_roots.update(roots)
                    else:
                        print(f"警告: 紫色表 {table_name} 的蓝色节点有交集，跳过该紫色表")
                        should_keep = False
                        break

                # 如果蓝色节点的源节点之间没有交集则保留该紫色表
                if should_keep:
                    purple_tables.append(table_name)
    
    return purple_tables

def load_manifest():
    """加载dbt manifest文件"""
    with open(manifest_path, 'r') as f:
        return json.load(f)
    
def build_dependency_graph(manifest):
    """构建完整的 dbt 依赖图（包括模型和源表）"""
    graph = nx.DiGraph()
    
    # 添加所有模型节点
    for node_id, node_data in manifest["nodes"].items():
        if node_data["resource_type"] == "model":
            graph.add_node(node_id, **node_data)
    
    # 添加所有源表节点
    for source_id, source_data in manifest["sources"].items():
        source_node = {
            "name": source_data["name"],
            "resource_type": "source",
            "source_name": source_data["source_name"],
            "identifier": source_data.get("identifier", source_data["name"]),
            "original_file_path": source_data["original_file_path"]
        }
        graph.add_node(source_id, **source_node)
    
    # 添加边（依赖关系）
    for node_id, node_data in manifest["nodes"].items():
        if node_data["resource_type"] == "model":
            for parent in node_data["depends_on"]["nodes"]:
                if parent in manifest["nodes"]:
                    parent_type = manifest["nodes"][parent]["resource_type"]
                    if parent_type == "model" or parent_type == "source":
                        graph.add_edge(parent, node_id)
                elif parent in manifest["sources"]:
                    graph.add_edge(parent, node_id)
    
    return graph

def find_node_id(manifest, table_name):
    """根据表名查找节点ID（支持模型和源表）"""
    # 在模型中查找
    for node_id, node_data in manifest["nodes"].items():
        if node_data["resource_type"] == "model" and node_data["name"] == table_name:
            return node_id
    
    # 在源表中查找
    for source_id, source_data in manifest["sources"].items():
        if source_data["name"] == table_name:
            return source_id
    
    return None

def find_sql_file(model_name):
    """递归查找模型的SQL文件路径"""
    exact_filename = f"{model_name}.sql"
    
    # 递归搜索整个编译目录
    for root, _, files in os.walk(compiled_dir):
        for file in files:
            if file == exact_filename:
                return Path(root) / file
            
            if file.endswith(".sql") and model_name in file:
                return Path(root) / file
    
    # 如果找不到，尝试在模型名中去掉可能的模式前缀
    if "__" in model_name:
        simplified_name = model_name.split("__")[-1]
        return find_sql_file(simplified_name)
    
    return None

def export_table_data(node_info, output_path):
    """根据节点信息导出表数据到CSV"""
    try:
        conn = psycopg2.connect(**db_config)
        cursor = conn.cursor()
        
        # 根据节点类型确定实际表名和模式
        if node_info["resource_type"] == "source":
            # 对于源表，使用 identifier 作为表名，schema 作为模式
            table_name = node_info.get("identifier", node_info["name"])
            schema_name = node_info.get("schema", "public")  # 默认使用 public 模式
        else:
            # 对于模型表，使用配置的模式
            table_name = node_info["name"]
            schema_name = db_schema
        
        # 检查表是否存在
        check_query = """
        SELECT EXISTS (
            SELECT FROM information_schema.tables 
            WHERE table_schema = %s 
            AND table_name = %s
        )
        """
        cursor.execute(check_query, (schema_name, table_name))
        table_exists = cursor.fetchone()[0]
        
        if not table_exists:
            print(f"警告: 表 {schema_name}.{table_name} 不存在（可能是暂存表或临时表），跳过导出")
            cursor.close()
            conn.close()

            if node_info["resource_type"] != "source":
                # 创建一个标记文件，表示这是暂存表
                with open(output_path.with_suffix('.temp_table_info'), 'w') as f:
                    f.write(f"Table {table_name} is a temporary or ephemeral table and does not exist in the database.\n")
                    f.write(f"This file was created as a placeholder for documentation purposes.")
            
            return False
        
        # 使用完整模式名导出
        full_table_name = f"{schema_name}.{table_name}"
        
        # 导出数据
        with open(output_path, 'w') as f:
            cursor.copy_expert(f"COPY (SELECT * FROM {full_table_name}) TO STDOUT WITH CSV HEADER", f)
        
        cursor.close()
        conn.close()
        print(f"导出表数据: {full_table_name} -> {output_path}")
        return True
    
    except Exception as e:
        print(f"错误: 导出表 {table_name} 时出错: {str(e)}")
        if node_info["resource_type"] != "source":
            with open(output_path.with_suffix('.error_info'), 'w') as f:
                f.write(f"Error exporting table {table_name}: {str(e)}\n")
                f.write(f"This might be a temporary table, an ephemeral model, or there might be connection issues.")
        return False

def export_table_schema(node_info, output_path):
    """根据节点信息导出表结构到文件"""
    try:
        conn = psycopg2.connect(**db_config)
        cursor = conn.cursor()
        
        # 根据节点类型确定实际表名和模式
        if node_info["resource_type"] == "source":
            # 对于源表，使用 identifier 作为表名，schema 作为模式
            table_name = node_info.get("identifier", node_info["name"])
            schema_name = node_info.get("schema", "public")  # 默认使用 public 模式
        else:
            # 对于模型表，使用配置的模式
            table_name = node_info["name"]
            schema_name = db_schema
        
        # 检查表是否存在
        check_query = """
        SELECT EXISTS (
            SELECT FROM information_schema.tables 
            WHERE table_schema = %s 
            AND table_name = %s
        )
        """
        cursor.execute(check_query, (schema_name, table_name))
        table_exists = cursor.fetchone()[0]
        
        if not table_exists:
            print(f"警告: 表 {schema_name}.{table_name} 不存在（可能是暂存表或临时表），跳过导出结构")
            cursor.close()
            conn.close()
            
            # 创建一个标记文件，表示这是暂存表
            if node_info["resource_type"] != "source":
                with open(output_path.with_suffix('.temp_table_info'), 'w') as f:
                    f.write(f"Table {table_name} is a temporary or ephemeral table and does not exist in the database.\n")
                    f.write(f"This file was created as a placeholder for documentation purposes.")
            
            return False
        
        # 查询表结构
        query = """
        SELECT column_name, data_type, character_maximum_length, 
               is_nullable, column_default
        FROM information_schema.columns
        WHERE table_schema = %s AND table_name = %s
        ORDER BY ordinal_position
        """
        
        cursor.execute(query, (schema_name, table_name))
        columns = cursor.fetchall()
        
        # 导出结构信息
        with open(output_path, 'w') as f:
            f.write(f"Table: {schema_name}.{table_name}\n")
            f.write("=" * 80 + "\n")
            f.write(f"{'Column Name':<30} {'Data Type':<20} {'Length':<10} {'Nullable':<10} {'Default':<20}\n")
            f.write("-" * 80 + "\n")
            
            for col in columns:
                col_name, data_type, max_len, nullable, default = col
                max_len_str = str(max_len) if max_len is not None else ""
                f.write(f"{col_name:<30} {data_type:<20} {max_len_str:<10} {nullable:<10} {str(default):<20}\n")
        
        cursor.close()
        conn.close()
        print(f"导出表结构: {schema_name}.{table_name} -> {output_path}")
        return True
    
    except Exception as e:
        print(f"错误: 导出表结构 {table_name} 时出错: {str(e)}")
        if node_info["resource_type"] != "source":
            with open(output_path.with_suffix('.error_info'), 'w') as f:
                f.write(f"Error exporting table schema {table_name}: {str(e)}\n")
                f.write(f"This might be a temporary table, an ephemeral model, or there might be connection issues.")
        return False

def export_sql_chain(graph, node_chain, output_path):
    """导出完整的SQL依赖链（包括源表）"""
    output_path.mkdir(exist_ok=True, parents=True)
    
    for i, node_id in enumerate(node_chain):
        node = graph.nodes[node_id]
        
        # 处理源表节点
        if node["resource_type"] == "source":
            source_info = {
                "source_name": node["source_name"],
                "table_name": node["name"],
                "identifier": node["identifier"],
                "definition_file": node["original_file_path"]
            }
            
            step_output = output_path / f"step_{i+1}_source_{node['name']}.json"
            with open(step_output, 'w') as f:
                json.dump(source_info, f, indent=2)
            print(f"导出源表信息: {node['name']} -> {step_output}")
            continue
        
        # 处理模型节点
        model_name = node["name"]
        sql_path = find_sql_file(model_name)
        
        if sql_path and sql_path.exists():
            step_output = output_path / f"step_{i+1}_{model_name}.sql"
            shutil.copy(sql_path, step_output)
            print(f"导出SQL: {model_name} -> {step_output}")
        else:
            print(f"警告: 找不到模型 {model_name} 的SQL文件")
            if "original_file_path" in node:
                original_path = project_dir / node["original_file_path"]
                if original_path.exists():
                    step_output = output_path / f"step_{i+1}_{model_name}.sql"
                    shutil.copy(original_path, step_output)
                    print(f"使用原始文件代替: {original_path} -> {step_output}")

# 增强判定join
def has_join_in_sql(sql_path):
    """检查SQL文件中是否存在JOIN操作（考虑宏定义和宏调用）"""
    if not sql_path or not sql_path.exists():
        return False
    
    try:
        with open(sql_path, 'r') as f:
            sql_content = f.read()
        
        # 预处理SQL内容：移除注释
        sql_content = re.sub(r'--.*?$', '', sql_content, flags=re.MULTILINE)  # 移除单行注释
        sql_content = re.sub(r'/\*.*?\*/', '', sql_content, flags=re.DOTALL)  # 移除多行注释
        
        # 移除配置宏
        sql_content = re.sub(r'\{\{\s*config\([^}]*\)\s*\}\}', '', sql_content, flags=re.IGNORECASE)
        
        # 处理其他宏调用
        def replace_macro(match):
            macro_content = match.group(1)
            # 处理 ref() 宏
            if 'ref(' in macro_content.lower():
                ref_match = re.search(r'ref\(([^)]+)\)', macro_content, re.IGNORECASE)
                if ref_match:
                    return ref_match.group(1)
            # 处理其他宏，保留可能包含表引用的内容
            return macro_content
        
        sql_content = re.sub(r'\{\{\s*([^}]*)\s*\}\}', replace_macro, sql_content)
        
        # 转换为小写以便于模式匹配
        sql_content_lower = sql_content.lower()
        
        # 检查是否存在JOIN操作
        join_pattern = re.compile(
            r'\bjoin\b|\binner join\b|\bleft join\b|\bright join\b|\bfull join\b|\bcross join\b',
            re.IGNORECASE
        )
        
        has_join = join_pattern.search(sql_content_lower) is not None
        
        # 如果没有明确的JOIN关键字，检查是否有多个表引用
        if not has_join:
            # 查找FROM子句中的表引用
            from_pattern = re.compile(r'\bfrom\s+([^\(\)\s,]+(?:\s*,\s*[^\(\)\s,]+)*)', re.IGNORECASE)
            from_matches = from_pattern.findall(sql_content_lower)
            
            # 检查是否有多个表引用（逗号分隔的隐式JOIN）
            for from_match in from_matches:
                tables = [table.strip() for table in from_match.split(',')]
                if len(tables) > 1:
                    has_join = True
                    break
        
        return has_join
    
    except Exception as e:
        print(f"警告: 读取SQL文件 {sql_path} 时出错: {str(e)}")
        return False

def is_single_dependency(graph, node_id):
    """检查节点是否只有一个依赖"""
    parents = list(graph.predecessors(node_id))
    return len(parents) == 1

def has_multiple_dependencies(graph, node_id):
    """检查节点是否有多个依赖"""
    parents = list(graph.predecessors(node_id))
    return len(parents) > 1

def get_node_name(graph, node_id):
    """获取节点名称"""
    return graph.nodes[node_id]["name"]

def get_direct_upstream_nodes(graph, node_id):
    """获取直接上游节点"""
    return list(graph.predecessors(node_id))

def get_direct_downstream_nodes(graph, node_id):
    """获取直接下游节点"""
    return list(graph.successors(node_id))

def find_single_child_downstream_nodes(graph, purple_node_id):
    """找到紫色表的单依赖下游节点"""
    result = []
    
    def dfs(current_id, path):
        successors = list(graph.successors(current_id))
        
        if not successors:
            if len(path) > 1 and current_id != purple_node_id:
                result.append(current_id)
            return
        
        for child_id in successors:
            if is_single_dependency(graph, child_id):
                dfs(child_id, path + [child_id])
            else:
                if current_id != purple_node_id and len(path) > 1:
                    result.append(current_id)
    
    dfs(purple_node_id, [purple_node_id])
    
    return list(set(result))

# 区分多依赖和含有join条件的先后顺序 先看距离紫色节点最近的上游多依赖节点，再看这个节点是否是join节点，如果不是join节点，则删除这个文件夹返回
def find_blue_nodes(graph, purple_node_id):
    """找到蓝色节点（距离紫色节点最近的JOIN操作的直接依赖）"""
    purple_deps = get_direct_upstream_nodes(graph, purple_node_id)
    
    # 如果是单依赖上游节点，继续向上查找
    if len(purple_deps) == 1:
        return find_blue_nodes(graph, purple_deps[0])
    
    # 检查紫色节点本身是否是多依赖节点，且检查是否是join节点
    else:
        purple_node = graph.nodes[purple_node_id]
        purple_table = purple_node["name"]
        purple_sql_path = find_sql_file(purple_table)
        
        if purple_sql_path.exists() and has_join_in_sql(purple_sql_path):
            return purple_deps
        else:
            print(f"警告: 紫色节点 {purple_table} 不是JOIN节点，删除该紫色表的输出目录")
            return []
    

def find_blue_nodes_single_dependency_upstream(graph, blue_nodes):
    """找到蓝色节点的入度是1的最上游节点"""
    result = {}
    
    for blue_node_id in blue_nodes:
        
        pred_id = blue_node_id
        single_dependency_upstream = None

        # 循环找到蓝色节点入度为1的最上游节点
        while graph.in_degree(pred_id) == 1:
            pred_id = list(graph.predecessors(pred_id))[0]

        single_dependency_upstream = pred_id if pred_id != blue_node_id else None
        
        if single_dependency_upstream:
            result[blue_node_id] = single_dependency_upstream
    
    return result

def find_path_to_node(graph, start_id, target_id):
    """找到从起始节点到目标节点的路径"""
    try:
        return nx.shortest_path(graph, start_id, target_id)
    except nx.NetworkXNoPath:
        return None

def find_nearest_join_node(graph, purple_node_id):
    """找到距离紫色节点最近的JOIN节点"""
    purple_deps = get_direct_upstream_nodes(graph, purple_node_id)
    
    # 如果是单依赖上游节点，继续向上查找
    if len(purple_deps) == 1:
        return find_nearest_join_node(graph, purple_deps[0])
    
    # 检查紫色节点本身是否是多依赖节点，且检查是否是join节点
    else:
        purple_node = graph.nodes[purple_node_id]
        purple_table = purple_node["name"]
        purple_sql_path = find_sql_file(purple_table)
        
        if purple_sql_path.exists() and has_join_in_sql(purple_sql_path):
            return purple_node_id
        else:
            print(f"警告: 紫色节点 {purple_table} 不是JOIN节点，删除该紫色表的输出目录")
            return []


def process_purple_table(manifest, graph, purple_table):
    """处理单个紫色表，按照新的要求分析依赖关系"""
    output_base_dir = output_dir / purple_table
    tables_dir = output_base_dir / "tables"
    schemas_dir = output_base_dir / "schemas"
    sql_blue_to_downstream_dir = output_base_dir / "sql" / "upstream_to_blue"
    sql_blue_to_purple_dir = output_base_dir / "sql" / "blue_to_purple"
    sql_purple_to_upstream_dir = output_base_dir / "sql" / "purple_to_downstream"
    
    for dir_path in [tables_dir, schemas_dir, sql_blue_to_downstream_dir, 
                    sql_blue_to_purple_dir, sql_purple_to_upstream_dir]:
        dir_path.mkdir(exist_ok=True, parents=True)
    
    purple_node_id = find_node_id(manifest, purple_table)
    if not purple_node_id:
        print(f"找不到紫色表: {purple_table}")
        return False
    
    # 1. 找到紫色表的单依赖下游节点
    single_dep_downstream_nodes = find_single_child_downstream_nodes(graph, purple_node_id)
    print(f"\n紫色表 {purple_table} 的单依赖下游节点:")
    single_dep_downstream_tables = []
    for node_id in single_dep_downstream_nodes:
        table_name = get_node_name(graph, node_id)
        single_dep_downstream_tables.append(table_name)
        print(f"  - {table_name}")
    
    # 2. 找到蓝色节点
    blue_node_ids = find_blue_nodes(graph, purple_node_id)
    if not blue_node_ids:
        print(f"警告: 紫色表 {purple_table} 没有找到蓝色节点，删除该紫色表的输出目录")
        shutil.rmtree(output_base_dir)
        return False
    
    print(f"\n紫色表 {purple_table} 的蓝色节点:")
    blue_tables = []
    for node_id in blue_node_ids:
        table_name = get_node_name(graph, node_id)
        blue_tables.append(table_name)
        print(f"  - {table_name}")
    
    # 3. 找到蓝色节点的单依赖上游节点
    blue_upstream_dict = find_blue_nodes_single_dependency_upstream(graph, blue_node_ids)
    print(f"\n蓝色节点的单依赖上游节点:")
    blue_upstream_tables = []
    for blue_id, upstream_ids in blue_upstream_dict.items():
        blue_name = get_node_name(graph, blue_id)
        upstream_name = get_node_name(graph, upstream_ids)
        blue_upstream_tables.append(upstream_name)
        print(f"  - {blue_name} -> {upstream_name}")
    
    # 导出所有相关表格数据
    all_tables = set([purple_table] + single_dep_downstream_tables + blue_tables + blue_upstream_tables)
    exported_tables = []
    skipped_tables = []
    
    for table in all_tables:
        # 首先需要获取节点的完整信息
        node_id = find_node_id(manifest, table)
        if node_id:
            node_info = graph.nodes[node_id]
            if export_table_data(node_info, tables_dir / f"{table}.csv"):
                exported_tables.append(table)
                export_table_schema(node_info, schemas_dir / f"{table}.schema")
            else:
                skipped_tables.append(table)
            
    print(f"\n导出表数据完成: {len(exported_tables)} 个成功, {len(skipped_tables)} 个跳过")
    print(skipped_tables)


    # 导出SQL转换链
    
    # 1. 蓝色节点单依赖上游节点到蓝色节点的SQL链
    print("\n导出蓝色节点单依赖上游节点到蓝色节点的SQL链:")
    for blue_id, upstream_ids in blue_upstream_dict.items():
        blue_name = get_node_name(graph, blue_id)
        upstream_name = get_node_name(graph, upstream_ids)
        path = find_path_to_node(graph, upstream_ids, blue_id)
        if path:
            chain_name = f"{upstream_name}-to-{blue_name}"
            export_sql_chain(graph, path, sql_blue_to_downstream_dir / chain_name)
    
    # 2. 多个蓝色节点转换成紫色表的SQL链
    print("\n导出多个蓝色节点转换成紫色表的SQL链:")
    join_node_id = find_nearest_join_node(graph, purple_node_id)
    if join_node_id:
        # 创建子目录
        blue_to_join_dir = sql_blue_to_purple_dir / "blue_to_join"
        join_to_purple_dir = sql_blue_to_purple_dir / "join_to_purple"
        blue_to_join_dir.mkdir(exist_ok=True, parents=True)
        join_to_purple_dir.mkdir(exist_ok=True, parents=True)

         # 删除旧的 blue_to_purple 目录（如果存在）
        old_blue_to_purple_dir = sql_blue_to_purple_dir / "blue_to_purple"
        print(f"检查旧目录是否存在: {old_blue_to_purple_dir}")
        print(f"旧目录存在: {old_blue_to_purple_dir.exists()}")
        if old_blue_to_purple_dir.exists():
            shutil.rmtree(old_blue_to_purple_dir)
            print(f"已删除旧目录: {old_blue_to_purple_dir}")

        # 找到从蓝色节点到JOIN节点的路径
        blue_to_join_paths = {}
        for blue_id in blue_node_ids:
            path = find_path_to_node(graph, blue_id, join_node_id)
            if path:
                blue_to_join_paths[blue_id] = path

        # 合并所有路径
        all_paths = []
        for blue_id, path in blue_to_join_paths.items():
            all_paths.extend(path[:-1])  # 排除重复的JOIN节点
        
        unique_path = list(dict.fromkeys(all_paths))  # 去重保持顺序
        chain_name = f"blue_to_join"
        export_sql_chain(graph, unique_path, blue_to_join_dir)
        
        join_to_purple_path = find_path_to_node(graph, join_node_id, purple_node_id)
            
        if join_to_purple_path:
            export_sql_chain(graph, join_to_purple_path, join_to_purple_dir)
        else:
            with open(join_to_purple_dir / "no_join_to_purple_note.txt", 'w') as f:
                f.write(f"紫色节点 {purple_table} 本身就是JOIN节点，因此没有单独的join_to_purple转换链。")
            print(f"紫色节点 {purple_table} 是JOIN节点，跳过导出join_to_purple转换链")
    
    
    # 3. 紫色表到紫色表单依赖下游节点的SQL链
    print("\n导出紫色表到紫色表单依赖下游节点的SQL链:")
    for downstream_id in single_dep_downstream_nodes:
        downstream_name = get_node_name(graph, downstream_id)
        path = find_path_to_node(graph, purple_node_id, downstream_id)
        if path:
            chain_name = f"{purple_table}_to_{downstream_name}"
            export_sql_chain(graph, path, sql_purple_to_upstream_dir / chain_name)
    
    return True

def main():
    # if output_dir.exists():
    #     shutil.rmtree(output_dir)
    # output_dir.mkdir(exist_ok=True, parents=True)
    
    try:
        manifest = load_manifest()
        graph = build_dependency_graph(manifest)
        print(f"依赖图构建完成: {graph.number_of_nodes()} 个节点, {graph.number_of_edges()} 条边")
    except Exception as e:
        print(f"加载manifest失败: {e}")
        return
    
    purple_tables = get_purple_tables(manifest, graph)
    if not purple_tables:
        print(f"无符合条件的紫色表，退出")
        return
    
    print(f"找到选择表: {', '.join(purple_tables)}")

    # 导出所有source表和对应的schema
    source_data_dir = output_dir / "datalake_sources"
    source_data_dir.mkdir(exist_ok=True, parents=True)
    source_table_dir = source_data_dir / "tables"
    source_table_dir.mkdir(exist_ok=True, parents=True)
    source_schema_dir = source_data_dir / "schemas"
    source_schema_dir.mkdir(exist_ok=True, parents=True)

    for source_id, source_data in manifest["sources"].items():
        node_info = {
            "name": source_data["name"],
            "resource_type": "source",
            "source_name": source_data["source_name"],
            "identifier": source_data.get("identifier", source_data["name"]),
            "original_file_path": source_data["original_file_path"]
        }
        export_table_data(node_info, source_table_dir / f"{source_data['name']}.csv")
        export_table_schema(node_info, source_schema_dir / f"{source_data['name']}.schema")

    # 导出紫色表相关的数据 （query准备)
    successful_tables = []
    target_table = "hubspot__daily_ticket_history"  # 指定目标表名

    for purple_table in purple_tables:
        # # 只处理目标表，其他表跳过
        # if purple_table != target_table:
        #     continue

        print(f"\n{'='*50}")
        print(f"处理紫色表: {purple_table}")
        print(f"{'='*50}")
        if process_purple_table(manifest, graph, purple_table):
            successful_tables.append(purple_table)
    
    print("\n" + "="*50)
    print(f"处理完成! 成功处理的紫色表数量: {len(successful_tables)}/{len(purple_tables)}")
    print(f"成功处理的表列表: {', '.join(successful_tables) if successful_tables else '无'}")
    print("="*50)

if __name__ == "__main__":
    main()