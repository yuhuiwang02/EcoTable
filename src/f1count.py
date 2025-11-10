import os
import json
import sys
from datetime import datetime
from f1count_llmjudge import Col_Judge
import csv
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
log_dir = "/data/new_dbt_dataset/111test/f1logs"
os.makedirs(log_dir, exist_ok=True)
log_file = os.path.join(log_dir, f"run_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log")

# 打开日志文件（追加模式）
log = open(log_file, "a", encoding="utf-8")

# 保存原始输出流
stdout_orig = sys.stdout
stderr_orig = sys.stderr

class Logger:
    def __init__(self, *streams):
        self.streams = streams
    def write(self, message):
        for s in self.streams:
            s.write(message)
            s.flush()
    def flush(self):
        for s in self.streams:
            s.flush()

# 同时输出到屏幕和文件
sys.stdout = Logger(sys.stdout, log)
sys.stderr = Logger(sys.stderr, log)

print(f"[INFO] 日志记录开始：{log_file}")
def process_conditions(conditions, csv_path):
    # 存放 col1 和 col2 的列表
    table1col = []
    table2col = []

    # 处理每个条件
    for condition in conditions:
        # 使用 AND 分割条件
        for cond in condition.split(' AND '):
            # 找到格式 'table1.col1 = table2.col2' 的部分
            parts = cond.split('=')
            if len(parts) == 2:
                left = parts[0].strip()  # 'table1.col1'
                right = parts[1].strip()  # 'table2.col2'

                # 拆分为 table1, col1, table2, col2
                table1, col1 = left.split('.')
                table2, col2 = right.split('.')

                # 添加到对应的 list 中
                table1col.append(col1)
                table2col.append(col2)

                # 检查是否在 CSV 中存在
                # 对于 table1col，检查 table1 表的列 col1
                if not check_column_exists(csv_path, table1, col1):
                    print(f"{col1} not found in {table1} table.")
                
                # 对于 table2col，检查 table2 表的列 col2
                if not check_column_exists(csv_path, table2, col2):
                    print(f"{col2} not found in {table2} table.")
    
    return table1col, table2col

def check_column_exists(csv_path, table_name, column_name):
    # 假设每个 CSV 文件的结构是以表名为文件名，且第一行是列名
    table_csv_path = os.path.join(csv_path, f"{table_name}.csv")
    
    if os.path.exists(table_csv_path):
        # 打开 CSV 文件并检查列名
        with open(table_csv_path, 'r') as file:
            reader = csv.reader(file)
            headers = next(reader)  # 读取第一行作为列名
            if column_name in headers:
                return True
    return False

def process_single_file(projectname, table_names, mapping_info, edge_dir):
    """处理单个文件的全流程"""
    # 提取 green_to_blue 映射
    
    csv_path = f"/data/new_dbt_dataset/111test/csv/{projectname}"
    blue_to_green = {item.split("-to-")[1]:item.split("-to-")[0] for item in mapping_info["green_to_blue"]}
    green_to_blue = {item.split("-to-")[0]:item.split("-to-")[1] for item in mapping_info["green_to_blue"]}
    json_folder=f"/data/new_dbt_dataset/111test/f1judge/{projectname}"
    os.makedirs(json_folder,exist_ok=True)
    if table_names[0] not in blue_to_green.values() or table_names[1] not in blue_to_green.values():
        return False
    # print(blue_to_green)
    edges_info=mapping_info["edges"]
    # 读取现有的 edge.json 文件，如果存在
    edge_filename = f"{table_names[0]}-{table_names[1]}.json"  # 根据实际需求，这里以table_names作为示例
    edge_filepath = os.path.join(edge_dir, projectname, edge_filename)
    table1_col=[]
    table2_col=[]
    is_correct = False
    conditions=[]
    if os.path.exists(edge_filepath):
        with open(edge_filepath, "r") as edge_file:
            edge_data = json.load(edge_file)
        if isinstance(edge_data,list):
            edge_data=edge_data[0]
        candidates = edge_data.get("candidates", {})
        
        # 只处理 candidates 中值为 True 的项
        for col_pair, is_true in candidates.items():
            if is_true:
                col1, col2 = col_pair.split("↔")
                col1 = col1.strip()
                col2 = col2.strip()
                if edge_data.get("table1")==table_names[1]:
                    col1,col2=col2,col1
                table1_col.append(col1)
                table2_col.append(col2)
                # 在这里执行您需要的后续操作
                # 例如，输出匹配的列名
                if not check_column_exists(csv_path, table_names[0], col1):
                    print(f"{col1} not found in {table_names[0]} table.")
                
                # 对于 table2col，检查 table2 表的列 col2
                if not check_column_exists(csv_path, table_names[1], col2):
                    print(f"{col2} not found in {table_names[1]} table.")
            # 建议：如果 table_names 是列表，先转成 set 提升 contains 判断效率
# table_names = set(table_names)

        for edge in edges_info:  # 假设你的边集合叫 edges
            s_name = blue_to_green.get(edge.get("source"))
            t_name = blue_to_green.get(edge.get("target"))

            # 先检查映射是否存在
            missing = []
            if not s_name:
                missing.append(f'{edge.get("source")} not found in blue_to_green')
            if not t_name:
                missing.append(f'{edge.get("target")} not found in blue_to_green')
            if missing:
                # print("; ".join(missing))
                continue  # 没有映射就不再往下判定
            
            # 两侧都映射成功且在 table_names -> 收集 condition
            if s_name in table_names and t_name in table_names:
                conditions.append(edge["condition"])
        if not conditions:
            print("can't find condition")
            return False
        if len(table2_col)>0 or len(table1_col)>0:
            print("-------------------------------------------------------------------------------")
            print(f"project:{projectname},Table1: {table_names[0]}, Col1: {table1_col}, Table2: {table_names[1]}, Col2: {table2_col}") 
            print(conditions)
            # print(blue_to_green)
            print(green_to_blue)
            # print(conditions)
            json_dir = f"{json_folder}/{table_names[0]}-{green_to_blue[table_names[0]]}---{table_names[1]}-{green_to_blue[table_names[1]]}.json"
            if not os.path.isfile(json_dir):
                tables={
                    "a1":table_names[0],
                    "b1":table_names[1],
                    "a2":green_to_blue.get(table_names[0]),
                    "b2":green_to_blue.get(table_names[1])
                }
                cols1,cols2=process_conditions(conditions,csv_path)
                cols={
                    table_names[0]:table1_col,
                    table_names[1]:table2_col,
                    green_to_blue.get(table_names[0]):cols1,
                    green_to_blue.get(table_names[1]):cols2,
                }
                print(table_names[0])
                print(green_to_blue.get(table_names[0]))
                Col_Judge(csv_path,tables,cols,json_dir)
            with open(json_dir, 'r', encoding='utf-8') as file:
                now_data = json.load(file)
            return now_data.get("is_correct")
    else:
        print(f"File {edge_filepath} does not exist!")
    return is_correct

def normalize_edge(edge):
    """标准化边的顺序，确保无论是1-2还是2-1都视为相同"""
    id1, id2 = int(edge['id1']), int(edge['id2'])
    a, b = sorted((id1, id2))
    return f"{a}-{b}"


def compare_edges_and_paths(edges, join_paths):
    """比较实验结果和ground truth，计算f1分数"""
    # 生成标准化后的边集
    alle={normalize_edge(edge) for edge in edges}
    predicted_edges = {
    normalize_edge(edge) if edge["is_correct"] else f"{placeholder_counter}-{placeholder_counter}"
    for edge in edges
    if not edge["is_correct"] and (placeholder_counter := placeholder_counter + 1) or edge["is_correct"]
    }
    ground_truth_paths = {normalize_edge({'id1': join_path.split('-')[0], 'id2': join_path.split('-')[1]})
                          for join_path in join_paths}
    # 计算指标
    all_edges = predicted_edges.union(ground_truth_paths)
    y_true = [1 if edge in ground_truth_paths else 0 for edge in all_edges]
    y_pred = [1 if edge in predicted_edges else 0 for edge in all_edges]
    print(predicted_edges)
    print(ground_truth_paths)
    print(alle)
    # 计算 f1
    f1 = f1_score(y_true, y_pred, zero_division=0)
    return f1

def process_project_folder(project_folder, mapping_json, edge_dir):
    """处理整个项目文件夹中的所有文件"""
    # 遍历 project_folder 中的所有子文件夹（即 project 文件夹）
    f1_statistics = []
    for item in os.listdir(project_folder):
        item_path = os.path.join(project_folder, item)
        # 确保当前项是子文件夹
        if os.path.isdir(item_path):
            # 遍历子文件夹中的文件
            f1_values = []
            for file in os.listdir(item_path):
                file_path = os.path.join(item_path, file)
                
                # 只处理符合条件的文件
                if file.endswith(".json"): 
                    # 获取 projectname 和 num
                    projectname = os.path.basename(item_path)
                    conbination = file.split(".")[0]
                    print(conbination)
                    # 加载 JSON 文件
                    with open(file_path, "r") as f:
                        json_data = json.load(f)
                      # 用于存储 f1 值
                    # 获取 table_name 和 gt (nodes 和 edges)
                    table_names = json_data.get("table_name", [])
                    gt = json_data.get("gt", {})
                    edges = gt.get("edges", [])

                    # 获取对应的映射关系
                    if f"{conbination}" in mapping_json:
                        mapping_info = mapping_json[conbination]
                        
                        # 对每个 edge 进行单独处理
                        for edge in edges:
                            id1 = edge["id1"]
                            id2 = edge["id2"]
                            table1 = table_names[id1]
                            table2 = table_names[id2]

                            # 处理单个文件的逻辑
                            edge["is_correct"]=process_single_file(projectname, [table1, table2], mapping_info, edge_dir)
                        join_paths = json_data.get("join_path", [])
                        f1 = compare_edges_and_paths(edges, join_paths)
                        f1_values.append(f1)

                    else:
                        print(f"{conbination} not found")
                # 计算当前子文件夹的平均 f1
            if f1_values:
                avg_f1 = sum(f1_values) / len(f1_values)
                f1_statistics.append({
                    'folder': item,
                    'total_f1_count': len(f1_values),
                    'avg_f1': avg_f1
                })
                print({
                    'folder': item,
                    'total_f1_count': len(f1_values),
                    'avg_f1': avg_f1
                })
    # 将所有统计信息写入 f1count.csv
    with open('f1count_top70.csv', mode='w', newline='') as csv_file:
        fieldnames = ['folder', 'total_f1_count', 'avg_f1']
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)

        writer.writeheader()
        for stat in f1_statistics:
            writer.writerow(stat)

                                    
nowdata="top70"

if __name__ == "__main__":
    project_folder = f"/data/new_dbt_dataset/111test/outputset/{nowdata}"  # 输入的文件夹路径
    edge_dir = "/data/new_dbt_dataset/111test/LLMjudgeair"  # edge 文件输出目录
    mapping_json_file = "/data/new_dbt_dataset/111test/dbt_dataset/final_items.json"  # 映射 JSON 文件路径
    
    # 读取映射 JSON 文件
    with open(mapping_json_file, "r") as f:
        mapping_json = json.load(f)
    
    # 处理项目文件夹
    process_project_folder(project_folder, mapping_json, edge_dir)
