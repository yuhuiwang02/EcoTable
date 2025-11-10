import json
import math
from JoinClassifier import JoinClassify, Table
import pandas as pd
from pathlib import Path
import os
import shutil
from typing import Dict, List, Tuple,Any
from tree_process import steiner_tree_2approx
from LLM_Judge import LLM_Judge
from Edges_Judge_normal import Edges_Judge
from check_edges import check_edges_from_file

import sys
from datetime import datetime

log_dir = "./newlogs"
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

def get_weight(graph, a, b):
    for neighbor, weight in graph.get(a, []):
        if neighbor == b:
            return weight
    return None
model="/data/new_dbt_dataset/111test/models/v2"


def update_edge_min(g, u, v, w):
    g.setdefault(u, {})
    g.setdefault(v, {})
    if v in g[u]:
        g[u][v] = min(g[u][v], w)
        g[v][u] = g[u][v]    # 保持对称
    else:
        g[u][v] = w
        g[v][u] = w

def process_project(json_data, project_name,tables_dir,num):
    """
    处理指定项目：
    1) 从 table_relevance 提取表名
    2) 仅移动同名 .csv 文件到 {tables_dir}/{project_name}/（顶层，覆盖同名）
    3) 加载表并计算连接概率
    """
    try:
        # with open(json_path, 'r', encoding='utf-8') as file:
        #     json_data = json.load(file)

        projects = json_data.get(project_name)
        if not projects or not isinstance(projects, list):
            raise ValueError(f"未找到项目 {project_name} 或数据格式错误")

        project = projects[num]
        table_relevance = project.get("table_relevance", {})
        ground_truth = project.get("ground_truth", [])
        join_path = []
        ground_truth_edges = project.get("edge_relation",{})
        query_related_table = project.get("query_related_table",[])
        purple_name=project.get("final_table")
        query=project.get("query")
        table_names = list(table_relevance.keys())
        print(f"提取的表名: {table_names}")

        # ① 移动顶层 CSV 文件
        dest_dir = tables_dir #_move_related_tables(tables_dir, project_name, table_names)
        project_json=f"/data/new_dbt_dataset/111test/project_json/{project_name}.json"
        # ② 加载并计算连接概率
        graph = {}
        table_name_to_index = {name: idx for idx, name in enumerate(table_names)}
        for t1, t2, _, _ in ground_truth_edges:
            join_path.append(f"{table_name_to_index[t1]}-{table_name_to_index[t2]}")
        if not os.path.isfile(project_json):
            JoinClassify(mode="predict",tables=dest_dir,output=project_json,model_dir=model,top_k=6)
        with open(project_json, "r", encoding="utf-8") as f:
            join_specs = json.load(f)

        if isinstance(join_specs, dict):
            join_specs = [join_specs]

        temp_graph = {}

        for spec in join_specs:
            t1, t2 = spec.get("table1"), spec.get("table2")
            prob = spec.get("prob", 0)
            col=spec.get("column1")
            # 确保数据有效
            if not isinstance(prob, (int, float)) or prob <= 0:
                continue
            if t1 not in table_name_to_index or t2 not in table_name_to_index:
                continue

            n1, n2 = table_name_to_index[t1], table_name_to_index[t2]
            weight = -math.log(prob)# + 0.001           
            # 双向更新
            update_edge_min(temp_graph, n1, n2, weight)

        # 转换为你需要的最终形式
        graph = {u: list(v.items()) for u, v in temp_graph.items()}

        ground_truth_indices = {n: table_name_to_index[n] for n in ground_truth if n in table_name_to_index}
        query_related_table_indices = {n: table_name_to_index[n] for n in query_related_table if n in table_name_to_index}
        query_related_table_index = list(query_related_table_indices.values())
        ground_truth_index = list(ground_truth_indices.values())

        return table_name_to_index,table_names, graph, ground_truth_index,query_related_table_index,project_json,join_path,query,purple_name

    except Exception as e:
        print(f"处理 {project_name} 时出错: {e}")
        return {}, {}, {},{},""
    
# 示例：输入 JSON 文件路径
file_path = "/data/new_dbt_dataset/111test/dataair.json"  # JSON 文件路径
project_list = ["google_ads", "snapchat_ads", "tiktok_ads", "microsoft_ads", "apple_search_ads", "amazon_ads", "facebook_pages", "instagram_business"] # 你需要统计的项目列表
project_list1=["mailchimp","pardot","pendo","zendesk"]
project_list2=["xero","amazon_selling_partner"]
project_list3=["github","asana","fivetran_log"]
top_k=f"withoutllm"
model_dir = "/data/new_dbt_dataset/111test/models/v2"  # 模型路径
output_judge = "/data/new_dbt_dataset/111test/LLMjudgeair"
# 获取并处理指定项目的数据
with open(file_path, 'r', encoding='utf-8') as file:
    json_data = json.load(file)
#project_names = [k for k, v in json_data.items() if isinstance(v, list)]



#print(f"[INFO] 共发现 {len(project_names)} 个项目: {project_names}")
#for k in project_names:
results = []
acc_avg = 0
for project_name in project_list3:
    output_judge_now=f"{output_judge}/{project_name}"
    os.makedirs(output_judge_now,exist_ok=True)
    file_path=f'/data/new_dbt_dataset/111test/dataair.json'
    tables_dir = f"/data/new_dbt_dataset/111test/csv/{project_name}"  # 表数据目录路径
    output_dir = f"/data/new_dbt_dataset/111test/outputset/{top_k}/{project_name}"
    os.makedirs(output_dir,exist_ok=True)
    with open(file_path, 'r', encoding='utf-8') as file:
        json_data = json.load(file)
    nums=len(json_data.get(project_name))
    for num in range(nums):
        table_index,table_name, graph, ground_truth,query_related,json_path,join_path,query,purple_table= process_project(json_data, project_name, tables_dir, num)
        Edgelegal=True
        count = 0
        node_related = "\n".join(
            [f"{table_name[i]}"
                for i in query_related]
            )
        gt = steiner_tree_2approx(graph,query_related)
        while not Edgelegal:
            usable = True
            while not usable:
                gt = steiner_tree_2approx(graph,query_related)
                edges=gt.get("edges")
                nodes=gt.get("nodes")
                usable = True
                for edge in edges:
                    a,b=edge.get("id1"),edge.get("id2")
                    if a>b:
                        a,b=b,a
                    json_dir=f"{output_judge_now}/{table_name[a]}-{table_name[b]}.json"
                    if not os.path.isfile(json_dir):
                        LLM_Judge(json_path,tables_dir,table_name[a],table_name[b],purple_table,query,json_dir)
                    with open(json_dir, 'r', encoding='utf-8') as file:
                        now_data = json.load(file)
                    if isinstance(now_data,list):
                        join_type = now_data[0].get("is_join")
                    else:
                        join_type = now_data.get("is_join")
                    if join_type == "False":
                        graph[a] = [(n, w) for n, w in graph[a] if n != b]
                        graph[b] = [(n, w) for n, w in graph[b] if n != a]
                        usable = False
                        break
                    else:
                        graph[a] = [(n, 0 if n == b else w) for n, w in graph[a]]
                        graph[b] = [(n, 0 if n == a else w) for n, w in graph[b]]
                        
            
            gt = steiner_tree_2approx(graph,query_related)

            # quan

            edges=gt.get("edges")
            edges_set=[]
            node_set=[]
            for edge in edges:
                    a,b=edge.get("id1"),edge.get("id2")
                    if a > b:
                        continue
                    t1,t2=table_name[a],table_name[b]
                    if t1 not in node_set:
                        node_set.append(t1)
                    if t2 not in node_set:
                        node_set.append(t2)
                    json_dir=f"{output_judge_now}/{t1}-{t2}.json"
                    with open(json_dir, 'r', encoding='utf-8') as file:
                        now_data = json.load(file)
                    if isinstance(now_data,list):
                        reason = now_data[0].get("other_notes")
                    else:
                        reason = now_data.get("other_notes")
                    edges_set.append(f"{t1}↔{t2}:{reason}")
            node_text = ",".join(
            [f"{node}"
                for node in node_set]
            )
            print(node_text)
            output_judge_now_query=f"{output_judge_now}/{num}"
            os.makedirs(output_judge_now_query,exist_ok=True)
            json_dir=f"{output_judge_now_query}/{top_k}{project_name}-{num}-{count}.json"
            Edges_Judge(tables_dir,edges_set,purple_table,query,node_text,node_related,json_dir)
            with open(json_dir, "r", encoding="utf-8") as f:
                data = json.load(f)
            excluded_edges = data.get("excluded_edges", [])
            if not excluded_edges:
                Edgelegal=True
                break
            for edge in excluded_edges:
                t1, t2 = edge.split("↔")
                a,b=table_index[t1],table_index[t2]
                if get_weight(graph,a,b) is None:
                    Edgelegal=True
                    break
                graph[a] = [(n, w) for n, w in graph[a] if n != b]
                graph[b] = [(n, w) for n, w in graph[b] if n != a]
            count+=1
            
            



        output_data = {
            "table_name": table_name,
            "gt": gt,
            "join_path":join_path,
            "ground_truth": ground_truth,
            "query_related":query_related
        }

        # 生成文件路径
        idx=1
        output_path = os.path.join(output_dir, f"{purple_table}__combination_{idx}.json")
        while os.path.isfile(output_path):
            idx+=1
            output_path = os.path.join(output_dir, f"{purple_table}__combination_{idx}.json")
        # 写入 JSON 文件
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(output_data, f, ensure_ascii=False, indent=4)
        result_text=check_edges_from_file(output_path)
        acc_avg+=float(result_text["accuracy"])
        print(result_text["accuracy"])
        results.append(result_text)
    acc_path=os.path.join(output_dir, f"{project_name}-acc.json")
    acc_avg=acc_avg/nums
    print(f"All query processed,the avg_acc is {acc_avg}.")



def run_join_graph_builder(
    *,
    dataset_json_path: str,      # 原 file_path：总数据 JSON
    csv_root_dir: str,           # 原 tables_dir 根：<csv_root_dir>/<project_name>/ 下放各项目 CSV
    project_json_root: str,      # JoinClassify 输出（每项目一个 json）目录（原硬编码 /project_json/）
    model_dir: str,              # 原 model_dir：JoinClassify 模型目录
    outputset_root_dir: str,     # 原 output_dir 根：最终组合 JSON 的输出根目录
    llm_judge_root: str,         # 原 output_judge 根：两个大模型检测的中间产物会写到这里
    log_dir: str = "./newlogs"   # 日志目录（与原脚本一致）
) -> None:
    """
    将原脚本“主流程”改为接口函数：
    - 完整保留两段大模型检测（LLM_Judge 与 Edges_Judge）与其文件读写逻辑
    - 不改变任何生成 JSON 的格式或文件命名
    - 不返回值
    """
    # ---------- 日志，保持与原脚本相同的“同时输出到屏幕和文件” ----------
    os.makedirs(log_dir, exist_ok=True)
    log_file = os.path.join(log_dir, f"run_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log")
    log = open(log_file, "a", encoding="utf-8")

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

    stdout_orig, stderr_orig = sys.stdout, sys.stderr
    sys.stdout = Logger(sys.stdout, log)
    sys.stderr = Logger(sys.stderr, log)

    try:
        print(f"[INFO] 日志记录开始：{log_file}")

        # ---------- 固定默认，与原脚本一致 ----------
        top_k = "withoutllm"
        project_list3 = ["github", "asana", "fivetran_log"]

        # ---------- 读取数据 ----------
        if not os.path.isfile(dataset_json_path):
            raise FileNotFoundError(f"dataset_json_path 不存在: {dataset_json_path}")
        with open(dataset_json_path, "r", encoding="utf-8") as f:
            json_data = json.load(f)

        # ---------- 工具函数（与原脚本一致） ----------
        def get_weight(graph: Dict[int, List[Tuple[int, float]]], a: int, b: int):
            for neighbor, weight in graph.get(a, []):
                if neighbor == b:
                    return weight
            return None

        def update_edge_min(g: Dict[int, Dict[int, float]], u: int, v: int, w: float):
            g.setdefault(u, {})
            g.setdefault(v, {})
            if v in g[u]:
                g[u][v] = min(g[u][v], w)
                g[v][u] = g[u][v]    # 保持对称
            else:
                g[u][v] = w
                g[v][u] = w

        def process_project(
            json_data: Dict[str, Any],
            project_name: str,
            tables_dir: str,
            num: int
        ):
            """
            等价于你原来的 process_project（移除硬编码路径），保持行为不变。
            返回：table_name_to_index, table_names, graph, ground_truth_idx,
                 query_related_idx, project_join_json, join_path, query, purple_name
            """
            projects = json_data.get(project_name)
            if not projects or not isinstance(projects, list):
                raise ValueError(f"未找到项目 {project_name} 或数据格式错误")

            project = projects[num]
            table_relevance = project.get("table_relevance", {})
            ground_truth = project.get("ground_truth", [])
            join_path = []
            ground_truth_edges = project.get("edge_relation", {})
            query_related_table = project.get("query_related_table", [])
            purple_name = project.get("final_table")
            query = project.get("query", "")
            table_names = list(table_relevance.keys())
            print(f"提取的表名: {table_names}")

            # 每项目 JoinClassify 输出
            os.makedirs(project_json_root, exist_ok=True)
            project_json = os.path.join(project_json_root, f"{project_name}.json")

            # 建索引 & 组装 join_path（仅用于记录）
            table_name_to_index = {name: idx for idx, name in enumerate(table_names)}
            for t1, t2, _, _ in ground_truth_edges:
                if t1 in table_name_to_index and t2 in table_name_to_index:
                    join_path.append(f"{table_name_to_index[t1]}-{table_name_to_index[t2]}")

            # 若无 join 预测文件，调用 JoinClassify 生成
            if not os.path.isfile(project_json):
                JoinClassify(mode="predict", tables=tables_dir, output=project_json,
                             model_dir=model_dir, top_k=6)

            with open(project_json, "r", encoding="utf-8") as f:
                join_specs = json.load(f)
            if isinstance(join_specs, dict):
                join_specs = [join_specs]

            # 构图：边权 = -log(prob) 的最小值
            temp_graph: Dict[int, Dict[int, float]] = {}
            for spec in join_specs:
                t1, t2 = spec.get("table1"), spec.get("table2")
                prob = spec.get("prob", 0)
                if not isinstance(prob, (int, float)) or prob <= 0:
                    continue
                if t1 not in table_name_to_index or t2 not in table_name_to_index:
                    continue
                n1, n2 = table_name_to_index[t1], table_name_to_index[t2]
                weight = -math.log(prob)
                update_edge_min(temp_graph, n1, n2, weight)

            graph = {u: list(v.items()) for u, v in temp_graph.items()}

            ground_truth_idx = [table_name_to_index[n] for n in ground_truth if n in table_name_to_index]
            query_related_idx = [table_name_to_index[n] for n in query_related_table if n in table_name_to_index]

            return (table_name_to_index, table_names, graph, ground_truth_idx,
                    query_related_idx, project_json, join_path, query, purple_name)

        # ---------- 主流程（完整保留两段大模型检测） ----------
        for project_name in project_list3:
            output_judge_now = os.path.join(llm_judge_root, project_name)
            os.makedirs(output_judge_now, exist_ok=True)

            tables_dir = os.path.join(csv_root_dir, project_name)                # 表数据目录
            output_dir = os.path.join(outputset_root_dir, top_k, project_name)   # 结果目录
            os.makedirs(output_dir, exist_ok=True)

            samples = json_data.get(project_name, [])
            nums = len(samples)
            if nums == 0:
                print(f"[WARN] {project_name} 无样本，跳过")
                continue

            acc_avg = 0.0
            for num in range(nums):
                (table_index, table_name, graph, ground_truth, query_related,
                 json_path, join_path, query, purple_table) = process_project(
                    json_data, project_name, tables_dir, num
                )

                # --------- 大模型检测相关准备 ---------
                Edgelegal = True           # 与原脚本一致的初始值
                count = 0
                node_related = "\n".join([f"{table_name[i]}" for i in query_related])

                # 初始最短斯坦纳树
                gt = steiner_tree_2approx(graph, query_related)

                # ============ 第一段大模型检测（边级别，逐边判断是否可连）============
                while not Edgelegal:
                    usable = True
                    while not usable:
                        gt = steiner_tree_2approx(graph, query_related)
                        edges = gt.get("edges")
                        nodes = gt.get("nodes")
                        usable = True
                        for edge in edges:
                            a, b = edge.get("id1"), edge.get("id2")
                            if a > b:
                                a, b = b, a
                            json_dir = os.path.join(output_judge_now, f"{table_name[a]}-{table_name[b]}.json")
                            if not os.path.isfile(json_dir):
                                # 逐边调用大模型判定是否 JOIN
                                LLM_Judge(json_path, tables_dir, table_name[a], table_name[b],
                                          purple_table, query, json_dir)
                            with open(json_dir, "r", encoding="utf-8") as file:
                                now_data = json.load(file)
                            if isinstance(now_data, list):
                                join_type = now_data[0].get("is_join")
                            else:
                                join_type = now_data.get("is_join")
                            if join_type == "False":
                                # 不可连：从图中移除该边
                                graph[a] = [(n, w) for n, w in graph[a] if n != b]
                                graph[b] = [(n, w) for n, w in graph[b] if n != a]
                                usable = False
                                break
                            else:
                                # 可信 JOIN：将该边权置为 0 优先选用
                                graph[a] = [(n, 0 if n == b else w) for n, w in graph[a]]
                                graph[b] = [(n, 0 if n == a else w) for n, w in graph[b]]

                    # 通过第一段检查后，再次求解
                    gt = steiner_tree_2approx(graph, query_related)

                    # ============ 第二段大模型检测（整体边集评审，可能排除某些边）============
                    edges = gt.get("edges")
                    edges_set = []
                    node_set = []
                    for edge in edges:
                        a, b = edge.get("id1"), edge.get("id2")
                        if a > b:
                            continue
                        t1, t2 = table_name[a], table_name[b]
                        if t1 not in node_set:
                            node_set.append(t1)
                        if t2 not in node_set:
                            node_set.append(t2)
                        json_dir = os.path.join(output_judge_now, f"{t1}-{t2}.json")
                        with open(json_dir, "r", encoding="utf-8") as file:
                            now_data = json.load(file)
                        if isinstance(now_data, list):
                            reason = now_data[0].get("other_notes")
                        else:
                            reason = now_data.get("other_notes")
                        edges_set.append(f"{t1}↔{t2}:{reason}")
                    node_text = ",".join([f"{node}" for node in node_set])
                    print(node_text)

                    output_judge_now_query = os.path.join(output_judge_now, f"{num}")
                    os.makedirs(output_judge_now_query, exist_ok=True)
                    json_dir = os.path.join(output_judge_now_query, f"{top_k}{project_name}-{num}-{count}.json")

                    # 调用整体评审的大模型
                    Edges_Judge(tables_dir, edges_set, purple_table, query, node_text, node_related, json_dir)

                    with open(json_dir, "r", encoding="utf-8") as f:
                        data = json.load(f)
                    excluded_edges = data.get("excluded_edges", [])
                    if not excluded_edges:
                        Edgelegal = True
                        break

                    # 对被排除的边，从图中删除
                    for edge in excluded_edges:
                        t1, t2 = edge.split("↔")
                        a, b = table_index[t1], table_index[t2]
                        if get_weight(graph, a, b) is None:
                            Edgelegal = True
                            break
                        graph[a] = [(n, w) for n, w in graph[a] if n != b]
                        graph[b] = [(n, w) for n, w in graph[b] if n != a]
                    count += 1

                # ---------（之后与原脚本一致）导出组合 JSON、评估并打印 ----------
                gt = steiner_tree_2approx(graph, query_related)

                idx = 1
                output_path = os.path.join(output_dir, f"{purple_table}__combination_{idx}.json")
                while os.path.isfile(output_path):
                    idx += 1
                    output_path = os.path.join(output_dir, f"{purple_table}__combination_{idx}.json")

                output_data = {
                    "table_name": table_name,
                    "gt": gt,
                    "join_path": join_path,
                    "ground_truth": ground_truth,
                    "query_related": query_related
                }
                with open(output_path, "w", encoding="utf-8") as f:
                    json.dump(output_data, f, ensure_ascii=False, indent=4)

                result_text = check_edges_from_file(output_path)
                acc_avg += float(result_text["accuracy"])
                print(result_text["accuracy"])

            acc_avg = acc_avg / max(1, nums)
            print(f"All query processed,the avg_acc is {acc_avg}.")

    finally:
        # 恢复 stdout/stderr，关闭日志
        sys.stdout = stdout_orig
        sys.stderr = stderr_orig
        try:
            log.close()
        except Exception:
            pass