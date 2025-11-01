import json
import itertools
import networkx as nx
from pathlib import Path
import argparse
import sys

def load_join_graph(purple_table, output_dir):
    """加载连接图"""
    graph_path = output_dir / purple_table / "join_graph.json"
    
    if not graph_path.exists():
        print(f"找不到连接图文件: {graph_path}")
        return None
    
    with open(graph_path, 'r') as f:
        return json.load(f)

def build_graph_from_json(join_graph):
    """从JSON构建NetworkX图"""
    G = nx.Graph()
    
    # 添加节点
    for node in join_graph["nodes"]:
        G.add_node(node["id"], table=node["table"], alias=node.get("alias", ""))
    
    # 添加边
    for edge in join_graph["edges"]:
        G.add_edge(edge["source"], edge["target"], 
                  condition=edge["condition"], 
                  type=edge.get("type", "INNER JOIN"))
    
    return G

def is_complete_graph(subgraph, full_graph):
    """
    验证子图是否等于完整图：
    1. 节点数量相同
    2. 边数量相同
    3. 所有边都相同
    """
    # 检查节点数量
    if set(subgraph.nodes()) != set(full_graph.nodes()):
        return False
    
    # 检查边数量
    if set(subgraph.edges()) != set(full_graph.edges()):
        return False
    
    return True

def find_connected_subgraphs(G):
    """查找所有包含完整连接路径的连通子图组合（排除单表）"""
    # 获取所有连通分量
    connected_components = list(nx.connected_components(G))
    
    # 存储有效连通子图
    connected_subgraphs = []
    
    # 遍历每个连通分量
    for comp in connected_components:
        comp_nodes = list(comp)
        n = len(comp_nodes)
        
        # 如果连通分量只有1个节点，跳过（排除单表）
        if n < 2:
            continue
            
        # 获取该连通分量中的所有节点组合（至少2个节点）
        all_combinations = []
        for r in range(2, n + 1):  # 从2到n
            all_combinations.extend(itertools.combinations(comp_nodes, r))
        
        
        for combo in all_combinations:
            # 找到包含这些节点的最小连通子图
            min_subgraph = find_minimal_connected_subgraph(G, combo)
            
            # 验证最小连通子图是否等于原始图G
            if min_subgraph is None or not is_complete_graph(min_subgraph, G):
                continue
                
           # 获取节点详细信息 - 只包含原始候选节点（combo）
            nodes_info = []
            for node_id in combo:  # 只遍历原始候选节点
                node_data = G.nodes[node_id]
                nodes_info.append({
                    "id": node_id,
                    "table": node_data["table"],
                    "alias": node_data["alias"]
                })

            # 获取边详细信息 - 包含所有必要的边
            edges_info = []
            for edge in min_subgraph.edges(data=True):
                edges_info.append({
                    "source": edge[0],
                    "target": edge[1],
                    "condition": edge[2]["condition"],
                    "type": edge[2]["type"]
                })

            connected_subgraphs.append({
                "nodes": nodes_info,
                "edges": edges_info
            })
            
    return connected_subgraphs

def find_minimal_connected_subgraph(G, node_set):
    """找到包含指定节点集的最小连通子图"""
    # 创建包含这些节点的子图
    subgraph = G.subgraph(node_set)
    
    # 如果已经是连通的，直接返回
    if nx.is_connected(subgraph):
        return subgraph
    
    # 找到连接这些节点的最小生成树
    try:
        # 使用Steiner树近似算法找到最小连通子图
        steiner_tree = nx.algorithms.approximation.steinertree.steiner_tree(G, node_set)
        return steiner_tree
    except:
        # 如果Steiner树算法失败，尝试其他方法
        # 添加必要的中间节点使子图连通
        full_nodes = set(node_set)
        for i in range(len(node_set)):
            for j in range(i + 1, len(node_set)):
                try:
                    path = nx.shortest_path(G, source=node_set[i], target=node_set[j])
                    full_nodes.update(path)
                except nx.NetworkXNoPath:
                    continue
        
        full_subgraph = G.subgraph(full_nodes)
        if nx.is_connected(full_subgraph):
            return full_subgraph
    
    return None


def save_subgraphs(subgraphs, purple_table, output_dir):
    """保存子图组合"""
    output_path = output_dir / purple_table / "subgraph_combinations.json"
    
    with open(output_path, 'w') as f:
        json.dump(subgraphs, f, indent=2)
    
    print(f"已保存 {len(subgraphs)} 个连通子图组合到: {output_path}")

def process_join_graphs(output_dir):
    """处理所有连接图"""
    try:
        # 获取所有有连接图的紫色表
        purple_tables = []
        for dir_path in output_dir.iterdir():
            if dir_path.is_dir() and (dir_path / "join_graph.json").exists():
                purple_tables.append(dir_path.name)
        
        if not purple_tables:
            print(f"在 {output_dir} 中没有找到连接图文件")
            return
        
        print(f"找到有连接图的紫色表: {', '.join(purple_tables)}")
        
        # 处理每个紫色表
        for purple_table in purple_tables:
            print(f"\n处理紫色表: {purple_table}")
            
            # 加载连接图
            join_graph = load_join_graph(purple_table, output_dir)
            if not join_graph:
                continue
            
            # 构建NetworkX图
            G = build_graph_from_json(join_graph)
            
            # 查找所有连通子图组合
            subgraphs = find_connected_subgraphs(G)

            # 保存结果
            if len(subgraphs) == 0:
                print(f"未找到有效的连通子图组合 for {purple_table}, 请手动检查该紫色表的join_graph.json")
            else:
                save_subgraphs(subgraphs, purple_table, output_dir)
        return True

    except Exception as e:
        print(f"处理失败: {e}", file=sys.stderr)
        return False


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="DBT查询选择器")
    parser.add_argument("project", help="项目名称")
    parser.add_argument("--verbose", "-v", action="store_true", help="详细模式")
    
    args = parser.parse_args()

    if args.verbose:
        print(f"开始处理项目: {args.project}")
    
    output_dir = Path(f"/home/wangyuhui/dbt_dataset_test/{args.project}")
    success = process_join_graphs(output_dir)
    
    if args.verbose:
        print(f"处理 {'成功' if success else '失败'}")
    
    sys.exit(0 if success else 1)

# python /home/wangyuhui/Auto-ETL/wyh-dataset/DBT_fivetran/query_select.py hubspot -v