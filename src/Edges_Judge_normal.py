import json
import os
import pandas as pd
from collections import defaultdict
from openai import OpenAI
import tiktoken
import requests
import time
client = OpenAI(
        api_key="",  # 替换为你的实际 API 密钥
        base_url="https://aihubmix.com/v1"  # 自定义 API 端点
    )
sample_number = 20
output_dir ="/data/new_dbt_dataset/111test/output/test.json"
def count_tokens(text, model="gpt-4o"):
    # 不同模型可能使用不同的编码器
    encoding = tiktoken.encoding_for_model(model)
    tokens = encoding.encode(text)
    return len(tokens)
def analyze_table_joins(csv_dir: str, tables: list, purple_table: str, query: str, nodetxt:str,node_related:str, temperature: float = 0.1):
    """
    一次性分析多表连接是否符合 query 语义，输出不合理的边
    """

    all_tables = sorted({t for t in nodetxt.split(",")})
    print(f"识别到的所有表: {all_tables}")

    # === 2. 读取每张表的列名 ===
    table_columns = {}
    for t in all_tables:
        csv_path = os.path.join(csv_dir, f"{t}.csv")
        if not os.path.exists(csv_path):
            table_columns[t] = ["⚠️ 未找到该表文件"]
            continue
        try:
            df = pd.read_csv(csv_path, nrows=1)  # 只读取首行
            table_columns[t] = list(df.columns)
        except Exception as e:
            table_columns[t] = [f"⚠️ 读取失败: {e}"]

    # === 3. 构造 prompt 输入 ===
    table_schema_text = "\n".join([
        f"### Table: {t}\nColumns: {', '.join(cols)}"
        for t, cols in table_columns.items()
    ])

    edges_text = "\n".join(f"- {e}" for e in tables)

    prompt = f"""
### Query
{query}

### Target Table
{purple_table}

### Table Schemas
{table_schema_text}

### Candidate Join Edges
{edges_text}

### Related Tables
{node_related}
---

### Your Task
You are a **professional data analyst** who specializes in **semantic query interpretation** and **table relationship reasoning**.  
Your objective is to evaluate each candidate join edge (A↔B) based on the **semantic meaning, keywords, and intent expressed in the query**,  
rather than only the technical join logic.

Focus on **how the query’s key terms, structure, and metrics depend on these joins** to form a coherent semantic data path.  
A join may still be **semantically required** even if it does not directly contain performance metrics —  
for example, if it provides **organizational hierarchy, campaign linkage, time filtering, or status context**.

> ✅ Keep joins that provide contextual, hierarchical, or temporal meaning.  
> ⚠️ Only exclude joins that are **clearly irrelevant** to both the query’s intent and its keywords.

---

### Core Evaluation Principles

1. **Keyword-Centric Semantic Relevance**  
   - Identify the **keywords and key phrases** in the query (e.g., "daily", "total", "revenue", "status", "date range", "organization", "campaign").  
   - Keep joins that directly or indirectly support these keywords.  
   - Exclude only joins that have **no connection** to any keyword or required concept in the query.

2. **Query-Intent Alignment**  
   - Understand what the query truly asks for: which **metrics**, **filters**, and **dimensions** are required.  
   - Keep joins that contribute directly or indirectly to fulfilling these needs.  
   - Exclude only joins that are completely unrelated to the query’s logical or semantic structure.

3. **Metric and Aggregation Awareness**  
   - If the query asks for **totals, sums, counts, downloads, or performance metrics**,  
     joins involving tables with **report, history, or log data** are **necessary** to compute and contextualize these metrics.  
   - These joins are essential for ensuring time-aligned, status-aware aggregations.  
   - **Do not exclude** such joins even if they seem redundant — they are part of the semantic foundation.

4. **Performance Context Dependency**  
   - Even if a table does not directly contain performance metrics, it may still be **semantically required**  
     if it provides the **organizational, campaign, or temporal context** necessary to interpret or aggregate those metrics correctly.  
   - For example, `organization`, `campaign_history`, and other hierarchy/context tables are essential  
     to connect ad performance data to their proper entities.  
   - Do **not exclude** contextual joins just because they lack numeric metrics.

5. **Historical / Temporal Context Relevance**  
   - If the query includes **time-related or status-related keywords** (e.g., “daily”, “date range”, “trend”, “status”, “period”),  
     then tables whose names or columns include **“history”, “log”, “record”, “start_time”, “end_time”, or “status”**  
     are **semantically necessary** and must not be excluded.  
   - These tables provide the **temporal and status context** that enable correct filtering and aggregation over time.  
   - Excluding them would break the query’s ability to report or filter by date and campaign status.

6. **Hierarchical Structure Rule**  
   - When the query includes hierarchical breakdowns (e.g., by organization → campaign → ad group → search term),  
     all joins that connect these hierarchical levels are **semantically essential**.  
   - Removing such joins (e.g., `organization↔campaign_history`, `campaign_history↔campaign_report`)  
     would break the semantic hierarchy, making it impossible to aggregate or filter correctly.  
   - Do **not exclude** any join that connects higher-level entities (organization, campaign)  
     with their corresponding history or report tables.

7. **Personal / Contextual Information Relevance**  
   - If the query refers to **users, individuals, accounts, or organizations**,  
     joins involving such contextual tables must **not be excluded**,  
     since they define key grouping or filtering relationships.

8. **Result Integrity and Semantic Cohesion**  
   - Keep joins that maintain **semantic completeness** and **temporal consistency**.  
   - Exclude only joins that clearly add **irrelevant data**, **semantic noise**, or **contradictory relationships**.

9. **Caution Rule**  
   - When uncertain, **keep the join** rather than exclude it.  
   - Exclusion should only happen when a join is **unambiguously irrelevant**  
     to both the query’s meaning and its required hierarchical or temporal context.

---

### Exclusion Rules (apply only when clearly justified)

Exclude a join edge **only if**:
- It connects entities that have **no logical, temporal, or semantic relevance** to the query;  
- It introduces data that does **not support** any keyword, metric, or dimension in the query;  
- It forms a **meaningless or conflicting relationship** that disrupts semantic alignment.

Do **not exclude** a join edge if:
- It supports any **keyword, metric, or concept** in the query;  
- It provides **hierarchical, historical, or contextual links** necessary for aggregation or filtering;  
- It connects **organization**, **campaign**, **ad group**, or **search term** layers;  
- It enables **date range filtering**, **status alignment**, or **entity grouping**;  
- It may seem redundant, but still adds **semantic or temporal value** for accurate reporting.

---

### Output Format
Return the result strictly in **JSON format**, exactly as shown below:
{{
  "excluded_edges": ["users↔orders", "sessions↔products"],
  "other_notes": "Briefly explain exclusion reasons (e.g., which are irrelevant, which add noise or cause semantic conflicts)."
}}
"""



    # === 4. 保存 prompt 到 prompt.json 方便调试 ===
    # prompt_path = os.path.join(csv_dir, f"nodetxt.json")
    # with open(prompt_path, "w", encoding="utf-8") as f:
    #     f.write(json.dumps({"prompt": prompt}, ensure_ascii=False, indent=2).replace("\\n", "\n"))
    # print(f"✅ 已将 prompt 写入: {prompt_path}")

    # === 5. 调用大模型（一次性处理所有边）===
    t0 = time.perf_counter()
    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {"role": "system", "content": "You are a database expert who evaluates the validity of multi-table joins."},
            {"role": "user", "content": prompt},
        ],
        response_format={"type": "json_object"},
        temperature=temperature,
        timeout=60
    )
    t1 = time.perf_counter()
    latency_ms = (t1 - t0) * 1000.0
    # === 6. 解析模型结果 ===
    result_text = response.choices[0].message.content
    results = json.loads(result_text)
    total_prompt_tokens = response.usage.prompt_tokens
    total_completion_tokens = response.usage.completion_tokens
    total_tokens = response.usage.total_tokens
    print(f"LLM 端到端耗时: {latency_ms:.2f} ms")
    print("累计输入 Token 数:", total_prompt_tokens)
    print("累计输出 Token 数:", total_completion_tokens)
    print("累计总 Token 数:", total_tokens)
    return results

def Edges_Judge(csv_path:str,tables: list,purple_table:str, query:str, nodetxt: str,node_related:str, output_dir:str):
    results = analyze_table_joins(csv_path, tables, purple_table, query, nodetxt,node_related, temperature=0.1)
    with open(output_dir, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    print("分析完成，结果已保存到 ",output_dir)

if __name__ == "__main__":
    json_path = "/data/new_dbt_dataset/111test/project_json/amazon_ads.json"
    csv_dir = "/data/new_dbt_dataset/111test/csv_files/amazon_ads"
    results = analyze_table_joins(json_path, csv_dir,"ad_group_history","targeting_keyword_report", temperature=0.1)

    with open(output_dir, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    print("分析完成，结果已保存到 ",output_dir)
