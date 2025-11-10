import json
import os
import pandas as pd
from collections import defaultdict
from openai import OpenAI
import tiktoken
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
def analyze_table_joins(json_path: str, csv_dir: str,table1: str, table2: str,purple_table:str, query:str, temperature: float = 0.1):
    with open(json_path, "r", encoding="utf-8") as f:
        join_specs = json.load(f)

    if isinstance(join_specs, dict):
        join_specs = [join_specs]

    # 按 (table1, table2) 分组合并
    grouped_specs = defaultdict(list)
    for spec in join_specs:
        key = (spec["table1"], spec["table2"])
        grouped_specs[key].append({
            "column1": spec["column1"],
            "column2": spec["column2"],
            "prob": spec.get("prob")
        })

    results = []
    total_prompt_tokens = 0
    total_completion_tokens = 0
    total_tokens = 0
    candidates=grouped_specs[(table1,table2)]
    if not candidates:
        candidates=grouped_specs[(table2,table1)]
        table1,table2=table2,table1
    csv1 = os.path.join(csv_dir, f"{table1}.csv")
    csv2 = os.path.join(csv_dir, f"{table2}.csv")

    if not os.path.exists(csv1) or not os.path.exists(csv2):
        print(f"⚠️ 找不到表格文件: {csv1} 或 {csv2}")

    df1 = pd.read_csv(csv1).head(sample_number)
    df2 = pd.read_csv(csv2).head(sample_number)

    # 构建候选连接列说明
    candidate_text = "\n".join(
        [f"- {c['column1']} ↔ {c['column2']}"
            for c in candidates]
    )
    print(candidate_text)
    # Prompt 设计：传入多个候选

    prompt = f"""
This task aims to analyze and determine the **semantic relationship** between two tables in a **data lake**.  
The primary goal is to identify whether the two tables can be **semantically joined**,  
and—when the join is valid—to describe the **business meaning** of the relationship.

For example:  
If one table represents *employees* and another represents *departments*,  
the system should infer—based on table names, column names, and data semantics—  
that these two tables can be joined through `department_id`.  
The semantic meaning of this join would be:  
> “Join the employees table with the departments table by `department_id`, meaning: retrieve all employees under each department.”

All reasoning and decisions must be based on **observable evidence**  
(such as table names, column semantics, and data patterns),  
not on subjective assumptions.  
The goal is not only to determine whether the two tables can be joined (True/False),  
but also to **explain the business meaning** of the connection when it exists.

---

### **Input**

- Natural language query: {query}  
- Target table name: {purple_table}  

- Table 1: {table1}  
- Table 2: {table2}  

- Candidate join columns (from model or heuristic detection):  
  {candidate_text}  

- First {sample_number} rows of Table 1:  
  {df1.to_csv(index=False)}  

- First {sample_number} rows of Table 2:  
  {df2.to_csv(index=False)}  

---

### **Analysis Steps**

#### **1. Overall Semantic Analysis**
Analyze both tables to determine whether they belong to the same or related **business domain** (e.g., HR, Sales, Finance, Operations),  
and identify possible **abbreviations, aliases, or terminology mappings**.

Focus on the following aspects:  
- Keywords, roots, or abbreviations in table/column names (e.g., “dept” = department, “cust” = customer);  
- Naming patterns of columns (e.g., *_id, *_code, *_key);  
- Data structure and column type characteristics;  
- Whether data value patterns indicate consistency or referencing relationships.

Output whether the two tables belong to the same semantic domain and whether a potential entity or foreign key relationship exists.

---

#### **2. Column-Level Semantic Consistency Evaluation**
For each candidate column pair, determine whether they represent the **same concept or entity**.  
If two columns can be semantically aligned through any of the following, they are considered semantically joinable:

- **Format normalization**: unify case, trim spaces, clean symbols;  
- **Type alignment**: string ↔ int, timestamp ↔ date;  
- **Concatenation or decomposition**: e.g., first_name + last_name → full_name;  
- **Hierarchical mapping**: e.g., city → region → country;  
- **Composite keys**: e.g., (user_id, source_system);  
- **Semantic mapping**: code-to-description lookup;  
- **Temporal or conditional alignment**: join only when records share the same effective period or status;  
- **Contextual semantics**: primary–foreign key relationships or hierarchical dependencies.

---

#### **3. Value Pattern Consistency Constraint**

When candidate columns are numeric, coded, or ID-type fields,  
they must satisfy **value pattern consistency** in addition to semantic similarity:

- The value length distributions of the two columns should be approximately similar (e.g., both 4-digit IDs, or both 10-digit codes);  
- Their numeric ranges or magnitudes should be roughly comparable;  
- If one column contains short IDs (3–4 digits) while the other contains long codes (9–10 digits),  
  they **must be marked unjoinable (false)** even if their names are semantically similar;  
- If the two columns’ value distributions or encoding structures differ completely  
  (e.g., purely numeric vs. alphanumeric with prefixes),  
  they must also be marked as unjoinable.

This constraint ensures that a join is not only semantically valid but also **feasible at the data level**.

---

#### **4. Join Type Determination and Business Semantics**

After confirming that the two tables have semantically and structurally compatible column pairs,  
determine the most appropriate **join type**, and briefly describe the resulting **business meaning**.

| Join Type | Typical Scenario | Example Semantic Description |
|------------|------------------|-------------------------------|
| **INNER JOIN** | Both tables contain matching entities | “Return only records that appear in both the Orders and Customers tables (i.e., valid customer orders).” |
| **LEFT JOIN** | The left table is the main table, and the right adds attributes | “Keep all orders and enrich them with customer information.” |
| **UNION / UNION ALL** | Both tables describe the same entity type but different subsets | “Combine online and offline sales data.” |

---

#### **5. INNER JOIN Strict Constraint**

Only when there exists **at least one pair of columns** between the two tables  
that can be matched by a **strict equality condition (=`=`)**  
and that pair is **semantically equivalent, type-compatible, and value-pattern consistent** (as per Step 3),  
may the relationship be classified as **INNER JOIN**.

If no such equality-based pair exists,  
even if the tables are semantically related,  
they **must not** be classified as INNER JOIN.  
Instead, consider LEFT JOIN, LOOKUP JOIN, or another descriptive relationship.

---

### **Output Format**

The final output must strictly follow this JSON structure:

{{
    "table1": "{table1}",
    "table2": "{table2}",
    "candidates": {{
        "<column1> ↔ <column2>": true/false,
        ...
    }},
    "is_join": "<True/False>",
    "other_notes": "Briefly describe the join type and the business meaning of the resulting relationship."
}}

---

### **Additional Notes**

- All reasoning must rely on **explicit evidence** from table names, column names, and data samples.  
- Do **not** make speculative judgments based on likely or intuitive relationships.  
- If multiple column pairs meet the condition, describe multi-key or primary–foreign key relationships in `other_notes`.  
- If no column pairs qualify, output `"is_join": "False"` and briefly explain the reason.
"""


    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {"role": "system", "content": "You are a database expert who evaluates the validity of multi-table joins.。"},
            {"role": "user", "content": prompt},
        ],
        response_format={"type": "json_object"},
        temperature=temperature,
    )

    result_text = response.choices[0].message.content
    results.append(json.loads(result_text))
    total_prompt_tokens += response.usage.prompt_tokens
    total_completion_tokens += response.usage.completion_tokens
    total_tokens += response.usage.total_tokens
        
    print("累计输入 Token 数:", total_prompt_tokens)
    print("累计输出 Token 数:", total_completion_tokens)
    print("累计总 Token 数:", total_tokens)
    return results

def LLM_Judge(json_path:str,csv_path:str,table1: str, table2: str,purple_table:str, query:str, output_dir:str):
    results = analyze_table_joins(json_path, csv_path, table1, table2, purple_table, query, temperature=0.1)
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
