import json
import os
import pandas as pd
from collections import defaultdict
from openai import OpenAI
import tiktoken
import requests
import time
import io
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

def analyze_table_joins(csv_dir: str, tables: dict, cols: dict,
                        model: str = "gpt-4o", temperature: float = 0.1, timeout: int = 60):
    # ---- normalize tables ----
    if isinstance(tables, dict):
        role2table = {
            "a1": tables.get("a1"),
            "b1": tables.get("b1"),
            "a2": tables.get("a2"),
            "b2": tables.get("b2"),
        }
    else:
        if not isinstance(tables, (list, tuple)) or len(tables) < 4:
            raise ValueError("`tables` must be [a1, b1, a2, b2] or a dict with those keys.")
        role2table = {"a1": tables[0], "b1": tables[1], "a2": tables[2], "b2": tables[3]}

    a1_name = role2table.get("a1") or ""
    b1_name = role2table.get("b1") or ""
    a2_name = role2table.get("a2") or ""
    b2_name = role2table.get("b2") or ""

    # ---- candidates only from cols ----
    def _c(role_key: str, table_name: str):
        cand = cols.get(role_key) if isinstance(cols, dict) else None
        if cand is None and table_name:
            cand = cols.get(table_name) if isinstance(cols, dict) else None
        return [str(c) for c in cand] if isinstance(cand, (list, tuple)) else []

    a1_cols, b1_cols = _c("a1", a1_name), _c("b1", b1_name)
    a2_cols, b2_cols = _c("a2", a2_name), _c("b2", b2_name)

    # ---- first 5 sample rows from CSVs (best-effort) ----
    def _sample_rows(table_name: str, n: int = 5):
        if not table_name:
            return []
        path = os.path.join(csv_dir, f"{table_name}.csv")
        try:
            df = pd.read_csv(path, nrows=n)
            # 将前 n 行转换为 CSV 格式字符串
            output = io.StringIO()
            df.to_csv(output, index=False, header=True)  # 保留列名，不输出索引
            output.seek(0)  # 回到字符串开头
            
            # 返回 CSV 格式的字符串
            return output.getvalue()
        except Exception:
            return []

    a1_samples = _sample_rows(a1_name, 5)
    b1_samples = _sample_rows(b1_name, 5)
    a2_samples = _sample_rows(a2_name, 5)
    b2_samples = _sample_rows(b2_name, 5)

    # ---- prompt ----
    prompt = f"""
You need to judge whether the two table mappings are correct for a comparison task:
- Raw tables: a1, b1 with their candidate columns
- Transformed tables: a2, b2 with their candidate columns
Mapping: (a1 → a2) and (b1 → b2).

Rule (iff): there exists at least one same-entity column pair between a1↔a2 AND at least one same-entity column pair between b1↔b2.
"Same entity" is semantic (e.g., student.id vs grades.student_id). Allow aliases, _id/uuid, PK/FK, synonyms, etc.

# Input
- a1:
  - name: {a1_name}
  - candidate_columns: {a1_cols}
  - first_5_rows: 
  {a1_samples}
- a2:
  - name: {a2_name}
  - candidate_columns: 
  {a2_cols}
  - first_5_rows: 
  {a2_samples}
- b1:
  - name: {b1_name}
  - candidate_columns: {b1_cols}
  - first_5_rows: 
  {b1_samples}
- b2:
  - name: {b2_name}
  - candidate_columns: {b2_cols}
  - first_5_rows: 
  {b2_samples}

# Output (strict JSON; lowercase true/false; no extra text)
{{
  "is_correct": true/false,
  "reason": "<1–3 concise sentences: if true, cite one strongest pair for a1↔a2 and one for b1↔b2; if false, explain what's missing.>"
}}
"""

    # ---- LLM call with latency + token usage logging ----
    t0 = time.perf_counter()
    try:
        resp = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": "You are a data engineer who judges whether two table pairs share at least one semantically identical entity column."},
                {"role": "user", "content": prompt},
            ],
            response_format={"type": "json_object"},
            temperature=temperature,
            timeout=timeout
        )
        t1 = time.perf_counter()
        latency_ms = (t1 - t0) * 1000.0

        # --- token usage (robust access) ---
        prompt_tokens = getattr(getattr(resp, "usage", None), "prompt_tokens", None)
        completion_tokens = getattr(getattr(resp, "usage", None), "completion_tokens", None)
        total_tokens = getattr(getattr(resp, "usage", None), "total_tokens", None)

        print(f"LLM latency: {latency_ms:.2f} ms")
        if prompt_tokens is not None:
            print(f"Prompt tokens: {prompt_tokens}")
            print(f"Completion tokens: {completion_tokens}")
            print(f"Total tokens: {total_tokens}")

        result_text = resp.choices[0].message.content
        return json.loads(result_text)

    except Exception as e:
        # 即使失败也保持可用
        print(f"LLM call failed: {e}")
        return {
            "is_correct": False,
            "reason": "Model inference failed or input was insufficient to confirm same-entity columns for both (a1↔a2) and (b1↔b2)."
        }


def Col_Judge(csv_path: str, tables: dict, cols: dict, output_dir: str):
    results = analyze_table_joins(csv_path, tables, cols)
    with open(output_dir, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    print("分析完成，结果已保存到 ", output_dir)