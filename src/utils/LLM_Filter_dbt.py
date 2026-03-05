import json
import asyncio
import openai
import re
import os
import sys
from datetime import datetime
from colorama import init, Fore, Style

init(autoreset=True)

TOTAL_PROMPT_TOKENS = 0
TOTAL_COMPLETION_TOKENS = 0


def make_serializable(obj, max_depth=10, current_depth=0):
    if current_depth > max_depth:
        return f"<max_depth_reached: {type(obj).__name__}>"
    if obj is None or isinstance(obj, (bool, int, float, str)):
        return obj
    if isinstance(obj, dict):
        return {str(k): make_serializable(v, max_depth, current_depth + 1)
                for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [make_serializable(item, max_depth, current_depth + 1)
                for item in obj]
    try:
        return str(obj)
    except Exception:
        return f"<non-serializable: {type(obj).__name__}>"


def load_column_info_from_joinpath(joinpath_file: str) -> dict:
    with open(joinpath_file, 'r', encoding='utf-8') as f:
        jp = json.load(f)

    table_columns = {}
    for template_name, paths in jp.items():
        for path in paths:
            dataset = path['dataset']
            for node_id, node_info in path['node_mapping'].items():
                subtable = node_info['subtable_file']
                full_name = f'{dataset}_{subtable}'
                cols = node_info.get('columns', [])
                if full_name not in table_columns or len(cols) > len(table_columns[full_name]):
                    table_columns[full_name] = cols
    return table_columns


class LLMTableFilterDBT216:
    SYSTEM_MESSAGE = (
        "You are a data lake schema expert specializing in enterprise-grade data warehouses "
        "constructed with dbt (data build tool). Your role is to identify, from a pool of "
        "candidate tables retrieved by a deep learning model, the minimal and semantically "
        "correct set of tables required to answer a given natural language query. "
        "You reason at the column level: a table is relevant if and only if its columns "
        "directly supply information requested by the query."
    )

    def __init__(self, api_key, model="gpt-4o", base_url=None):
        self.client = openai.AsyncOpenAI(
            api_key=api_key,
            base_url=base_url
        )
        self.model = model

    def _build_prompt(self, query: str, candidates: list, column_info: dict) -> str:
        tables_section = ""
        for i, (table_name, score) in enumerate(candidates, 1):
            cols = column_info.get(table_name, [])
            if isinstance(cols, dict):
                cols = list(cols.keys())
            cols_str = ", ".join(cols[:40]) if cols else "(no columns available)"
            tables_section += f"{i}. `{table_name}`\n   Columns: {cols_str}"
            if len(cols) > 40:
                tables_section += f"... (+{len(cols)-40} more)"
            tables_section += "\n\n"

        prompt = f"""# QUERY
{query}

# CANDIDATE TABLES WITH COLUMNS
{tables_section}
# OPERATION 1 — Query Understanding
Decompose the query above into a list of atomic semantic components. Each component should
represent one self-contained information need: a key entity, a measurable attribute or metric,
a filter or scoping condition, or a dimensional attribute (e.g., entity name or status).
Also identify the data source referenced by the query, if any.

# OPERATION 2 — Schema Reasoning and Verification
Using the semantic components identified above, evaluate each candidate table. For each table,
determine whether it contains at least one column that directly satisfies a semantic component
of the query. Apply the following principles:

  Principle 1 — Column-Level Matching
    A table is relevant only if its columns directly supply data for at least one semantic
    component. Do not retain a table based on its name alone.

  Principle 2 — Data Source Scoping
    If the query explicitly references a single data source (e.g., "in Google Ads"), discard
    tables from unrelated sources. If no source is specified, evaluate all candidates on
    column relevance alone.

  Principle 3 — Representational Layer Deduplication
    In dbt projects, the same entity may appear at multiple layers (raw source, staging stg_,
    intermediate int_, mart/report models). When semantically equivalent tables exist at
    different layers, prefer the most granular, un-aggregated version. Retain a staging table
    only when no corresponding un-prefixed source table is present in the candidate pool.

  Principle 4 — Granularity Preservation
    When candidate tables cover the same subject at multiple granularities (e.g., hourly vs.
    daily, ad-level vs. campaign-level), retain all granularities consistent with the query's
    requirements. Prune a granularity only when the query explicitly restricts the required
    level of detail.

  Principle 5 — Selective Dimension Table Inclusion
    Retain a dimension or history table (e.g., campaign_history) only when the query
    explicitly requests a descriptive attribute of that entity (e.g., its name or status).
    Do not include dimension tables solely for potential join utility.

# OUTPUT FORMAT
Return valid JSON only (no markdown):
{{
  "data_source_mentioned": "<source name, or null>",
  "semantic_components": ["component 1", "component 2", ...],
  "table_reasoning": {{
    "<table_name>": {{
      "matched_components": ["component i"],
      "matched_columns":    ["col_a"],
      "verdict":            "keep",
      "reason":             "<one-sentence justification>"
    }}
  }},
  "selected_tables": ["table1", "table2", ...]
}}"""
        return prompt

    async def filter_candidates(self, query: str, candidates: list,
                                column_info: dict) -> dict:
        prompt = self._build_prompt(query, candidates, column_info)

        llm_interaction = {
            "pass1_prompt": prompt,
            "pass1_response": None,
            "pass2_prompt": None,
            "pass2_response": None,
            "parsed_result": None,
            "error": None,
            "usage": None
        }

        try:
            resp = None
            for attempt in range(3):
                try:
                    resp = await self.client.chat.completions.create(
                        model=self.model,
                        messages=[
                            {"role": "system", "content": self.SYSTEM_MESSAGE},
                            {"role": "user", "content": prompt}
                        ],
                        temperature=0.1,
                        max_tokens=4000,
                        timeout=120
                    )
                    break
                except Exception as retry_err:
                    print(f"{Fore.YELLOW}  Attempt {attempt+1} failed: {retry_err}{Style.RESET_ALL}")
                    if attempt < 2:
                        await asyncio.sleep(5 * (attempt + 1))
            if resp is None:
                raise Exception("LLM failed after 3 retries")

            content = resp.choices[0].message.content.strip()
            llm_interaction["pass1_response"] = content

            global TOTAL_PROMPT_TOKENS, TOTAL_COMPLETION_TOKENS
            if resp.usage:
                TOTAL_PROMPT_TOKENS += resp.usage.prompt_tokens
                TOTAL_COMPLETION_TOKENS += resp.usage.completion_tokens
                llm_interaction["usage"] = {
                    "prompt_tokens": TOTAL_PROMPT_TOKENS,
                    "completion_tokens": TOTAL_COMPLETION_TOKENS,
                }

            content_clean = re.sub(r'```json\s*', '', content)
            content_clean = re.sub(r'```\s*', '', content_clean)
            json_match = re.search(r'\{.*\}', content_clean, re.DOTALL)
            result = json.loads(json_match.group(0) if json_match else content_clean)
            llm_interaction["parsed_result"] = result

            selected_tables = result.get('selected_tables', result.get('remaining_tables', []))
            if len(selected_tables) == 0:
                selected_tables = [t for t, s in candidates[:5]]

            return {
                'filtered_tables': selected_tables,
                'pass1_tables': selected_tables,
                'summary': {},
                'llm_interaction': llm_interaction
            }

        except Exception as e:
            llm_interaction["error"] = str(e)
            return {
                'filtered_tables': [t for t, s in candidates[:5]],
                'pass1_tables': [],
                'summary': {},
                'llm_interaction': llm_interaction
            }


    def __init__(self, api_key, model="gpt-4o", base_url=None):
        self.client = openai.AsyncOpenAI(
            api_key=api_key,
            base_url=base_url
        )
        self.model = model


def compute_metrics(gt_set: set, pred_set: set) -> dict:
    tp = len(gt_set & pred_set)
    fp = len(pred_set - gt_set)
    fn = len(gt_set - pred_set)
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
    return {
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'tp': tp,
        'fp': fp,
        'fn': fn
    }


async def main():

    # API配置
    API_KEY = ""
    API_BASE = ""
    MODEL = ""

    filter_obj = LLMTableFilterDBT216(
        api_key=API_KEY,
        model=MODEL,
        base_url=API_BASE
    )

    input_file = ''
    old_results_file = ''
    output_dir = f''
    os.makedirs(output_dir, exist_ok=True)

    TOP_K = int(sys.argv[2]) if len(sys.argv) > 2 else 30

    num_samples = int(sys.argv[1]) if len(sys.argv) > 1 else None

    with open(input_file, 'r', encoding='utf-8') as f:
        dl_data = json.load(f)

    raw_data = []
    for r in dl_data['results']:
        top_k = 30
        raw_data.append({
            'sample_id': r['sample_id'],
            'query': r['query'],
            'sql': r.get('sql', ''),
            'gt': r['gt'],
            'candidates': r['candidates'],
            'category': r.get('category', ''),
            'top_k': top_k,
        })

    annotation_file = ''
    with open(annotation_file, 'r', encoding='utf-8') as f:
        annotation_data = json.load(f)
    column_info = {}
    for cat, queries in annotation_data.items():
        for q in queries:
            for table_name, cols in q.get('column_relevance', {}).items():
                if table_name not in column_info:
                    column_info[table_name] = list(cols.keys()) if isinstance(cols, dict) else cols
    old_metrics_map = {}
    old_selected_map = {}
    if os.path.exists(old_results_file):
        with open(old_results_file, 'r', encoding='utf-8') as f:
            old_data = json.load(f)
        for r in old_data['detailed_results']:
            old_metrics_map[r['sample_id']] = r['metrics']
            old_selected_map[r['sample_id']] = r['selected_tables']
            f"R={old_data['metrics']['recall']:.4f} F1={old_data['metrics']['f1']:.4f}{Style.RESET_ALL}"

    if num_samples and num_samples < len(raw_data):
        sample_indices = list(range(num_samples))
        sample_data = raw_data[:num_samples]
    else:
        sample_data = raw_data
        sample_indices = list(range(len(raw_data)))

    semaphore = asyncio.Semaphore(20)
    progress = {'done': 0, 'total': len(sample_data)}

    async def process_one(idx, item):
        async with semaphore:
            sid = item['sample_id']
            candidates = item['candidates'][:item.get('top_k', TOP_K)]

            filter_result = await filter_obj.filter_candidates(
                item['query'], candidates, column_info
            )

            selected_tables = filter_result['filtered_tables']
            gt_set = set(item['gt'])
            pred_set = set(selected_tables)
            topk_set = set(t for t, s in candidates)
            topk_metrics = compute_metrics(gt_set, topk_set)
            new_metrics = compute_metrics(gt_set, pred_set)
            old_m = old_metrics_map.get(sid, {})
            old_sel = old_selected_map.get(sid, [])
            old_f1 = old_m.get('f1', 0)
            new_f1 = new_metrics['f1']
            delta = new_f1 - old_f1

            progress['done'] += 1
            print(f"[{progress['done']:3d}/{progress['total']}] ID={sid} "
                  f"P={new_metrics['precision']:.2f} R={new_metrics['recall']:.2f} F1={new_f1:.2f} "
                  f"selected={len(selected_tables)} | {item['query'][:60]}...", flush=True)

            if new_metrics['fn'] > 0:
                print(f"  Missing: {gt_set - pred_set}", flush=True)

            return {
                'sample_id': sid,
                'query': item['query'],
                'sql': item.get('sql', ''),
                'ground_truth': item['gt'],
                'candidates_top30': [[t, s] for t, s in candidates],
                'old_selected': old_sel,
                'old_metrics': old_m,
                'pass1_tables': filter_result.get('pass1_tables', []),
                'new_selected': selected_tables,
                'new_summary': filter_result['summary'],
                'new_llm_interaction': filter_result['llm_interaction'],
                'topk_metrics': topk_metrics,
                'new_metrics': new_metrics,
                'f1_delta': delta,
            }

    tasks = [process_one(idx, item) for idx, item in enumerate(sample_data, 1)]
    results = await asyncio.gather(*tasks)

    def _mean(lst): return sum(lst) / len(lst) if lst else 0
    avg_topk_p = _mean([r['topk_metrics']['precision'] for r in results])
    avg_topk_r = _mean([r['topk_metrics']['recall'] for r in results])
    avg_topk_f1 = _mean([r['topk_metrics']['f1'] for r in results])

    avg_old_p = _mean([r['old_metrics'].get('precision', 0) for r in results])
    avg_old_r = _mean([r['old_metrics'].get('recall', 0) for r in results])
    avg_old_f1 = _mean([r['old_metrics'].get('f1', 0) for r in results])

    avg_new_p = _mean([r['new_metrics']['precision'] for r in results])
    avg_new_r = _mean([r['new_metrics']['recall'] for r in results])
    avg_new_f1 = _mean([r['new_metrics']['f1'] for r in results])
    avg_num_selected = _mean([len(r['new_selected']) for r in results])

    improved = sum(1 for r in results if r['f1_delta'] > 0.001)
    degraded = sum(1 for r in results if r['f1_delta'] < -0.001)
    unchanged = len(results) - improved - degraded

    output_file = os.path.join(
        output_dir,
        f'dbt_216_llm_filter_top{TOP_K}_{len(results)}samples.json'
    )
    cost_input = TOTAL_PROMPT_TOKENS / 1_000_000 * 2.5
    cost_output = TOTAL_COMPLETION_TOKENS / 1_000_000 * 10
    output_data = {
        'metadata': {
            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'model': MODEL,
            'source': input_file,
            'top_k': TOP_K,
            'total_samples': len(results),
            'sample_indices': sample_indices,
            'description': '',
            'token_usage': {
                'prompt_tokens': TOTAL_PROMPT_TOKENS,
                'completion_tokens': TOTAL_COMPLETION_TOKENS,
                'total_tokens': TOTAL_PROMPT_TOKENS + TOTAL_COMPLETION_TOKENS,
                'estimated_cost_usd': cost_input + cost_output
            }
        },
        'comparison': {
            'topk_baseline': {
                'precision': float(avg_topk_p),
                'recall': float(avg_topk_r),
                'f1': float(avg_topk_f1)
            },
            'old_llm_filter': {
                'precision': float(avg_old_p),
                'recall': float(avg_old_r),
                'f1': float(avg_old_f1)
            },
            'new_llm_filter': {
                'precision': float(avg_new_p),
                'recall': float(avg_new_r),
                'f1': float(avg_new_f1),
                'avg_num_selected': float(avg_num_selected)
            },
            'improvement': {
                'f1_delta': float(avg_new_f1 - avg_old_f1),
                'precision_delta': float(avg_new_p - avg_old_p),
                'recall_delta': float(avg_new_r - avg_old_r),
                'improved_count': improved,
                'degraded_count': degraded,
                'unchanged_count': unchanged
            }
        },
        'detailed_results': make_serializable(results)
    }

    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(output_data, f, ensure_ascii=False, indent=2)

if __name__ == "__main__":
    asyncio.run(main())
