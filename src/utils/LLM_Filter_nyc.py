import json
import asyncio
import openai
import re
import os
import sys
from datetime import datetime
from colorama import init, Fore, Style
import numpy as np

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
    except:
        return f"<non-serializable: {type(obj).__name__}>"


class LLMTableFilter:

    def __init__(self, api_key, model="gpt-4o", base_url=None):
        self.client = openai.AsyncOpenAI(
            api_key=api_key,
            base_url=base_url
        )
        self.model = model

    async def filter_candidates(self, query: str, candidates: list,
                                 column_info: dict) -> dict:
        tables_section = ""
        for i, (table_name, score) in enumerate(candidates[:30], 1):
            cols = list(column_info.get(table_name, {}).keys())
            cols_str = ", ".join(cols[:30])
            tables_section += f"{i}. `{table_name}` (Score: {score:.6f})\n"
            tables_section += f"   Columns: {cols_str}"
            if len(cols) > 30:
                tables_section += f"... (+{len(cols) - 30} more)"
            tables_section += "\n\n"

        prompt = f"""# QUERY
{query}

# CANDIDATE TABLES
{tables_section}
# OPERATION 1 — Query Understanding
Decompose the query into a structured list of atomic semantic components. Each component
should represent one self-contained information need: a key entity, a measurable attribute,
a filter condition, or a temporal specification.

Pay particular attention to temporal references. If the query mentions a specific year,
fiscal year, or date range, record it explicitly, as it will be used to resolve temporal
ambiguity among candidate tables in the verification step.

# OPERATION 2 — Schema Reasoning and Verification
Using the semantic components identified above, evaluate each candidate table. For each
table, determine whether it contains at least one column that directly satisfies a semantic
component of the query. Apply the following principles:

  Principle 1 — Column-Level Matching
    A table is relevant only if its columns directly supply data for at least one semantic
    component. Do not retain a table based on its name alone.

  Principle 2 — Temporal Scope Consistency
    If a temporal scope was identified, include only tables whose names or schemas are
    consistent with that temporal scope. Exclude tables that clearly correspond to a
    different time period, even if their schema is otherwise similar.

  Principle 3 — Partition-Complete Selection
    In this data lake, large datasets may be stored as horizontally partitioned sub-tables
    sharing a base name with a numeric suffix (e.g., records, records_1, records_2). These
    sub-tables are disjoint partitions, not duplicates. When any sub-table in a series is
    relevant, examine all numbered variants present in the candidate pool independently and
    retain each one that satisfies the column-level relevance criterion.

  Principle 4 — Non-Redundancy
    If two candidate tables are semantically identical in schema and content, retain only
    the one with broader column coverage or the lower partition index. Do not deduplicate
    based on naming similarity alone.

# OUTPUT FORMAT
Return valid JSON (no markdown):

{{
  "temporal_scope": "<year / fiscal year / date range, or null>",
  "semantic_components": ["component 1", "component 2", ...],
  "table_reasoning": {{
    "<table_name>": {{
      "matched_components":  ["component i"],
      "matched_columns":     ["col_a"],
      "temporal_consistent": true,
      "verdict":             "keep",
      "reason":              "<one-sentence justification>"
    }}
  }},
  "selected_tables": ["table1", "table2", ...]
}}"""

        llm_interaction = {
            "prompt": prompt,
            "response": None,
            "parsed_result": None,
            "error": None,
            "usage": None
        }

        try:
            resp = await self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system",
                     "content": "You are a data lake schema expert. Your task is to identify "
                                "from a set of candidate tables the minimal and semantically "
                                "correct set required to answer a given natural language query. "
                                "Reason at the column level: a table is relevant if and only if "
                                "its columns directly supply information requested by the query."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.1,
                max_tokens=8000
            )

            content = resp.choices[0].message.content.strip()
            llm_interaction["response"] = content

            global TOTAL_PROMPT_TOKENS, TOTAL_COMPLETION_TOKENS
            if resp.usage:
                TOTAL_PROMPT_TOKENS += resp.usage.prompt_tokens
                TOTAL_COMPLETION_TOKENS += resp.usage.completion_tokens
                llm_interaction["usage"] = {
                    "prompt_tokens": resp.usage.prompt_tokens,
                    "completion_tokens": resp.usage.completion_tokens,
                    "total_tokens": resp.usage.total_tokens
                }

            content = re.sub(r'```json\s*', '', content)
            content = re.sub(r'```\s*', '', content)

            json_match = re.search(r'\{.*\}', content, re.DOTALL)
            result = json.loads(json_match.group(0) if json_match else content)
            llm_interaction["parsed_result"] = result

            selected_tables = result.get('selected_tables', [])

            if len(selected_tables) == 0:
                selected_tables = [t for t, s in candidates[:10]]

            return {
                'filtered_tables': selected_tables,
                'temporal_scope': result.get('temporal_scope', 'null'),
                'summary': result.get('summary', {}),
                'llm_interaction': llm_interaction
            }

        except Exception as e:
            llm_interaction["error"] = str(e)
            return {
                'filtered_tables': [t for t, s in candidates[:10]],
                'temporal_scope': 'error',
                'summary': {},
                'llm_interaction': llm_interaction
            }


def load_all_table_columns(stats_file: str) -> dict:
    with open(stats_file, 'r', encoding='utf-8') as f:
        stats = json.load(f)

    table_columns = {}
    for category in ['normal', 'no_overlap', 'no_join_cols']:
        for table_info in stats.get(category, []):
            parent_name = table_info['table_name']
            if 'node_mapping' in table_info:
                for node_id, node_info in table_info['node_mapping'].items():
                    subtable_name = node_info['subtable_file']
                    columns = node_info.get('columns', [])

                    table_columns[subtable_name] = columns

                    if subtable_name.startswith('node_'):
                        full_name = f"{parent_name}_{subtable_name}"
                        table_columns[full_name] = columns

    return table_columns


def prepare_data(results_file: str, all_columns: dict) -> list:

    with open(results_file, 'r', encoding='utf-8') as f:
        results_data = json.load(f)

    prepared_data = []
    for sample in results_data:
        column_info = {}
        for table_name, score in sample['candidates']:
            if table_name in all_columns:
                column_info[table_name] = {col: '' for col in all_columns[table_name]}

        prepared_data.append({
            'sample_id': sample['sample_id'],
            'query': sample['query'],
            'sql': sample.get('sql', ''),
            'candidates': sample['candidates'],
            'column_info': column_info,
            'ground_truth': sample['gt']
        })

    return prepared_data


async def main():

    API_KEY = ""
    API_BASE = ""
    MODEL = ""

    filter_obj = LLMTableFilter(
        api_key=API_KEY,
        model=MODEL,
        base_url=API_BASE
    )

    results_file = ''
    stats_file = ''

    all_data = prepare_data(results_file, all_columns)

    num_samples = int(sys.argv[1]) if len(sys.argv) > 1 else None
    if num_samples and num_samples < len(all_data):
        import random
        random.seed(42)
        sample_indices = random.sample(range(len(all_data)), num_samples)
        sample_data = [all_data[i] for i in sorted(sample_indices)]
    else:
        sample_data = all_data
        sample_indices = list(range(len(all_data)))


    results = []
    for idx, item in enumerate(sample_data, 1):

        filter_result = await filter_obj.filter_candidates(
            item['query'], item['candidates'], item['column_info']
        )

        selected_tables = filter_result['filtered_tables']

        gt_set = set(item['ground_truth'])
        pred_set = set(selected_tables)

        # Stage1 Top-30 metrics
        top30_set = set([t for t, s in item['candidates'][:30]])
        top30_tp = len(gt_set & top30_set)
        top30_fp = len(top30_set - gt_set)
        top30_fn = len(gt_set - top30_set)
        top30_precision = top30_tp / (top30_tp + top30_fp) if (top30_tp + top30_fp) > 0 else 0
        top30_recall = top30_tp / (top30_tp + top30_fn) if (top30_tp + top30_fn) > 0 else 0
        top30_f1 = 2 * top30_precision * top30_recall / (top30_precision + top30_recall) if (top30_precision + top30_recall) > 0 else 0

        # LLM Filter metrics
        tp = len(gt_set & pred_set)
        fp = len(pred_set - gt_set)
        fn = len(gt_set - pred_set)
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

        print(f"{Fore.GREEN}Selected ({len(selected_tables)}): {selected_tables}{Style.RESET_ALL}")
        print(f"{Fore.CYAN}[Stage1 Top-30] P: {top30_precision:.4f}, R: {top30_recall:.4f}, F1: {top30_f1:.4f} ({len(top30_set)} tables){Style.RESET_ALL}")
        print(f"{Fore.MAGENTA}[LLM Filter]    P: {precision:.4f}, R: {recall:.4f}, F1: {f1:.4f} ({len(selected_tables)} tables){Style.RESET_ALL}")

        if fn > 0:
            print(f"{Fore.RED}Missing: {gt_set - pred_set}{Style.RESET_ALL}")
        if fp > 0:
            print(f"{Fore.YELLOW}Extra: {pred_set - gt_set}{Style.RESET_ALL}")

        results.append({
            'sample_id': item['sample_id'],
            'query': item['query'],
            'sql': item['sql'],
            'ground_truth': item['ground_truth'],
            'candidates_top30': item['candidates'][:30],
            'selected_tables': selected_tables,
            'num_selected': len(selected_tables),
            'year_detected': filter_result.get('temporal_scope', 'null'),
            'summary': filter_result['summary'],
            'llm_interaction': filter_result['llm_interaction'],
            'stage1_metrics': {
                'precision': top30_precision,
                'recall': top30_recall,
                'f1': top30_f1,
                'num_tables': len(top30_set)
            },
            'llm_filter_metrics': {
                'precision': precision,
                'recall': recall,
                'f1': f1,
                'tp': tp,
                'fp': fp,
                'fn': fn
            }
        })

        await asyncio.sleep(1.0)

    avg_s1_precision = np.mean([r['stage1_metrics']['precision'] for r in results])
    avg_s1_recall = np.mean([r['stage1_metrics']['recall'] for r in results])
    avg_s1_f1 = np.mean([r['stage1_metrics']['f1'] for r in results])

    avg_precision = np.mean([r['llm_filter_metrics']['precision'] for r in results])
    avg_recall = np.mean([r['llm_filter_metrics']['recall'] for r in results])
    avg_f1 = np.mean([r['llm_filter_metrics']['f1'] for r in results])
    avg_num_selected = np.mean([r['num_selected'] for r in results])

    cost_input = TOTAL_PROMPT_TOKENS / 1_000_000 * 2.5
    cost_output = TOTAL_COMPLETION_TOKENS / 1_000_000 * 10

    output_dir = ''
    output_file = os.path.join(output_dir, f'llm_filter_clean_863_{len(results)}samples_final.json')

    output_data = {
        'metadata': {
            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'model': MODEL,
            'source': results_file,
            'total_samples': len(results),
            'sample_indices': sample_indices,
            'token_usage': {
                'prompt_tokens': TOTAL_PROMPT_TOKENS,
                'completion_tokens': TOTAL_COMPLETION_TOKENS,
                'total_tokens': TOTAL_PROMPT_TOKENS + TOTAL_COMPLETION_TOKENS,
                'estimated_cost_usd': cost_input + cost_output
            }
        },
        'stage1_metrics': {
            'precision': avg_s1_precision,
            'recall': avg_s1_recall,
            'f1': avg_s1_f1
        },
        'llm_filter_metrics': {
            'precision': avg_precision,
            'recall': avg_recall,
            'f1': avg_f1,
            'avg_num_selected': avg_num_selected
        },
        'detailed_results': make_serializable(results)
    }

    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(output_data, f, ensure_ascii=False, indent=2)

if __name__ == "__main__":
    asyncio.run(main())
