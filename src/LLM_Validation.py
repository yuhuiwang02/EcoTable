import json
import numpy as np
from typing import List, Dict, Optional, Set, Tuple
import asyncio
import openai
import time
from sklearn.metrics import precision_recall_fscore_support
import re
import os
from tqdm import tqdm
from datetime import datetime
from colorama import init, Fore, Back, Style
from math import log2

# 初始化colorama用于彩色输出
init(autoreset=True)


def make_serializable(obj, max_depth=10, current_depth=0):
    """将对象转换为可JSON序列化的格式，避免循环引用"""
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
    
    # 对于其他类型，尝试转换为字符串
    try:
        return str(obj)
    except:
        return f"<non-serializable: {type(obj).__name__}>"


def _normalize(s: str) -> str:
    return re.sub(r'\s+', ' ', s.strip().lower())


def extract_explicit_terms_from_query(query: str) -> Set[str]:
    """从 query 中抽取可能的列名/关键词"""
    q = _normalize(query)
    tokens = set(re.findall(r'[a-zA-Z_][a-zA-Z0-9_]{1,}', q))
    hints = {'id', 'name', 'code', 'date', 'time', 'amount', 'price',
             'count', 'qty', 'quantity', 'city', 'state', 'region',
             'revenue', 'sales', 'total', 'avg', 'mean', 'sum', 'min', 'max'}
    for h in hints:
        if h in q:
            tokens.add(h)
    return tokens


def dcg_select(candidates: List[Dict],
               max_k: int = 20,
               min_gain_ratio: float = 0.15) -> List[str]:
    """基于 DCG 的初选"""
    if not candidates:
        return []
    cands = sorted(candidates, key=lambda x: x['score'], reverse=True)
    selected = []
    first_gain = max(cands[0]['score'], 0.0)
    first_discounted = first_gain / log2(1 + 1)
    for idx, c in enumerate(cands[:max_k], start=1):
        gain = max(c['score'], 0.0)
        discounted_gain = gain / log2(idx + 1)
        if idx == 1:
            selected.append(c['table_name'])
            continue
        if first_discounted <= 0:
            if discounted_gain > 0:
                selected.append(c['table_name'])
            else:
                break
        else:
            if discounted_gain >= first_discounted * min_gain_ratio:
                selected.append(c['table_name'])
            else:
                break
    return selected


class LLMTwoStepFilter:
    """改进后的两步筛选（DCG 初选 → 第一步列检验 → 第二步漏选补全）"""

    def __init__(self, api_key=None, model="gpt-4o", base_url=None,
                 initial_threshold: float = 0.0,
                 dcg_max_k: int = 20,
                 dcg_min_gain_ratio: float = 0.15,
                 save_llm_process: bool = True):
        self.client = openai.AsyncOpenAI(
            api_key=api_key or os.getenv("OPENAI_API_KEY"),
            base_url=base_url or os.getenv("OPENAI_BASE_URL")
        )
        self.model = model
        self.initial_threshold = initial_threshold
        self.dcg_max_k = dcg_max_k
        self.dcg_min_gain_ratio = dcg_min_gain_ratio
        self.save_llm_process = save_llm_process

    async def step1_column_filter(self, query: str, candidates: List[Dict],
                          top_k: int = 50) -> Dict:
        """第一步：仅针对"DCG选中的表"做列相关性检验"""
        if not candidates:
            return {"filtered_tables": [], "column_analysis": {}, "query_entities": [], "llm_interaction": None}

        top_candidates = candidates[:top_k]
        query_terms = sorted(list(extract_explicit_terms_from_query(query)))[:50]

        prompt = f"""You are a database schema expert. Your task is to identify which tables are ACTUALLY NEEDED to answer a specific query.

    # QUERY
    {query}

    # CONTEXT
    Heuristic Query Terms: {', '.join(query_terms) if query_terms else '(none detected)'}
    (These are auto-extracted keywords - use as hints only, not absolute truth)

    # CANDIDATE TABLES
    Below are {len(top_candidates)} pre-selected tables with their columns and relevance scores:

    """
        # 分批展示，每批10-15个表，提高可读性
        for i, c in enumerate(top_candidates, 1):
            cols = c.get('columns', [])
            score = c.get('score', 0)
            prompt += f"""
    Table #{i}: `{c['table_name']}` (Relevance Score: {score:.3f})
    Columns: {', '.join(cols[:100])}
    """
            if len(cols) > 100:
                prompt += f"... (and {len(cols) - 100} more columns)\n"

        prompt += """

    # YOUR TASK
    Analyze each table and determine if it should be KEPT or REMOVED based on these criteria:

    ## ✅ KEEP a table if:
    1. **Direct Column Match**: Table has columns explicitly mentioned in the query (e.g., query says "customer_name" and table has that column)
    2. **Semantic Match**: Table contains columns semantically related to query intent
    - Example: Query asks for "revenue" → table has "sales_amount" or "total_price"
    3. **Aggregation Source**: Table contains measures needed for calculations (SUM, AVG, COUNT, etc.)
    4. **Filter/Grouping Dimension**: Table provides grouping or filtering attributes mentioned in query
    5. **Query-Critical Tables**: **IMPORTANT - Even if another table already provides similar columns, KEEP this table if it is SEMANTICALLY IMPORTANT to the query context**
    - Example: Query mentions "customer orders" → keep BOTH customer_table AND order_table even if they share columns like customer_id
    - Example: Query needs "sales by region" → keep BOTH sales_table AND region_table even if sales has region_id
    - **Do NOT remove a table just because its columns appear elsewhere - consider the TABLE'S SEMANTIC ROLE in answering the query**

    ## ❌ REMOVE a table if:
    1. **Zero Column Relevance**: NO columns relate to query semantics AND the table has no semantic role in the query
    2. **Pure Metadata**: Only contains system columns (created_at, updated_at, id) with no business meaning AND not referenced in query

    ## 🔍 SPECIAL CASES - Be Careful:
    - **Lookup Tables**: Small dimension tables (e.g., status_codes, categories) - keep if referenced
    - **Denormalized Tables**: May contain duplicated info - **keep BOTH if the query semantically refers to both tables**, prefer the one with more relevant columns ONLY if one is clearly redundant
    - **Column Duplication is OK**: If a query mentions or implies multiple entities (customers, orders, products), keep ALL relevant tables even if they share key columns

    ## 🚨 CRITICAL PRINCIPLE:
    **Prioritize SEMANTIC RELEVANCE over COLUMN UNIQUENESS**
    - A table's importance depends on whether the QUERY CONCEPT refers to it, not just whether its columns are unique
    - When in doubt, KEEP the table if it matches query semantics

    # DECISION FRAMEWORK
    For EACH table, follow this process:
    1. Identify which columns (if any) are relevant to the query
    2. Determine the table's SEMANTIC ROLE (fact/dimension/bridge/lookup) and if the query refers to this entity
    3. Assess if the table is ESSENTIAL (query explicitly needs it), HELPFUL (query implies it), or IRRELEVANT (no semantic connection)
    4. **Check: Even if columns exist elsewhere, does the QUERY SEMANTICS require this table's perspective/entity?**
    5. Make final KEEP/REMOVE decision with clear reasoning

    # OUTPUT FORMAT
    Provide a valid JSON object with this exact structure:

    {{
    "query_analysis": {{
        "main_intent": "What is the query trying to achieve?",
        "key_entities": ["entity1", "entity2"],
        "required_operations": ["aggregation", "join", "filter"],
        "expected_table_types": ["fact table", "customer dimension"]
    }},
    "table_decisions": {{
        "table_name_1": {{
        "decision": "KEEP",
        "confidence": "HIGH",
        "relevant_columns": ["col1", "col2", "col3"],
        "role": "fact_table",
        "reasoning": "Contains sales_amount needed for revenue calculation and has date column for filtering. Even though order_table has similar columns, this table is semantically important as the query explicitly asks about sales data."
        }},
        "table_name_2": {{
        "decision": "REMOVE",
        "confidence": "HIGH",
        "relevant_columns": [],
        "role": "metadata",
        "reasoning": "Only contains system audit columns (created_by, updated_at) with no business data relevant to sales query AND no semantic role in the query context."
        }}
    }},
    "filtered_tables": ["table_name_1", "table_name_3"],
    "summary": {{
        "total_analyzed": {len(top_candidates)},
        "kept": 5,
        "removed": {len(top_candidates) - 5},
        "confidence_distribution": {{"HIGH": 7, "MEDIUM": 2, "LOW": 1}}
    }}
    }}

    # IMPORTANT REMINDERS
    - **Do NOT remove tables just because their columns appear in other tables - focus on SEMANTIC IMPORTANCE**
    - Be SELECTIVE but not overly aggressive - when the query implies multiple entities, keep all semantically relevant tables
    - If confidence is LOW, explain uncertainty in reasoning
    - Consider the ENTIRE query context and the semantic relationships between entities
    - **Column overlap is acceptable**
    - Output ONLY valid JSON, no markdown formatting or extra text


    Now analyze the tables:"""

        llm_interaction = {
            "step": "step1_column_filter",
            "prompt": prompt,
            "response": None,
            "parsed_result": None,
            "error": None
        }

        try:
            resp = await self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system",
                    "content": "You are an expert database architect specializing in query optimization and schema analysis. You provide precise, well-reasoned decisions about table relevance."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.2,  # 稍微提高以允许更灵活的推理
                max_completion_tokens=12000  # 增加token限制
            )
            content = resp.choices[0].message.content.strip()
            llm_interaction["response"] = content
            
            # 清理可能的markdown格式
            content = re.sub(r'```json\s*', '', content)
            content = re.sub(r'```\s*', '', content)
            
            json_match = re.search(r'\{.*\}', content, re.DOTALL)
            result = json.loads(json_match.group(0) if json_match else content)
            llm_interaction["parsed_result"] = result

            # 提取最终筛选结果
            filtered_tables = []
            table_decisions = result.get('table_decisions', {})
            
            for table_name, decision_info in table_decisions.items():
                if decision_info.get('decision') == 'KEEP':
                    filtered_tables.append(table_name)
            
            # 如果JSON中有filtered_tables字段，以它为准
            if 'filtered_tables' in result and result['filtered_tables']:
                filtered_tables = result['filtered_tables']

            return {
                'filtered_tables': filtered_tables,
                'column_analysis': table_decisions,
                'query_entities': result.get('query_analysis', {}).get('key_entities', []),
                'query_analysis': result.get('query_analysis', {}),
                'summary': result.get('summary', {}),
                'llm_interaction': llm_interaction if self.save_llm_process else None
            }
            
        except Exception as e:
            llm_interaction["error"] = str(e)
            print(f"{Fore.RED}Step1 LLM调用失败: {e}{Style.RESET_ALL}")
            # 失败时保守策略：保留所有候选表
            return {
                "filtered_tables": [c['table_name'] for c in candidates],
                "column_analysis": {},
                "query_entities": list(query_terms),
                "llm_interaction": llm_interaction if self.save_llm_process else None
            }
    async def step2_supplement_tables(self, query: str,
                                      selected_tables: List[str],
                                      all_tables_info: Dict[str, List[str]],
                                      column_analysis_step1: Dict) -> Dict:
        """第二步：在剩余表里查找漏选表并补入"""
        selected_set = set(selected_tables)
        remaining = [(t, cols) for t, cols in all_tables_info.items() if t not in selected_set]
        if not remaining:
            return {"supplemented_tables": selected_tables, "additions": [], "analysis": {}, "llm_interaction": None}

        related_cols_from_step1 = {}
        for t, info in (column_analysis_step1 or {}).items():
            if info and info.get('relevant_columns'):
                related_cols_from_step1[t] = info['relevant_columns'][:25]

        query_terms = sorted(list(extract_explicit_terms_from_query(query)))[:60]
        show_rest_max = 40
        remaining_show = remaining[:show_rest_max]

        def likely_keys(cols: List[str]) -> List[str]:
            keys = []
            for c in cols:
                cl = c.lower()
                if cl == 'id' or cl.endswith('_id') or cl.endswith('id') or cl.endswith('_code') or cl.endswith('code'):
                    keys.append(c)
            return keys[:10]

        prompt = f"""QUERY: {query}

QUERY TERMS (heuristic): {', '.join(query_terms) if query_terms else '(none)'}

CURRENTLY SELECTED TABLES (after Step 1) WITH RELEVANT COLUMNS:
"""
        if selected_tables:
            for t in selected_tables[:30]:
                cols = all_tables_info.get(t, [])[:80]
                rel = related_cols_from_step1.get(t, [])
                prompt += f"\n- {t}\n  columns: {', '.join(cols)}\n  relevant_in_query: {', '.join(rel) if rel else '(unspecified)'}\n  likely_keys: {', '.join(likely_keys(cols))}\n"
        else:
            prompt += "(none)\n"

        prompt += f"""

POTENTIAL TABLES (NOT SELECTED YET) TO CONSIDER ({len(remaining)} total, showing {len(remaining_show)}):
"""
        for t, cols in remaining_show:
            prompt += f"\n* {t}\n  columns: {', '.join(cols[:80])}\n  likely_keys: {', '.join(likely_keys(cols))}\n"

        prompt += """
TASK:
From the NOT-YET-SELECTED tables above, identify any CRITICAL tables that are MISSING but ESSENTIAL for answering the query.
A table is "essential" if it provides UNIQUE information not present in the selected set, such as:
  - Required dimensions/measures explicitly or strongly implied by the query
  - Essential lookup/reference mappings needed to interpret selected columns
Be SELECTIVE — only add if absolutely necessary.

IMPORTANT:
Focus on tables whose columns directly relate to (a) the query-mentioned columns/terms, or (b) the "relevant_in_query" columns already identified in the selected tables.

OUTPUT FORMAT (JSON only):
{
  "additions": ["table1", "table2"],
  "analysis": {
    "table1": {
      "reason": "Contains essential key to link orders to customers",
      "columns_needed": ["customer_id", "region"],
      "critical": true
    }
  }
}
"""
        llm_interaction = {
            "step": "step2_supplement_tables",
            "prompt": prompt,
            "response": None,
            "parsed_result": None,
            "error": None
        }

        try:
            resp = await self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system",
                     "content": "You are a database expert. Identify missing essential tables for query execution, focusing on query-mentioned columns and join keys."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.1,
                max_completion_tokens=8192
            )
            content = resp.choices[0].message.content.strip()
            llm_interaction["response"] = content
            
            json_match = re.search(r'\{.*\}', content, re.DOTALL)
            parsed = json.loads(json_match.group(0) if json_match else content)
            llm_interaction["parsed_result"] = parsed

            additions = [t for t in parsed.get('additions', []) if t not in selected_set]
            supplemented = list(selected_tables) + additions
            return {
                "supplemented_tables": supplemented,
                "additions": additions,
                "analysis": parsed.get('analysis', {}),
                "llm_interaction": llm_interaction if self.save_llm_process else None
            }
        except Exception as e:
            llm_interaction["error"] = str(e)
            print(f"{Fore.RED}Step2 LLM调用失败: {e}{Style.RESET_ALL}")
            return {
                "supplemented_tables": selected_tables,
                "additions": [],
                "analysis": {},
                "llm_interaction": llm_interaction if self.save_llm_process else None
            }

    def calculate_metrics(self, predicted: List[str], ground_truth: List[str],
                          all_tables: List[str]) -> Optional[Dict]:
        y_true = [1 if t in ground_truth else 0 for t in all_tables]
        y_pred = [1 if t in predicted else 0 for t in all_tables]
        if sum(y_true) == 0:
            return None
        precision, recall, f1, _ = precision_recall_fscore_support(
            y_true, y_pred, average='binary', zero_division=0
        )
        predicted_set = set(predicted)
        gt_set = set(ground_truth)
        return {
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'predicted_count': len(predicted),
            'ground_truth_count': len(ground_truth),
            'correct_count': len(predicted_set & gt_set),
            'false_positives': list(predicted_set - gt_set),
            'false_negatives': list(gt_set - predicted_set)
        }

    async def batch_process(self, data_list: List[Dict], delay: float = 1.0) -> List[Dict]:
        results = []
        pbar = tqdm(total=len(data_list), desc="处理进度", ncols=100)

        cumulative_stats = {
            'initial': {'p': [], 'r': [], 'f1': []},
            'step1': {'p': [], 'r': [], 'f1': []},
            'step2': {'p': [], 'r': [], 'f1': []}
        }

        for idx, item in enumerate(data_list, 1):
            pbar.set_description(f"处理 Query {idx}/{len(data_list)}")

            query = item['query']
            contrastive_results = item['contrastive_results']
            ground_truth = item.get('ground_truth', [])

            # 准备候选表
            candidates = []
            table_scores = contrastive_results.get('table_scores', {})
            for table_name, score in table_scores.items():
                candidates.append({
                    'table_name': table_name,
                    'score': float(score),
                    'columns': list(item['column_relevance'].get(table_name, {}).keys())
                })

            # DCG 初选
            dcg_selected = dcg_select(
                candidates,
                max_k=self.dcg_max_k,
                min_gain_ratio=self.dcg_min_gain_ratio
            )

            # 评估：DCG 初选
            all_tables = list(item['table_relevance'].keys())
            initial_tables = list(dcg_selected)
            initial_metrics = self.calculate_metrics(initial_tables, ground_truth, all_tables)

            # 第一步：列级别筛选
            step1_candidates = [c for c in candidates if c['table_name'] in dcg_selected]
            step1_result = await self.step1_column_filter(query, step1_candidates)
            step1_tables = step1_result.get('filtered_tables', [])
            step1_metrics = self.calculate_metrics(step1_tables, ground_truth, all_tables)

            # 第二步：漏选补充
            all_tables_info = {t: list(cols.keys()) for t, cols in item['column_relevance'].items()}
            step2_result = await self.step2_supplement_tables(
                query=query,
                selected_tables=step1_tables,
                all_tables_info=all_tables_info,
                column_analysis_step1=step1_result.get('column_analysis', {})
            )
            step2_tables = step2_result['supplemented_tables']
            step2_metrics = self.calculate_metrics(step2_tables, ground_truth, all_tables)

            # 累计统计
            if initial_metrics:
                cumulative_stats['initial']['p'].append(initial_metrics['precision'])
                cumulative_stats['initial']['r'].append(initial_metrics['recall'])
                cumulative_stats['initial']['f1'].append(initial_metrics['f1'])
            if step1_metrics:
                cumulative_stats['step1']['p'].append(step1_metrics['precision'])
                cumulative_stats['step1']['r'].append(step1_metrics['recall'])
                cumulative_stats['step1']['f1'].append(step1_metrics['f1'])
            if step2_metrics:
                cumulative_stats['step2']['p'].append(step2_metrics['precision'])
                cumulative_stats['step2']['r'].append(step2_metrics['recall'])
                cumulative_stats['step2']['f1'].append(step2_metrics['f1'])

            # 保存LLM思考过程
            llm_interactions = {
                'step1': step1_result.get('llm_interaction'),
                'step2': step2_result.get('llm_interaction')
            }

            # 打印LLM思考过程（可选）
            if self.save_llm_process and idx <= 3:  # 只打印前3个样本
                print(f"\n{Fore.CYAN}{'='*80}{Style.RESET_ALL}")
                print(f"{Fore.GREEN}Query {idx} LLM思考过程:{Style.RESET_ALL}")
                
                try:
                    # 使用清理函数避免循环引用
                    safe_interactions = make_serializable(llm_interactions)
                    print(json.dumps(safe_interactions, ensure_ascii=False, indent=2))
                except Exception as e:
                    print(f"{Fore.YELLOW}打印LLM交互数据失败: {e}{Style.RESET_ALL}")
                    # 备用方案：只打印关键信息
                    for step_name, interaction in llm_interactions.items():
                        if interaction:
                            print(f"\n{step_name}:")
                            print(f"  - Prompt前100字符: {str(interaction.get('prompt', ''))[:100]}")
                            print(f"  - Response前100字符: {str(interaction.get('response', ''))[:100]}")
                
                print(f"{Fore.CYAN}{'='*80}{Style.RESET_ALL}")

            results.append({
                'query': query,
                'ground_truth': ground_truth,
                'dcg_params': {'max_k': self.dcg_max_k, 'min_gain_ratio': self.dcg_min_gain_ratio},
                'initial_tables': initial_tables,
                'initial_metrics': initial_metrics,
                'step1_filtered': step1_tables,
                'step1_column_analysis': step1_result.get('column_analysis', {}),
                'step1_metrics': step1_metrics,
                'step2_final': step2_tables,
                'step2_additions': step2_result.get('additions', []),
                'step2_analysis': step2_result.get('analysis', {}),
                'step2_metrics': step2_metrics,
                'llm_interactions': llm_interactions,  # 保存LLM交互过程
                'improvements': {
                    'step1_vs_initial': {
                        'precision_change': (step1_metrics['precision'] - initial_metrics['precision'])
                        if step1_metrics and initial_metrics else None,
                        'recall_change': (step1_metrics['recall'] - initial_metrics['recall'])
                        if step1_metrics and initial_metrics else None,
                        'f1_change': (step1_metrics['f1'] - initial_metrics['f1'])
                        if step1_metrics and initial_metrics else None
                    },
                    'step2_vs_step1': {
                        'precision_change': (step2_metrics['precision'] - step1_metrics['precision'])
                        if step2_metrics and step1_metrics else None,
                        'recall_change': (step2_metrics['recall'] - step1_metrics['recall'])
                        if step2_metrics and step1_metrics else None,
                        'f1_change': (step2_metrics['f1'] - step1_metrics['f1'])
                        if step2_metrics and step1_metrics else None
                    }
                }
            })

            pbar.update(1)
            if delay > 0:
                await asyncio.sleep(delay)

        pbar.close()
        
        # 打印累计统计
        print(f"\n{Fore.CYAN}累计统计 (所有{len(data_list)}个样本):{Style.RESET_ALL}")
        print(f"  初始(DCG): P={np.mean(cumulative_stats['initial']['p']):.4f} "
              f"R={np.mean(cumulative_stats['initial']['r']):.4f} "
              f"F1={np.mean(cumulative_stats['initial']['f1']):.4f}")
        print(f"  步骤1: P={np.mean(cumulative_stats['step1']['p']):.4f} "
              f"R={np.mean(cumulative_stats['step1']['r']):.4f} "
              f"F1={np.mean(cumulative_stats['step1']['f1']):.4f}")
        print(f"  步骤2: P={np.mean(cumulative_stats['step2']['p']):.4f} "
              f"R={np.mean(cumulative_stats['step2']['r']):.4f} "
              f"F1={np.mean(cumulative_stats['step2']['f1']):.4f}")
        
        return results


def prepare_data_for_llm_filter(contrastive_results_file: str,
                                original_data_file: str) -> List[Dict]:
    print(f"{Fore.CYAN}加载数据文件...{Style.RESET_ALL}")
    with open(contrastive_results_file, 'r', encoding='utf-8') as f:
        contrastive_results = json.load(f)
    with open(original_data_file, 'r', encoding='utf-8') as f:
        original_data = json.load(f)

    query_index = {}
    for project_name, project_data in original_data.items():
        for item in project_data:
            query_key = _normalize(item['query'])
            query_index[query_key] = item
            if len(query_key) > 100:
                query_index[query_key[:100]] = item

    prepared_data = []
    not_found = 0
    for contrast_result in contrastive_results:
        query = contrast_result['query']
        query_key = _normalize(query)
        original_item = query_index.get(query_key)
        if not original_item and len(query_key) > 100:
            original_item = query_index.get(query_key[:100])

        if not original_item:
            not_found += 1
            continue

        table_scores_raw = contrast_result.get('table_scores', [])
        if isinstance(table_scores_raw, list):
            table_scores = {item[0]: float(item[1]) for item in table_scores_raw if len(item) >= 2}
        elif isinstance(table_scores_raw, dict):
            table_scores = {k: float(v) for k, v in table_scores_raw.items()}
        else:
            table_scores = {}

        prepared_data.append({
            'query': query,
            'contrastive_results': {
                'query': contrast_result['query'],
                'table_scores': table_scores,
                'threshold_used': contrast_result.get('threshold', 0.0)
            },
            'ground_truth': original_item.get('ground_truth', []),
            'table_relevance': original_item.get('table_relevance', {}),
            'column_relevance': original_item.get('column_relevance', {})
        })

    print(f"{Fore.GREEN}数据匹配完成: {len(prepared_data)}/{len(contrastive_results)}{Style.RESET_ALL}")
    if not_found > 0:
        print(f"{Fore.YELLOW}未匹配: {not_found} 条{Style.RESET_ALL}")
    return prepared_data


def print_final_analysis(results: List[Dict], threshold: float = 0.0):
    valid = [r for r in results if r['initial_metrics'] and r['step1_metrics'] and r['step2_metrics']]
    if not valid:
        print(f"{Fore.RED}没有有效的评估结果{Style.RESET_ALL}")
        return {}

    print(f"\n{Fore.CYAN}{'='*80}{Style.RESET_ALL}")
    print(f"{Fore.GREEN}最终汇总分析{Style.RESET_ALL}")
    print(f"{Fore.CYAN}{'='*80}{Style.RESET_ALL}")

    analysis = {
        'sample_count': len(valid),
        'initial_threshold': threshold,
        'initial_metrics': {
            'avg_precision': np.mean([r['initial_metrics']['precision'] for r in valid]),
            'avg_recall': np.mean([r['initial_metrics']['recall'] for r in valid]),
            'avg_f1': np.mean([r['initial_metrics']['f1'] for r in valid]),
            'std_precision': np.std([r['initial_metrics']['precision'] for r in valid]),
            'std_recall': np.std([r['initial_metrics']['recall'] for r in valid]),
            'std_f1': np.std([r['initial_metrics']['f1'] for r in valid]),
            'avg_table_count': np.mean([r['initial_metrics']['predicted_count'] for r in valid])
        },
        'step1_metrics': {
            'avg_precision': np.mean([r['step1_metrics']['precision'] for r in valid]),
            'avg_recall': np.mean([r['step1_metrics']['recall'] for r in valid]),
            'avg_f1': np.mean([r['step1_metrics']['f1'] for r in valid]),
            'std_precision': np.std([r['step1_metrics']['precision'] for r in valid]),
            'std_recall': np.std([r['step1_metrics']['recall'] for r in valid]),
            'std_f1': np.std([r['step1_metrics']['f1'] for r in valid]),
            'tables_removed': np.mean([len(r['initial_tables']) - len(r['step1_filtered']) for r in valid])
        },
        'step2_metrics': {
            'avg_precision': np.mean([r['step2_metrics']['precision'] for r in valid]),
            'avg_recall': np.mean([r['step2_metrics']['recall'] for r in valid]),
            'avg_f1': np.mean([r['step2_metrics']['f1'] for r in valid]),
            'std_precision': np.std([r['step2_metrics']['precision'] for r in valid]),
            'std_recall': np.std([r['step2_metrics']['recall'] for r in valid]),
            'std_f1': np.std([r['step2_metrics']['f1'] for r in valid]),
            'tables_added': np.mean([len(r['step2_additions']) for r in valid])
        }
    }

    print(f"\n{Fore.YELLOW}样本数: {analysis['sample_count']}{Style.RESET_ALL}")
    print(f"{'-'*70}")
    print(f"{'阶段':<20} {'精确率':^15} {'召回率':^15} {'F1分数':^15}")
    print(f"{'-'*70}")
    print(f"{'初始(DCG)':<20} {analysis['initial_metrics']['avg_precision']:^15.4f} "
          f"{analysis['initial_metrics']['avg_recall']:^15.4f} "
          f"{analysis['initial_metrics']['avg_f1']:^15.4f}")
    print(f"{'步骤1(列筛选)':<20} {analysis['step1_metrics']['avg_precision']:^15.4f} "
          f"{analysis['step1_metrics']['avg_recall']:^15.4f} "
          f"{analysis['step1_metrics']['avg_f1']:^15.4f}")
    print(f"{'步骤2(补充表)':<20} {analysis['step2_metrics']['avg_precision']:^15.4f} "
          f"{analysis['step2_metrics']['avg_recall']:^15.4f} "
          f"{analysis['step2_metrics']['avg_f1']:^15.4f}")
    print(f"{'-'*70}")

    tf1 = analysis['step2_metrics']['avg_f1'] - analysis['initial_metrics']['avg_f1']
    print(f"\n{Fore.GREEN}F1改进: {tf1:+.4f}{Style.RESET_ALL}")
    print(f"{Fore.CYAN}{'='*80}{Style.RESET_ALL}")
    return analysis


async def main():
    print(f"{Fore.CYAN}{'='*80}{Style.RESET_ALL}")
    print(f"{Fore.GREEN}LLM两步筛选系统（保存LLM思考过程）{Style.RESET_ALL}")
    print(f"{Fore.CYAN}{'='*80}{Style.RESET_ALL}\n")

    DCG_MAX_K = 20
    DCG_MIN_GAIN_RATIO = 0.15

    filter_obj = LLMTwoStepFilter(
        api_key="",
        model="gpt-5",
        base_url="https://aihubmix.com/v1",
        dcg_max_k=DCG_MAX_K,
        dcg_min_gain_ratio=DCG_MIN_GAIN_RATIO,
        save_llm_process=True  # 开启LLM过程保存
    )

    contrastive_results_file = '/data2/liujinqi/AutoETL/AutoETL-main/data/schema_linking_dataset/DCG_results/detailed_results_top50_singlepair.json'
    original_data_file = '/data2/liujinqi/AutoETL/AutoETL-main/data/schema_linking_dataset/SL_easy.json'

    data = prepare_data_for_llm_filter(contrastive_results_file, original_data_file)
    if not data:
        print(f"{Fore.RED}错误: 没有数据可处理{Style.RESET_ALL}")
        return

    print(f"\n{Fore.GREEN}准备处理 {len(data)} 个样本{Style.RESET_ALL}")
    test_data = data[:88] #485
    print(f"{Fore.YELLOW}实际处理: {len(test_data)} 个样本{Style.RESET_ALL}\n")

    results = await filter_obj.batch_process(test_data, delay=1.0)

    analysis = print_final_analysis(results, threshold=0.0)

    output_dir = '/data2/liujinqi/AutoETL/reimplementation/figures/no_contrastive_learning_easy_50'
    os.makedirs(output_dir, exist_ok=True)

    # 保存完整结果（包含LLM思考过程）
    out_json = os.path.join(output_dir, f'two_step_llm_filter_results_with_process.json')
    try:
        # 使用清理函数避免循环引用
        safe_results = make_serializable({
            'results': results,
            'analysis': analysis,
            'dcg_params': {'max_k': DCG_MAX_K, 'min_gain_ratio': DCG_MIN_GAIN_RATIO},
            'timestamp': time.strftime('%Y-%m-%d %H:%M:%S')
        })
        
        with open(out_json, 'w', encoding='utf-8') as f:
            json.dump(safe_results, f, ensure_ascii=False, indent=2)
        print(f"\n{Fore.GREEN}完整结果已保存到: {out_json}{Style.RESET_ALL}")
    except Exception as e:
        print(f"{Fore.RED}保存完整结果失败: {e}{Style.RESET_ALL}")

    # 单独保存LLM交互过程（方便分析）
    llm_process_json = os.path.join(output_dir, f'llm_interactions_only.json')
    try:
        llm_interactions_only = [
            {
                'query_id': idx,
                'query': r['query'],
                'ground_truth': r['ground_truth'],
                'llm_interactions': r['llm_interactions']
            }
            for idx, r in enumerate(results, 1)
        ]
        
        # 使用清理函数避免循环引用
        safe_llm_interactions = make_serializable(llm_interactions_only)
        
        with open(llm_process_json, 'w', encoding='utf-8') as f:
            json.dump(safe_llm_interactions, f, ensure_ascii=False, indent=2)
        print(f"{Fore.GREEN}LLM思考过程已单独保存到: {llm_process_json}{Style.RESET_ALL}")
    except Exception as e:
        print(f"{Fore.RED}保存LLM交互过程失败: {e}{Style.RESET_ALL}")

    print(f"\n{Fore.CYAN}{'='*80}{Style.RESET_ALL}")
    print(f"{Fore.GREEN}处理完成！{Style.RESET_ALL}")
    print(f"{Fore.CYAN}{'='*80}{Style.RESET_ALL}")


if __name__ == "__main__":
    asyncio.run(main())

def run_two_step_filter_by_paths(
    contrastive_results_file: str,
    original_data_file: str,
    output_dir: str,
) -> Dict[str, str]:
    """
    端到端入口（仅需路径）：
      - contrastive_results_file: 你的对比学习/检索阶段结果 JSON（含 table_scores）
      - original_data_file: 原始标注数据 JSON（含 table_relevance/column_relevance/ground_truth）
      - output_dir: 结果输出目录

    其余参数全部沿用原脚本默认值：
      * DCG_MAX_K = 20, DCG_MIN_GAIN_RATIO = 0.15
      * 仅处理前 88 个样本（与原 main 相同）
      * batch_process(delay=1.0)
      * LLMTwoStepFilter 使用类里的默认 model/base_url，
        以及环境变量 OPENAI_API_KEY / OPENAI_BASE_URL（如已设置）
    
    返回:
      {
        "results_json": "<two_step_llm_filter_results_with_process.json 的路径>",
        "llm_interactions_json": "<llm_interactions_only.json 的路径>"
      }
    """
    async def _inner() -> Dict[str, str]:
        # --- 与原 main 保持一致的默认设置 ---
        DCG_MAX_K = 20
        DCG_MIN_GAIN_RATIO = 0.15

        # 不显式传 api_key/model/base_url，默认走环境变量与类默认值
        filter_obj = LLMTwoStepFilter(
            dcg_max_k=DCG_MAX_K,
            dcg_min_gain_ratio=DCG_MIN_GAIN_RATIO,
            save_llm_process=True
        )

        # 准备数据（与原逻辑一致）
        data = prepare_data_for_llm_filter(contrastive_results_file, original_data_file)
        if not data:
            raise ValueError("没有数据可处理：请检查输入路径/文件内容")

        # 与原 main 一致：默认只处理前 88 条
        test_data = data[:88]

        # 执行两步筛选
        results = await filter_obj.batch_process(test_data, delay=1.0)

        # 统计汇总
        analysis = print_final_analysis(results, threshold=0.0)

        # 输出目录与文件名（保持原命名）
        os.makedirs(output_dir, exist_ok=True)
        out_json = os.path.join(output_dir, 'two_step_llm_filter_results_with_process.json')
        llm_process_json = os.path.join(output_dir, 'llm_interactions_only.json')

        # 保存完整结果（包含 LLM 过程）
        safe_results = make_serializable({
            'results': results,
            'analysis': analysis,
            'dcg_params': {'max_k': DCG_MAX_K, 'min_gain_ratio': DCG_MIN_GAIN_RATIO},
            'timestamp': time.strftime('%Y-%m-%d %H:%M:%S')
        })
        with open(out_json, 'w', encoding='utf-8') as f:
            json.dump(safe_results, f, ensure_ascii=False, indent=2)

        # 单独保存 LLM 交互过程
        llm_interactions_only = [
            {
                'query_id': idx,
                'query': r['query'],
                'ground_truth': r['ground_truth'],
                'llm_interactions': r['llm_interactions']
            }
            for idx, r in enumerate(results, 1)
        ]
        with open(llm_process_json, 'w', encoding='utf-8') as f:
            json.dump(make_serializable(llm_interactions_only), f, ensure_ascii=False, indent=2)

        return {
            "results_json": out_json,
            "llm_interactions_json": llm_process_json
        }

    # 同步封装：直接返回两个 JSON 的路径
    return asyncio.run(_inner())