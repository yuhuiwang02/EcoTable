"""
LLM-based join verification module

Provides coarse and fine-grained table join verification:
- Coarse verification: Uses sample data (first n rows) to identify candidate column pairs
- Fine verification: Validates candidate column pairs one by one in the same conversation context
"""

import json
import time
import os
from typing import Dict, List, Tuple, Optional, Any
import pandas as pd
from openai import OpenAI

# Import prompt templates
from .prompts import PROMPTS as DEFAULT_PROMPTS


class LLMVerifier:
    """LLM-based table join relationship verifier (supports conversation context)"""

    def __init__(self, api_key: Optional[str] = None, model: str = "gpt-4o",
                 sample_rows: int = 10,
                 custom_prompts: Dict[str, str] = None,
                 prompt_template_path: str = None,
                 base_url: str = "https://www.dmxapi.com/v1",
                 test_mode: bool = False):
        """
        Initialize LLM verifier

        Args:
            api_key: API key for LLM service (if None, read from environment variable)
            model: Name of the model to use
            sample_rows: Number of rows to use for coarse verification
            custom_prompts: Custom prompt dictionary, can include the following keys:
                          'coarse_system', 'coarse_user'
                          Note: Fine verification no longer uses independent prompt templates
                          If provided, will override default prompts
            prompt_template_path: Prompt template file path (for phase1 verification)
            base_url: API endpoint address
            test_mode: Test mode, only generate prompts without calling LLM
        """
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        self.model = model
        self.sample_rows = sample_rows
        self.test_mode = test_mode
        self.base_url = base_url

        # Initialize OpenAI client
        if not self.test_mode:
            self.client = OpenAI(api_key=self.api_key, base_url=self.base_url)
        else:
            self.client = None

        # Load prompt templates
        self.prompts = DEFAULT_PROMPTS.copy()

        # If custom prompts are provided, override default values
        if custom_prompts:
            self.prompts.update(custom_prompts)
            print(f"[LLM] Using custom prompts")

        # Load external prompt template file (for phase1/phase2 verification)
        self.prompt_template = None
        if prompt_template_path:
            if not os.path.exists(prompt_template_path):
                raise FileNotFoundError(f"Prompt template not found: {prompt_template_path}")
            with open(prompt_template_path, 'r', encoding='utf-8') as f:
                self.prompt_template = f.read()
            print(f"[LLM] Loaded prompt template: {prompt_template_path}")

        if self.test_mode:
            print(f"[LLM] ⚠️  Test mode enabled - Will only generate prompts without calling LLM API")

        # Conversation history management (each edge has its own conversation)
        self.conversations = {}  # {edge_key: [messages]}

        # Statistics tracking
        self.stats = {
            'total_calls': 0,
            'coarse_calls': 0,
            'fine_calls': 0,
            'total_tokens': 0,
            'input_tokens': 0,
            'output_tokens': 0,
            'total_latency': 0.0
        }

    @staticmethod
    def _get_display_name(table_path: str) -> str:
        """Extract table name for display from full path (keep last two path levels)"""
        parts = os.path.normpath(table_path).split(os.sep)
        if len(parts) >= 2:
            return os.path.join(parts[-2], parts[-1])
        return parts[-1]

    def _get_edge_key(self, table1_path: str, table2_path: str) -> str:
        """Generate unique identifier for edge"""
        tables = tuple(sorted([table1_path, table2_path]))
        return f"{tables[0]}<->{tables[1]}"

    def _init_conversation(self, edge_key: str, system_prompt: str):
        """Initialize conversation history"""
        self.conversations[edge_key] = [
            {"role": "system", "content": system_prompt}
        ]

    def _add_message(self, edge_key: str, role: str, content: str):
        """Add message to conversation history"""
        if edge_key not in self.conversations:
            raise ValueError(f"Conversation not initialized: {edge_key}")
        self.conversations[edge_key].append({
            "role": role,
            "content": content
        })

    def _call_llm_with_history(self, edge_key: str, user_message: str = None) -> Dict[str, Any]:
        """
        Call LLM API with conversation history

        Args:
            edge_key: Unique identifier for edge
            user_message: User message (if provided, will be added to history)

        Returns:
            Dictionary containing 'response', 'tokens', 'input_tokens', 'output_tokens', 'latency'
        """
        if edge_key not in self.conversations:
            raise ValueError(f"Conversation not initialized: {edge_key}")

        # If new message is provided, add to history
        if user_message:
            self._add_message(edge_key, "user", user_message)

        start_time = time.time()

        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=self.conversations[edge_key],
                temperature=0.1
            )
            result_text = response.choices[0].message.content
            tokens_used = response.usage.total_tokens
            input_tokens = response.usage.prompt_tokens
            output_tokens = response.usage.completion_tokens

        except Exception as e:
            print(f"[Error] LLM API call failed: {e}")
            raise

        latency = time.time() - start_time

        # Add assistant response to history
        self._add_message(edge_key, "assistant", result_text)

        # Update statistics
        self.stats['total_calls'] += 1
        self.stats['total_tokens'] += tokens_used
        self.stats['input_tokens'] += input_tokens
        self.stats['output_tokens'] += output_tokens
        self.stats['total_latency'] += latency

        return {
            'response': result_text,
            'tokens': tokens_used,
            'input_tokens': input_tokens,
            'output_tokens': output_tokens,
            'latency': latency
        }

    def _get_column_info(self, df: pd.DataFrame) -> str:
        """Get column information summary"""
        info = []
        for col in df.columns:
            dtype = df[col].dtype
            null_count = df[col].isnull().sum()
            unique_count = df[col].nunique()
            info.append(f"  - {col}: {dtype}, {unique_count} unique values, {null_count} null values")
        return "\n".join(info)

    def _serialize_table_to_html(self, df: pd.DataFrame, table_name: str) -> str:
        """
        Serialize a table to HTML format following the join_verification.py pattern.

        Args:
            df: DataFrame to serialize
            table_name: Name/title of the table

        Returns:
            HTML formatted string
        """
        display_name = self._get_display_name(table_name) if table_name else ""
        additional_knowledge = f"<title>\n{display_name}\n" if display_name else ""

        header = False if len(df.columns) == 1 and df.columns[0] == "" else True
        structured_data_html = df.to_html(header=header)

        grammar = "<HTML grammar>\n Each table cell is defined by a <td> and a </td> tag.\n Each table row starts with a <tr> and ends with a </tr> tag.\n th stands for table header.\n"
        return additional_knowledge + grammar + structured_data_html + "\n"

    def _serialize_table_to_autoprep(self, df: pd.DataFrame, table_name: str, cut_line: int = -1) -> str:
        """
        Serialize a table to autoprep format (col/row labeled style).

        Args:
            df: DataFrame to serialize
            table_name: Name/title of the table
            cut_line: Max rows to show (-1 for all)

        Returns:
            Autoprep formatted string
        """
        display_name = self._get_display_name(table_name) if table_name else ""
        ret = f"<title>\n{display_name}\n" if display_name else ""
        col_str = 'col : ' + ' | '.join(df.columns) + '\n'
        ret += col_str
        for i in range(len(df)):
            if cut_line != -1 and i > cut_line - 1:
                ret += '......\n'
                break
            row_str = 'row' + str(i + 1) + ' : ' + ' | '.join([str(x) for x in df.iloc[i].values]) + '\n'
            ret += row_str
        return ret.strip() + "\n"

    def _serialize_column_pair(self, col1: pd.Series, col2: pd.Series,
                               col1_name: str, col2_name: str,
                               table1_name: str, table2_name: str) -> str:
        """
        Serialize two columns for detailed comparison.
        Shows all rows without limit.

        Args:
            col1: First column
            col2: Second column
            col1_name: Name of first column
            col2_name: Name of second column
            table1_name: Name of first table
            table2_name: Name of second table

        Returns:
            Formatted string with both columns
        """
        display1 = self._get_display_name(table1_name)
        display2 = self._get_display_name(table2_name)
        result = f"### Column Pair for Detailed Analysis\n\n"
        result += f"**Table 1**: {display1}\n"
        result += f"**Column**: {col1_name}\n"
        max_rows = 200
        col1_values = col1.head(max_rows).tolist()
        col1_label = f"first {max_rows} of {len(col1)} rows" if len(col1) > max_rows else f"all {len(col1)} rows"
        result += f"**Values** ({col1_label}):\n"
        result += str(col1_values) + "\n\n"

        result += f"**Table 2**: {display2}\n"
        result += f"**Column**: {col2_name}\n"
        col2_values = col2.head(max_rows).tolist()
        col2_label = f"first {max_rows} of {len(col2)} rows" if len(col2) > max_rows else f"all {len(col2)} rows"
        result += f"**Values** ({col2_label}):\n"
        result += str(col2_values) + "\n\n"

        return result

    def phase1_verification(self, table1_path: str, table2_path: str,
                           candidate_pairs: List[Dict]) -> Dict[str, Any]:
        """
        Phase 1: Identify potential joinable column pairs using sample data (first 5 rows)

        Args:
            table1_path: Path to the first table
            table2_path: Path to the second table
            candidate_pairs: List of candidate column pairs from deep learning model
                            Each dictionary contains: {'column1', 'column2', 'overlap'}

        Returns:
            Dictionary containing:
                - 'selected_pairs': List of selected column pairs (format: "Table1.col1 <-> Table2.col2")
                - 'reason': LLM's reasoning process
                - 'tokens': Number of tokens used
                - 'input_tokens': Number of input tokens
                - 'output_tokens': Number of output tokens
                - 'latency': Time elapsed
        """
        if not self.prompt_template:
            raise ValueError("Phase1 verification requires prompt_template_path to be set")

        edge_key = self._get_edge_key(table1_path, table2_path)

        # Load first 5 rows
        try:
            df1 = pd.read_csv(table1_path, nrows=5)
            df2 = pd.read_csv(table2_path, nrows=5)
        except Exception as e:
            print(f"[Error] Unable to load tables: {e}")
            return {
                'selected_pairs': [],
                'reason': f"Unable to load tables: {e}",
                'tokens': 0,
                'input_tokens': 0,
                'output_tokens': 0,
                'latency': 0
            }

        # Serialize to autoprep format
        table1_serialized = self._serialize_table_to_autoprep(df1, table1_path)
        table2_serialized = self._serialize_table_to_autoprep(df2, table2_path)

        # Build candidate column pair list (don't show overlap to avoid bias)
        candidate_list = "### 【Candidate Column Pair Reference List】\n"
        for i, pair in enumerate(candidate_pairs, 1):
            col1 = pair['column1']
            col2 = pair['column2']
            candidate_list += f"{i}. `(Table1.{col1}, Table2.{col2})`\n"

        # Build prompt using template
        table1_display = self._get_display_name(table1_path)
        table2_display = self._get_display_name(table2_path)
        table_names = f"- Table 1: `{table1_display}`\n- Table 2: `{table2_display}`"

        prompt = self.prompt_template.replace("{TABLE_NAMES}", table_names)
        prompt = prompt.replace("{TABLE1_HTML}", "### 【Table 1】\n" + table1_serialized)
        prompt = prompt.replace("{TABLE2_HTML}", "### 【Table 2】\n" + table2_serialized)
        prompt = prompt.replace("{CANDIDATE_LIST}", candidate_list)

        # Initialize conversation
        system_prompt = "You are a rigorous data architect."
        self._init_conversation(edge_key, system_prompt)

        # Test mode: Only output prompt without calling LLM
        if self.test_mode:
            print("\n" + "="*80)
            print("【Test Mode】Phase 1 Prompt")
            print("="*80)
            print(f"\n[System Prompt]")
            print(system_prompt)
            print(f"\n[User Prompt]")
            print(prompt)
            print("="*80 + "\n")

            # Return mock result (select all candidate pairs for testing Phase2)
            mock_selected_pairs = []
            for pair in candidate_pairs:
                mock_selected_pairs.append(f"Table1.{pair['column1']} <-> Table2.{pair['column2']}")

            return {
                'selected_pairs': mock_selected_pairs,
                'reason': '[Test Mode] Mock selection of all candidate pairs',
                'tokens': 0,
                'input_tokens': 0,
                'output_tokens': 0,
                'latency': 0
            }

        # Call LLM
        result = self._call_llm_with_history(edge_key, prompt)
        self.stats['coarse_calls'] += 1  # Keep same statistics name for compatibility

        # Parse JSON response
        try:
            raw = result['response']
            # Try to strip markdown code blocks
            if '```' in raw:
                import re
                match = re.search(r'```(?:json)?\s*([\s\S]*?)```', raw)
                if match:
                    raw = match.group(1).strip()
            # Try to extract JSON object
            start_idx = raw.find('{')
            end_idx = raw.rfind('}') + 1
            if start_idx >= 0 and end_idx > start_idx:
                raw = raw[start_idx:end_idx]
            response_data = json.loads(raw)
            return {
                'selected_pairs': response_data.get('selected_pairs', []),
                'reason': response_data.get('reason', ''),
                'tokens': result['tokens'],
                'input_tokens': result['input_tokens'],
                'output_tokens': result['output_tokens'],
                'latency': result['latency']
            }
        except (json.JSONDecodeError, ValueError):
            # Fallback parsing
            print(f"[Warning] Unable to parse Phase1 JSON response, using fallback")
            return {
                'selected_pairs': [],
                'reason': result['response'],
                'tokens': result['tokens'],
                'input_tokens': result['input_tokens'],
                'output_tokens': result['output_tokens'],
                'latency': result['latency']
            }

    def phase2_verification(self, table1_path: str, table2_path: str,
                           column_pairs: List[Tuple[str, str]]) -> Dict[str, Any]:
        """
        Phase 2: Classify each column pair (joinable/non-joinable)

        Args:
            table1_path: Path to the first table
            table2_path: Path to the second table
            column_pairs: List of (col1_name, col2_name) tuples from phase1

        Returns:
            Dictionary containing:
                - 'valid_pairs': List of valid column pairs (can_join=True pairs)
                - 'reasoning': Dictionary of reasoning for each column pair
                - 'tokens': Number of tokens used
                - 'input_tokens': Number of input tokens
                - 'output_tokens': Number of output tokens
                - 'latency': Time elapsed
        """
        edge_key = self._get_edge_key(table1_path, table2_path)

        # If conversation not initialized (fallback when readable cache doesn't exist), auto-initialize
        if edge_key not in self.conversations:
            print(f"[Warning] Phase2 conversation not initialized, auto-initializing (conversation context will lack Phase1 history): {edge_key}")
            system_prompt = "You are a rigorous data architect."
            self._init_conversation(edge_key, system_prompt)

        # Load full table data
        try:
            df1 = pd.read_csv(table1_path)
            df2 = pd.read_csv(table2_path)
        except Exception as e:
            print(f"[Error] Unable to load tables: {e}")
            return {
                'valid_pairs': [],
                'reasoning': {},
                'tokens': 0,
                'input_tokens': 0,
                'output_tokens': 0,
                'latency': 0
            }

        valid_pairs = []
        reasoning = {}
        total_tokens = 0
        total_input_tokens = 0
        total_output_tokens = 0
        total_latency = 0.0

        for i, (col1, col2) in enumerate(column_pairs):
            # Check if columns exist
            if col1 not in df1.columns or col2 not in df2.columns:
                print(f"[Warning] Column does not exist: {col1} or {col2}")
                reasoning[f"{col1}<->{col2}"] = f"Column does not exist in table"
                continue

            # Serialize column pair
            column_data = self._serialize_column_pair(
                df1[col1], df2[col2], col1, col2, table1_path, table2_path
            )

            # Build prompt
            if i == 0:
                # First call: Include format instructions
                prompt = f"""Analyze the column pair below. Return a JSON object:
{{
  "data_quality": "Is each column suitable as a join key? A valid join key should identify or distinguish entities (e.g., names, codes, IDs, dates). Columns dominated by nan/empty/blank values, or columns whose non-empty values are mostly repetitive simple flags/states (e.g., Yes/No, 0/1), are NOT valid join keys — answer invalid. Exception: pairs structurally identified in Step 1 are always valid. Answer valid or invalid.",
  "same_attribute": "Can these two columns be semantically connected? Consider both column names AND values. Names differing only by numbered suffixes (e.g., score_3 vs score_1) are different instances of the same category — answer no. Column names pointing to different attributes or stages of the same entity (e.g., start_date vs end_date, origin vs destination) represent different business semantics even if their data types are identical — answer no. But if one column's values contain composite information that embeds the other column's data, they are semantically connected — answer yes. Exception: pairs structurally identified in Step 1 are always yes. Answer yes or no.",
  "alignment_method": "If joinable, show 2-3 example value pairs from the actual data that demonstrate a real correspondence. Each pair must be verifiable from the provided data — do not fabricate or assume mappings. Otherwise N/A.",
  "can_join": "true only if data_quality=valid AND same_attribute=yes AND alignment_method exists.",
  "reason": "Brief explanation."
}}

---

{column_data}"""
            else:
                # Subsequent calls: Only provide column data with brief rule reminder
                prompt = f"""Analyze the next column pair. Remember: columns dominated by nan/empty or repetitive flags (Yes/No, 0/1) are not valid join keys. Composite values embedding the other column's data → semantically connected.

{column_data}"""

            # Test mode: Only output prompt without calling LLM
            if self.test_mode:
                print("\n" + "="*80)
                print(f"【Test Mode】Phase 2 Prompt - Column pair {i+1}/{len(column_pairs)}: {col1} <-> {col2}")
                print("="*80)
                print(prompt)
                print("="*80 + "\n")

                # Mock result: Assume all column pairs are joinable
                reasoning[f"{col1}<->{col2}"] = '[Test Mode] Mock joinable'
                valid_pairs.append({
                    'column1': col1,
                    'column2': col2,
                    'reason': '[Test Mode] Mock joinable'
                })
                continue

            # Call LLM
            result = self._call_llm_with_history(edge_key, prompt)
            self.stats['fine_calls'] += 1  # Keep same statistics name for compatibility

            total_tokens += result['tokens']
            total_input_tokens += result['input_tokens']
            total_output_tokens += result['output_tokens']
            total_latency += result['latency']

            # Parse response
            try:
                start_idx = result['response'].find('{')
                end_idx = result['response'].rfind('}') + 1
                if start_idx >= 0 and end_idx > start_idx:
                    json_str = result['response'][start_idx:end_idx]
                    response_data = json.loads(json_str)

                    can_join = response_data.get('can_join', False)
                    reason = response_data.get('reason', '')

                    reasoning[f"{col1}<->{col2}"] = reason

                    if can_join:
                        valid_pairs.append({
                            'column1': col1,
                            'column2': col2,
                            'reason': reason
                        })
                else:
                    print(f"[Warning] JSON not found in Phase2 response: {col1}<->{col2}")
                    reasoning[f"{col1}<->{col2}"] = result['response']
            except (json.JSONDecodeError, ValueError) as e:
                print(f"[Warning] Unable to parse Phase2 response {col1}<->{col2}: {e}")
                reasoning[f"{col1}<->{col2}"] = result['response']

        return {
            'valid_pairs': valid_pairs,
            'reasoning': reasoning,
            'tokens': total_tokens,
            'input_tokens': total_input_tokens,
            'output_tokens': total_output_tokens,
            'latency': total_latency
        }

    def get_conversation_history(self, table1_path: str, table2_path: str) -> List[Dict]:
        """
        Get conversation history for specified edge

        Args:
            table1_path: Path to the first table
            table2_path: Path to the second table

        Returns:
            List of conversation history messages
        """
        edge_key = self._get_edge_key(table1_path, table2_path)
        return self.conversations.get(edge_key, [])

    def restore_conversation(self, table1_path: str, table2_path: str, conversation_history: list):
        """
        Restore conversation context from cached conversation history

        Args:
            table1_path: Path to the first table
            table2_path: Path to the second table
            conversation_history: List of conversation history messages
        """
        edge_key = self._get_edge_key(table1_path, table2_path)
        self.conversations[edge_key] = conversation_history

    def clear_conversation(self, table1_path: str, table2_path: str):
        """
        Clear conversation history for specified edge

        Args:
            table1_path: Path to the first table
            table2_path: Path to the second table
        """
        edge_key = self._get_edge_key(table1_path, table2_path)
        if edge_key in self.conversations:
            del self.conversations[edge_key]

    def get_statistics(self) -> Dict[str, Any]:
        """Get verification statistics"""
        avg_latency = self.stats['total_latency'] / self.stats['total_calls'] if self.stats['total_calls'] > 0 else 0
        return {
            'total_calls': self.stats['total_calls'],
            'coarse_calls': self.stats['coarse_calls'],
            'fine_calls': self.stats['fine_calls'],
            'total_tokens': self.stats['total_tokens'],
            'input_tokens': self.stats['input_tokens'],
            'output_tokens': self.stats['output_tokens'],
            'total_latency': self.stats['total_latency'],
            'average_latency': avg_latency,
            'active_conversations': len(self.conversations)
        }

    def print_statistics(self):
        """Print verification statistics"""
        stats = self.get_statistics()
        print("\n" + "=" * 60)
        print("LLM Verification Statistics")
        print("=" * 60)
        print(f"Total API calls:       {stats['total_calls']}")
        print(f"  - Coarse verify:     {stats['coarse_calls']}")
        print(f"  - Fine verify:       {stats['fine_calls']}")
        print(f"Total tokens:          {stats['total_tokens']:,}")
        print(f"  - Input tokens:      {stats['input_tokens']:,}")
        print(f"  - Output tokens:     {stats['output_tokens']:,}")
        print(f"Total latency:         {stats['total_latency']:.2f}s")
        print(f"Average latency:       {stats['average_latency']:.2f}s")
        print(f"Active conversations:  {stats['active_conversations']}")
        print("=" * 60)
