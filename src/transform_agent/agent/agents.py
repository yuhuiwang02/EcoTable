import base64
import json
import logging
import os
import re
import time
import uuid
import pandas as pd
from http import HTTPStatus
from io import BytesIO
from typing import Dict, List
from transform_agent.agent.prompts import BIGQUERY_SYSTEM, LOCAL_SYSTEM, DBT_SYSTEM,DBT_SYSTEM_2, SNOWFLAKE_SYSTEM, CH_SYSTEM, PG_SYSTEM,REFERENCE_PLAN_SYSTEM
from transform_agent.agent.action import Action, Bash, Terminate, CreateFile, EditFile, LOCAL_DB_SQL, BIGQUERY_EXEC_SQL, SNOWFLAKE_EXEC_SQL, BQ_GET_TABLES, BQ_GET_TABLE_INFO, BQ_SAMPLE_ROWS, SF_GET_TABLES, SF_GET_TABLE_INFO, SF_SAMPLE_ROWS
from transform_agent.envs.transform_agent import Transform_Agent_Env
from transform_agent.agent.models import call_llm
import duckdb
from transform_agent.agent.JoinClassifier import JoinClassify,Pair_join
from openai import AzureOpenAI
from typing import Dict, List, Optional, Tuple, Any, TypedDict
import json
from pathlib import Path
from transform_agent.agent.overlap import compare_overlap_duckdb
import shutil
import tiktoken
import re, ast
result_dir = "./result"
logger = logging.getLogger("transform_agent")
def append_jsonl(file_path, record: dict):
    Path(file_path).parent.mkdir(parents=True, exist_ok=True)
    with open(file_path, "a", encoding="utf-8") as f:
        json.dump(record, f, ensure_ascii=False)
        f.write("\n")

import duckdb, re

def _latest_version_name(db_path: str, base: str, schema: str = "main") -> str | None:
    """
    仅在指定 schema 中，返回 {base}_{num} 或 {base}-{num} 的最大 num 的表名
    """
    like_us = base.replace('\\','\\\\') + r"\_%"  # 匹配 base_%
    like_hy = base + "-%"                         # 匹配 base-%

    with duckdb.connect(db_path) as con:
        # 调试：打印候选
        candidates = con.execute("""
            SELECT table_schema, table_name
            FROM information_schema.tables
            WHERE table_schema = ?
              AND (table_name LIKE ? ESCAPE '\\' OR table_name LIKE ?)
            ORDER BY table_schema, table_name
        """, [schema, like_us, like_hy]).fetchall()
        print(f"[debug] schema={schema} 的候选：")
        for sch, tbl in candidates:
            print(f"  - {sch}.{tbl}")

        row = con.execute("""
            WITH c AS (
              SELECT table_schema, table_name
              FROM information_schema.tables
              WHERE table_schema = ?
                AND (table_name LIKE ? ESCAPE '\\' OR table_name LIKE ?)
            ),
            p AS (
              SELECT
                table_schema,
                table_name,
                COALESCE(
                  NULLIF(REGEXP_EXTRACT(table_name, '_(\\d+)$', 1), ''),
                  NULLIF(REGEXP_EXTRACT(table_name, '-(\\d+)$', 1), '')
                ) AS ver_str
              FROM c
            ),
            v AS (
              SELECT
                table_schema,
                table_name,
                TRY_CAST(ver_str AS INTEGER) AS ver
              FROM p
              WHERE ver_str IS NOT NULL
            )
            SELECT table_name
            FROM v
            WHERE ver IS NOT NULL
            ORDER BY ver DESC
            LIMIT 1
        """, [schema, like_us, like_hy]).fetchone()

    return row[0] if row else None

def has_table(duckdb_path: str, table_name: str) -> bool:
    """
    检查 DuckDB 文件中是否存在指定表。

    参数:
        duckdb_path (str): DuckDB 文件路径
        table_name (str): 要检测的表名（不区分大小写）

    返回:
        bool: 存在返回 True，不存在返回 False
    """
    db_file = Path(duckdb_path)
    if not db_file.is_file():
        raise FileNotFoundError(f"未找到 DuckDB 文件: {db_file}")

    con = duckdb.connect(str(db_file), read_only=True)
    try:
        # 获取所有表名（当前数据库）
        tables = con.execute("SHOW TABLES").fetchall()
        table_list = [t[0].lower() for t in tables]
        return table_name.lower() in table_list
    finally:
        con.close()

class PromptAgent:
    def __init__(
        self,
        model="gpt-4",
        max_tokens=1500,
        top_p=0.9,
        temperature=0.5,
        max_memory_length=10,
        max_steps=15,
        use_plan=False,
    ):
        self.duckdb=""
        self.model = model
        self.max_tokens = max_tokens
        self.top_p = top_p
        self.temperature = temperature
        self.max_memory_length = max_memory_length
        self.max_steps = max_steps
        
        self.thoughts = []
        self.responses = []
        self.actions = []
        self.observations = []
        self.system_message = ""
        self.history_messages = []
        self.env = None
        self.codes = []
        self.work_dir = "/workspace"
        self.use_plan = use_plan
        
    def set_env_and_task(self, env: Transform_Agent_Env):
        self.env = env
        self.thoughts = []
        self.responses = []
        self.actions = []
        self.observations = []
        self.codes = []
        self.history_messages = []
        self.instruction = []
        if 'plan' in self.env.task_config:
            self.reference_plan = self.env.task_config['plan']
        
        
        # if self.env.task_config['type'] == 'Bigquery':
        #     self._AVAILABLE_ACTION_CLASSES = [Bash, Terminate, BIGQUERY_EXEC_SQL, BQ_GET_TABLES, BQ_GET_TABLE_INFO, BQ_SAMPLE_ROWS, CreateFile, EditFile]
        #     action_space = "".join([action_cls.get_action_description() for action_cls in self._AVAILABLE_ACTION_CLASSES])
        #     self.system_message = BIGQUERY_SYSTEM.format(work_dir=self.work_dir, action_space=action_space, task=self.instruction, max_steps=self.max_steps)
        # elif self.env.task_config['type'] == 'Snowflake':
        #     self._AVAILABLE_ACTION_CLASSES = [Bash, Terminate, SNOWFLAKE_EXEC_SQL, SF_GET_TABLES, SF_GET_TABLE_INFO, SF_SAMPLE_ROWS, CreateFile, EditFile]
        #     action_space = "".join([action_cls.get_action_description() for action_cls in self._AVAILABLE_ACTION_CLASSES])
        #     self.system_message = SNOWFLAKE_SYSTEM.format(work_dir=self.work_dir, action_space=action_space, task=self.instruction, max_steps=self.max_steps)
        # elif self.env.task_config['type'] == 'Local':
        #     self._AVAILABLE_ACTION_CLASSES = [Bash, Terminate, CreateFile, EditFile, LOCAL_DB_SQL]
        #     action_space = "".join([action_cls.get_action_description() for action_cls in self._AVAILABLE_ACTION_CLASSES])
        #     self.system_message = LOCAL_SYSTEM.format(work_dir=self.work_dir, action_space=action_space, task=self.instruction, max_steps=self.max_steps)
        # elif self.env.task_config['type'] == 'DBT':
        #     self._AVAILABLE_ACTION_CLASSES = [Bash, Terminate, CreateFile, EditFile, LOCAL_DB_SQL]
        #     action_space = "".join([action_cls.get_action_description() for action_cls in self._AVAILABLE_ACTION_CLASSES])
        #     self.system_message = DBT_SYSTEM.format(work_dir=self.work_dir, action_space=action_space, max_steps=self.max_steps)
        # elif self.env.task_config['type'] == 'Postgres':
        #     self._AVAILABLE_ACTION_CLASSES = [Bash, Terminate, CreateFile, EditFile]
        #     action_space = "".join([action_cls.get_action_description() for action_cls in self._AVAILABLE_ACTION_CLASSES])
        #     self.system_message = PG_SYSTEM.format(work_dir=self.work_dir, action_space=action_space, task=self.instruction, max_steps=self.max_steps)
        # elif self.env.task_config['type'] == 'Clickhouse':
        #     self._AVAILABLE_ACTION_CLASSES = [Bash, Terminate, CreateFile, EditFile]
        #     action_space = "".join([action_cls.get_action_description() for action_cls in self._AVAILABLE_ACTION_CLASSES])
        #     self.system_message = CH_SYSTEM.format(work_dir=self.work_dir, action_space=action_space, task=self.instruction, max_steps=self.max_steps)
        
        if self.use_plan:
            self.system_message += REFERENCE_PLAN_SYSTEM.format(plan=self.reference_plan)
        


        self.history_messages.append({
            "role": "system",
            "content": [
                {
                    "type": "text",
                    "text": self.system_message 
                },
            ]
        })
    
    def parse_join_response(self, response_text: str) -> List[str]:
        """
        从模型输出文本中解析 JoinPair 段，并返回一个字符串列表。
        支持：
        - JoinPair: a-b
        - JoinPair: ["a-b", "c-d"]
        - 多行/代码块/大小写不同
        - 行内 # 注释（仅在非列表形式时移除）
        返回：按出现顺序的列表；会去掉引号与两侧空白，并规范化 "a - b" 为 "a-b"。
        """
        text = (response_text or "").strip()

        # 抓取 JoinPair 后面的内容，直到下一个字段(Thought|Action|JoinPair)或文本结束
        m = re.search(r'(?ims)^\s*JoinPair\s*:\s*(.+?)(?=\n\s*(?:Thought|Action|JoinPair)\s*:|\Z)', text)
        if not m:
            return []

        block = m.group(1).strip()
        pairs: List[str] = []

        if block.startswith("["):
            # 列表形式：尝试安全解析
            try:
                obj = ast.literal_eval(block)
                if isinstance(obj, (list, tuple)):
                    for item in obj:
                        s = item if isinstance(item, str) else str(item)
                        s = s.strip().strip('`"\'')

                        s = re.sub(r'\s*-\s*', '-', s)  # 规范化 a - b -> a-b
                        if s:
                            pairs.append(s)
            except Exception:
                # 回退：简单逗号切分
                inner = block.strip("[]")
                for p in inner.split(","):
                    s = p.strip().strip('`"\'')
                    s = re.sub(r'\s*-\s*', '-', s)
                    if s:
                        pairs.append(s)
        else:
            # 非列表形式：先去掉行内注释
            block = block.split("#", 1)[0].strip()
            if "," in block:
                items = [x for x in block.split(",")]
            else:
                items = [block] if block else []

            for p in items:
                s = p.strip().strip('`"\'')
                s = re.sub(r'\s*-\s*', '-', s)
                if s:
                    pairs.append(s)

        return pairs


    def predict(self, obs: Dict=None) -> Tuple[str, Optional[Action], Dict[str, Optional[str]]]:
        """
        Predict the next action(s) based on the current observation.
        """
        assert len(self.observations) == len(self.actions) and len(self.actions) == len(self.thoughts), \
            "The number of observations and actions should be the same."

        status = False
        response = ""
        while not status:
            messages = self.history_messages.copy()
            messages.append({
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": "Observation: {}\n".format(str(obs))
                    }
                ]
            })

            # === CHANGE START: 统一判定 4o + Azure 凭证，优先直连 Azure 获取真实 usage & latency ===
            # 你的 call_llm 里 Azure 使用的是 AZURE_API_KEY / AZURE_ENDPOINT
            azure_api_key = os.environ.get("AZURE_API_KEY") or os.environ.get("AZURE_OPENAI_API_KEY")
            azure_endpoint = os.environ.get("AZURE_ENDPOINT") or os.environ.get("AZURE_OPENAI_ENDPOINT")
            azure_api_version = os.environ.get("AZURE_OPENAI_API_VERSION", "2024-02-15-preview")
            deployment_env = os.environ.get("AZURE_OPENAI_DEPLOYMENT", "")

            model_lower = str(self.model).lower()
            is_4o_name = model_lower in {"gpt-4o", "4o", "gpt-4o-mini", "4o-mini"} or \
                        deployment_env.lower().startswith(("gpt-4o", "4o"))
            azure_creds_ok = bool(azure_api_key and azure_endpoint)
            use_azure_4o = is_4o_name and azure_creds_ok
            # === CHANGE END ===

            if use_azure_4o:
                # 将多模态 messages 压平为 Chat Completions 需要的文本 messages（仅保留 text）
                def _to_openai_messages(msgs):
                    out = []
                    for m in msgs:
                        role = m.get("role", "user")
                        content = m.get("content", "")
                        if isinstance(content, list):
                            texts = []
                            for part in content:
                                if isinstance(part, dict) and part.get("type") == "text":
                                    texts.append(part.get("text", ""))
                            content = "\n".join(texts)
                        elif not isinstance(content, str):
                            content = str(content)
                        out.append({"role": role, "content": content})
                    return out

                openai_messages = _to_openai_messages(messages)
                stop = ["Observation:", "\n\n\n\n", "\n \n \n"]
                t0 = time.perf_counter()
                try:
                    if not hasattr(self, "_azure_client") or self._azure_client is None:
                        self._azure_client = AzureOpenAI(
                            api_key=azure_api_key,
                            api_version=azure_api_version,
                            azure_endpoint=azure_endpoint,
                        )
                    deployment = deployment_env or self.model  # 支持直接用模型名
                    resp = self._azure_client.chat.completions.create(
                        model=deployment,
                        messages=openai_messages,
                        temperature=self.temperature,
                        top_p=self.top_p,
                        max_tokens=self.max_tokens,
                        stop=stop,
                    )
                    latency_ms = round((time.perf_counter() - t0) * 1000.0, 2)
                    response = (resp.choices[0].message.content or "").strip()

                    # 真实 usage
                    usage_raw = getattr(resp, "usage", None) or {}
                    is_dict = isinstance(usage_raw, dict)
                    prompt_tokens = getattr(usage_raw, "prompt_tokens", None) or (usage_raw.get("prompt_tokens") if is_dict else None)
                    completion_tokens = getattr(usage_raw, "completion_tokens", None) or (usage_raw.get("completion_tokens") if is_dict else None)
                    total_tokens = getattr(usage_raw, "total_tokens", None) or (usage_raw.get("total_tokens") if is_dict else None)
                    input_tokens = getattr(usage_raw, "input_tokens", None) or (usage_raw.get("input_tokens") if is_dict else None) or prompt_tokens
                    output_tokens = getattr(usage_raw, "output_tokens", None) or (usage_raw.get("output_tokens") if is_dict else None) or completion_tokens
                    if total_tokens is None:
                        total_tokens = (input_tokens or 0) + (output_tokens or 0)

                    finish_reason = getattr(resp.choices[0], "finish_reason", None)
                    logger.info("[LLM:4o/Azure] input_tokens=%s, output_tokens=%s, total_tokens=%s, latency_ms=%.2f, finish_reason=%s",
                                input_tokens, output_tokens, total_tokens, latency_ms, finish_reason)
                    status = True

                except Exception as e:
                    latency_ms = round((time.perf_counter() - t0) * 1000.0, 2)
                    logger.exception("[LLM:4o/Azure] 调用失败 (latency_ms=%.2f): %s —— 将回退到 call_llm()", latency_ms, e)
                    # 回退到 call_llm
                    t1 = time.perf_counter()
                    status, response = call_llm({
                        "model": self.model,
                        "messages": messages,
                        "max_tokens": self.max_tokens,
                        "top_p": self.top_p,
                        "temperature": self.temperature
                    })
                    response = (response or "").strip()
                    latency_ms = round((time.perf_counter() - t1) * 1000.0, 2)
                    # 用 tiktoken 估算 tokens
                    try:
                        enc = tiktoken.get_encoding("cl100k_base")
                    except Exception:
                        enc = tiktoken.encoding_for_model("gpt-3.5-turbo")
                    def _flatten_text(msgs):
                        txts = []
                        for m in msgs:
                            c = m.get("content", "")
                            if isinstance(c, list):
                                for p in c:
                                    if isinstance(p, dict) and p.get("type") == "text":
                                        txts.append(p.get("text", ""))
                            elif isinstance(c, str):
                                txts.append(c)
                        return "\n".join(txts)
                    input_txt = _flatten_text(messages)
                    input_tokens_est = len(enc.encode(input_txt)) if input_txt else 0
                    output_tokens_est = len(enc.encode(response)) if response else 0
                    total_est = input_tokens_est + output_tokens_est
                    logger.info("[LLM:fallback-estimate] input_tokens≈%s, output_tokens≈%s, total_tokens≈%s, latency_ms=%.2f",
                                input_tokens_est, output_tokens_est, total_est, latency_ms)

            else:
                # 统一走 call_llm；计时 + 估算 tokens
                t0 = time.perf_counter()
                status, response = call_llm({
                    "model": self.model,
                    "messages": messages,
                    "max_tokens": self.max_tokens,
                    "top_p": self.top_p,
                    "temperature": self.temperature
                })
                response = (response or "").strip()
                latency_ms = round((time.perf_counter() - t0) * 1000.0, 2)

                # === CHANGE START: 使用 tiktoken 进行近似估算，便于统一日志观察 ===
                try:
                    enc = tiktoken.get_encoding("cl100k_base")
                except Exception:
                    enc = tiktoken.encoding_for_model("gpt-3.5-turbo")

                def _flatten_text(msgs):
                    txts = []
                    for m in msgs:
                        c = m.get("content", "")
                        if isinstance(c, list):
                            for p in c:
                                if isinstance(p, dict) and p.get("type") == "text":
                                    txts.append(p.get("text", ""))
                        elif isinstance(c, str):
                            txts.append(c)
                    return "\n".join(txts)

                input_txt = _flatten_text(messages)
                input_tokens_est = len(enc.encode(input_txt)) if input_txt else 0
                output_tokens_est = len(enc.encode(response)) if response else 0
                total_est = input_tokens_est + output_tokens_est

                logger.info("[LLM:estimate] model=%s, input_tokens≈%s, output_tokens≈%s, total_tokens≈%s, latency_ms=%.2f",
                            self.model, input_tokens_est, output_tokens_est, total_est, latency_ms)
                # === CHANGE END ===

            # 统一错误分支（两条路径都适用）
            if not status:
                if response in ["context_length_exceeded", "rate_limit_exceeded", "max_tokens", "unknown_error"]:
                    self.history_messages = [self.history_messages[0]] + self.history_messages[3:]
                    # while 会继续重试
                else:
                    raise Exception(f"Failed to call LLM, response: {response}")

        # 解析与轨迹记录（与你现有逻辑一致）
        try:
            action = self.parse_action(response)
            join_relation = self.parse_join_response(response)
            thought = re.search(r'Thought:(.*?)Action', response, flags=re.DOTALL)
            thought = thought.group(1).strip() if thought else response
        except ValueError as e:
            print("Failed to parse action from response", e)
            action = None
            validjoin = False
            join_relation = None
            thought = response

        logger.info("Observation: %s", obs)
        logger.info("Response: %s", response)
        self._add_message(obs, thought, action)
        self.observations.append(obs)
        self.thoughts.append(thought)
        self.responses.append(response)
        self.actions.append(action)

        return response, action, join_relation
        
    
    def _add_message(self, observations: str, thought: str, action: Action):
        self.history_messages.append({
            "role": "user",
            "content": [
                {
                    "type": "text",
                    "text": "Observation: {}".format(observations)
                }
            ]
        })
        self.history_messages.append({
            "role": "assistant",
            "content": [
                {
                    "type": "text",
                    "text": "Thought: {}\n\nAction: {}".format(thought, str(action))
                }
            ]
        })
        if len(self.history_messages) > self.max_memory_length*2+1:
            self.history_messages = [self.history_messages[0]] + self.history_messages[-self.max_memory_length*2:]
    
    def parse_action(self, output: str) -> Action:
        """ Parse action from text """
        if output is None or len(output) == 0:
            pass
        action_string = ""
        patterns = [r'["\']?Action["\']?:? (.*?)Observation',r'["\']?Action["\']?:? (.*?)Thought', r'["\']?Action["\']?:? (.*?)$', r'^(.*?)Observation']

        for p in patterns:
            match = re.search(p, output, flags=re.DOTALL)
            if match:
                action_string = match.group(1).strip()
                break
        if action_string == "":
            action_string = output.strip()
        
        output_action = None
        for action_cls in self._AVAILABLE_ACTION_CLASSES:
            action = action_cls.parse_action_from_text(action_string)
            if action is not None:
                output_action = action
                break
        if output_action is None:
            action_string = action_string.replace("\_", "_").replace("'''","```")
            for action_cls in self._AVAILABLE_ACTION_CLASSES:
                action = action_cls.parse_action_from_text(action_string)
                if action is not None:
                    output_action = action
                    break
        
        return output_action
    
    def init_for_each(self,join_pairs:str,t1:str,t2:str,t1n:str,t2n:str,query:str):
        self.thoughts = []
        self.responses = []
        self.actions = []
        self.observations = []
        self.codes = []
        self.history_messages = []
        self._AVAILABLE_ACTION_CLASSES = [Bash, Terminate, CreateFile, EditFile, LOCAL_DB_SQL]
        action_space = "".join([action_cls.get_action_description() for action_cls in self._AVAILABLE_ACTION_CLASSES])
        obs = f"Now I need you to handle this"
        self.system_message = DBT_SYSTEM.format(work_dir=self.work_dir,table1=t1,table2=t2,action_space=action_space,join_pair=join_pairs, max_steps=self.max_steps,table1new=t1n,table2new=t2n,duckdb=self.duckdb,query=query)
        self.history_messages.append({
            "role": "system",
            "content": [
                {
                    "type": "text",
                    "text": self.system_message 
                },
            ]
        })
        return obs
    
    def run(self,input_dir,output_dir,project,idx,projectname):
        assert self.env is not None, "Environment is not set."
        result = ""
        done = False
        step_idx = 0
        retry_count = 0
        last_action = None
        repeat_action = False
        join_relations = []
        files = os.listdir(input_dir)
        for file in files:
            if file.endswith('.duckdb'):  # 检查文件是否以 .duckdb 结尾
                self.duckdb=file  # 保存文件的完整路径
                duckdbnow=os.path.join(output_dir,file)
                break

        query=project.get("query")
        edges = project.get("edges", {})
        for key, pair_list in edges.items():
            edge_strings = []
            table1, table2 = key.split("-", 1)
            table1new,table2new=f"{table1}",f"{table2}"
            for pair in pair_list:
                if isinstance(pair, (list, tuple)) and len(pair) == 2:
                    p1, p2 = pair
                    overlap=compare_overlap_duckdb(duckdbnow,table1,table2,p1,p2)
                    edge_strings.append(f"{p1}-{p2}: overlap:{overlap}")
                    
            edge_text = "\n".join(edge_strings)
            print(input_dir)
            print(edge_text)
            obs = self.init_for_each(edge_text,table1,table2,table1new,table2new,query)
            done=False
            join_record=[]
            while not done and step_idx < self.max_steps:
                _, action, join_relation= self.predict(
                    obs
                )
                if join_relation:
                    join_record=join_relation
                if action is None:
                    logger.info("Failed to parse action from response, try again.")
                    retry_count += 1
                    if retry_count > 3:
                        logger.info("Failed to parse action from response, stop.")
                        break
                    obs = "Failed to parse action from your response, make sure you provide a valid action."
                else:
                    logger.info("Step %d: %s", step_idx + 1, action)
                    # obs, done = self.env.step(action)
                    if last_action is not None and last_action == action:
                        if repeat_action:
                            return False, "ERROR: Repeated action"
                        else:
                            obs = "The action is the same as the last one, you MUST provide a DIFFERENT SQL code or Python Code or different action. you MUST provide a DIFFERENT SQL code or Python Code or different action. you MUST provide a DIFFERENT SQL code or Python Code or different action."
                            repeat_action = True
                    else:
                        obs, done = self.env.step(action)
                        last_action = action
                        repeat_action = False

                if done and isinstance(action, Terminate):
                        if join_record:
                            # 计算以 table1 / table2 为前缀、形如 {name}-{num} 的最新表名
                            t1_latest = _latest_version_name(duckdbnow, table1)
                            t2_latest = _latest_version_name(duckdbnow, table2)

                            # 若找到了，就用它们覆盖当前的 table1new / table2new；否则保留原值
                            if t1_latest:
                                table1new = t1_latest
                            if t2_latest:
                                table2new = t2_latest
                            join_relations.append({
                                "tables": [table1new, table2new],
                                "join_record": join_record
                            })
                        print(join_record)
                        result = action.output
                        logger.info("The task is done.")
                step_idx += 1
        if join_relations:
            base_name = f"{projectname}-{idx}"
            json_filename = f"{base_name}.json"
            project_sql=f"{result_dir}/transformation/{projectname}-{idx}"
            json_filepath = os.path.join(project_sql, json_filename)
            os.makedirs(project_sql,exist_ok=True)
            with open(json_filepath, 'w') as json_file:
                json.dump(join_relations, json_file, indent=4)

            # 拷贝 duckdb，命名与 json 同名（仅后缀不同）
            if duckdbnow and os.path.exists(duckdbnow):
                duckdb_dst = os.path.join(project_sql, f"{base_name}.duckdb")
                try:
                    shutil.copy2(duckdbnow, duckdb_dst)
                    logger.info(f"Copied duckdb to {duckdb_dst}")
                except Exception as e:
                    logger.warning(f"Failed to copy duckdb file from {duckdbnow} to {duckdb_dst}: {e}")
            else:
                logger.warning(f"DuckDB source file not found, skip copy. duckdbnow={duckdbnow}")
        return done, result

    def get_trajectory(self):
        trajectory = []
        for i in range(len(self.observations)):
            trajectory.append({
                "observation": self.observations[i],
                "thought": self.thoughts[i],
                "action": str(self.actions[i]),
                # "code": self.codes[i],
                "response": self.responses[i]
            })
        trajectory_log = {
            "Task": self.instruction,
            "system_message": self.system_message,
            "trajectory": trajectory
        }
        return trajectory_log


if __name__ == "__main__":
    agent = PromptAgent()
    response = """
BIGQUERY_EXEC_SQL(sql_query=\"\"\"
WITH purchase_users AS (
  SELECT DISTINCT user_pseudo_id
  FROM `bigquery-public-data.ga4_obfuscated_sample_ecommerce.events_*`
  WHERE event_name = 'purchase' AND _TABLE_SUFFIX BETWEEN '20201201' AND '20201231'
),
pageviews AS (
  SELECT user_pseudo_id, COUNT(*) AS pageviews
  FROM `bigquery-public-data.ga4_obfuscated_sample_ecommerce.events_*`
  WHERE event_name = 'page_view' AND _TABLE_SUFFIX BETWEEN '20201201' AND '20201231'
  GROUP BY user_pseudo_id
),
pageviews_by_user AS (
  SELECT 
    p.user_pseudo_id, 
    p.pageviews,
    CASE WHEN pu.user_pseudo_id IS NOT NULL THEN 'purchaser' ELSE 'non-purchaser' END AS user_type
  FROM pageviews p
  LEFT JOIN purchase_users pu ON p.user_pseudo_id = pu.user_pseudo_id
)
SELECT user_type, AVG(pageviews) AS avg_pageviews
FROM pageviews_by_user
GROUP BY user_type
\"\"\", is_save=True, save_path="avg_pageviews_dec_2020.csv")
"""

    response = """
BIGQUERY_EXEC_SQL(sql_query=\"\"\"
SELECT DISTINCT user_pseudo_id
FROM bigquery-public-data.ga4_obfuscated_sample_ecommerce.events_*
WHERE event_name = 'purchase' AND _TABLE_SUFFIX BETWEEN '20201201' AND '20201231'
\"\"\", is_save=False)
"""


    action = agent.parse_action(response)
    
    print(action)