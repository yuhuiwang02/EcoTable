import argparse
import datetime
import json
import logging
import os
import random
import sys
import glob
from typing import Optional, Dict, Any
from tqdm import tqdm

from transform_agent.envs.transform_agent import Transform_Agent_Env
from transform_agent.agent.agents import PromptAgent
from transform_agent.agent.dataset_process import process_join_graph
import urllib3
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)


#  Logger Configs {{{ #
logger = logging.getLogger("transform_agent")
logger.setLevel(logging.DEBUG)

datetime_str: str = datetime.datetime.now().strftime("%Y%m%d@%H%M%S")

file_handler = logging.FileHandler(os.path.join("logs", "normal-{:}.log".format(datetime_str)), encoding="utf-8")
debug_handler = logging.FileHandler(os.path.join("logs", "debug-{:}.log".format(datetime_str)), encoding="utf-8")
stdout_handler = logging.StreamHandler(sys.stdout)
sdebug_handler = logging.FileHandler(os.path.join("logs", "sdebug-{:}.log".format(datetime_str)), encoding="utf-8")

file_handler.setLevel(logging.INFO)
debug_handler.setLevel(logging.DEBUG)
stdout_handler.setLevel(logging.INFO)
sdebug_handler.setLevel(logging.DEBUG)

formatter = logging.Formatter(
    fmt="\x1b[1;33m[%(asctime)s \x1b[31m%(levelname)s \x1b[32m%(module)s/%(lineno)d-%(processName)s\x1b[1;33m] \x1b[0m%(message)s")
file_handler.setFormatter(formatter)
debug_handler.setFormatter(formatter)
stdout_handler.setFormatter(formatter)
sdebug_handler.setFormatter(formatter)

stdout_handler.addFilter(logging.Filter("transform_agent"))
sdebug_handler.addFilter(logging.Filter("transform_agent"))

logger.addHandler(file_handler)
logger.addHandler(debug_handler)
logger.addHandler(stdout_handler)
logger.addHandler(sdebug_handler)
#  }}} Logger Configs # 



def config() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run end-to-end evaluation on the benchmark"
    )
    
    parser.add_argument("--max_steps", type=int, default=100)
    
    parser.add_argument("--max_memory_length", type=int, default=25)
    parser.add_argument("--suffix", '-s', type=str, default="gpt-4-try1")
    
    parser.add_argument("--model", type=str, default="gpt-4o")
    parser.add_argument("--temperature", type=float, default=0)
    parser.add_argument("--top_p", type=float, default=0.9)
    parser.add_argument("--max_tokens", type=int, default=2500)
    parser.add_argument("--stop_token", type=str, default=None)
    parser.add_argument("--project", type=str, default="snapchat_ads")
    # example config
    parser.add_argument("--input_dir", type=str, default=None)
    parser.add_argument("--example_index", "-i", type=str, default="all", help="index range of the examples to run, e.g., '0-10', '2,3', 'all'")
    parser.add_argument("--example_name", "-n", type=str, default="", help="name of the example to run")
    parser.add_argument("--overwriting", action="store_true", default=False)
    parser.add_argument("--retry_failed", action="store_true", default=False)

    # output related
    parser.add_argument("--output_dir", type=str, default="output")
    parser.add_argument("--plan", action="store_true")

    parser.add_argument("--dbt_only", action="store_true",default=True)
    
    args = parser.parse_args()

    return args



def test(
    args: argparse.Namespace,
    test_all_meta: dict = None
) -> None:
    scores = []
    
    # log args
    logger.info("Args: %s", args)

    if args.suffix == "":
        logger.warning("No suffix is provided, the experiment id will be the model name.")
        experiment_id = args.model.split("/")[-1]
    else:
        experiment_id = args.model.split("/")[-1] + "-" + args.suffix
        
    if args.plan:
        experiment_id = f"{experiment_id}-plan"

    projectname=args.project
    agent = PromptAgent(
        model=args.model,
        max_tokens=args.max_tokens,
        top_p=args.top_p,
        temperature=args.temperature,
        max_memory_length=args.max_memory_length,
        max_steps=args.max_steps,
        use_plan=args.plan,
    )
    valid_ids = []
    ## load task configs
    output_path = experiment_id +"/"+ projectname
    input_path =f"{args.input_dir}/{projectname}"
    output_path = os.path.join(args.output_dir, output_path)
    # process_join_graph(input_path,input_path)
    input_json=f"{input_path}/{projectname}_edges.json"
    with open(input_json,"r") as f:
        projects = json.load(open(input_json, "r", encoding="utf-8"))
    idx=0
    for project in projects:
        output_path = experiment_id +"/"+ f"{projectname}-{idx}"
        output_path = os.path.join(args.output_dir, output_path)
        task_config={}
        task_config["instance_id"]=f"{projectname}"
        task_config["instruction"]=project.get("query")
        result_json_path =os.path.join(output_path, f"result/result-{idx}.json")
        instance_id=f"{projectname}-{idx}"
        os.makedirs(f"{output_path}/{instance_id}",exist_ok=True)
        valid_types = set()

        if args.dbt_only: valid_types.add('dbt')



        valid_ids.append(task_config["instance_id"])
        if not args.overwriting and os.path.exists(result_json_path):
            logger.info("Skipping %s", instance_id)
            continue
        elif os.path.exists(result_json_path):
            logger.info("Overwriting %s", instance_id)
        else:
            logger.info("Running %s", instance_id)
        if args.retry_failed and os.path.exists(result_json_path):
            with open(result_json_path, "r") as f:
                result = json.load(f)
                if result["finished"] and (not "FAIL" in result["result"]) and (not "error" in result["result"].lower()):
                    logger.info("Skipping %s", instance_id)
                    continue
            logger.info("Retrying %s", instance_id)
            
        if os.path.exists(output_path):
            os.system(f"rm -rf {output_path}")
            logger.info("Removed existing %s", output_path)

        os.makedirs(output_path, exist_ok=True)


        source_data_dir = args.input_dir

        env_config = \
        {
            "init_args": {
                "name": experiment_id,
                "work_dir": "/workspace",
            }
        }

        env_config["image_name"] = "transform_agent-image"
        task_config['config'] = [{"type": "copy_all_subfiles", "parameters": {"dirs": [os.path.join(source_data_dir, task_config["instance_id"])]}}]


        env_config["init_args"]["name"] = experiment_id +"-"+ task_config["instance_id"]

          


        env = Transform_Agent_Env(
            env_config=env_config,
            task_config=task_config,
            cache_dir="./cache",
            mnt_dir=output_path
        )
    
        agent.set_env_and_task(env)
    
        logger.info('Task input:' + projectname)
        done, result_output = agent.run(input_path,output_path,project,idx,projectname)
        trajectory = agent.get_trajectory()

        os.makedirs(os.path.join(output_path, "result"), exist_ok=True)
        result_files = env.post_process()
        transform_result = {"finished": done, "steps": len(trajectory["trajectory"]),
                           "result": result_output,"result_files": result_files, **trajectory}
        with open(os.path.join(output_path, "transform/result.json"), "w") as f:
            json.dump(transform_result, f, indent=2)
        

        logger.info("Finished %s", projectname)
        env.close()
        idx+=1




if __name__ == '__main__':
    args = config()
    test(args)

def run_transform_agent_for_project(
    *,
    project: str,
    input_dir: str,          # 形如：<root>/<project> 下放入 *_edges.json 或可由 process_join_graph 生成
    output_dir: str = "output",
    # 推理相关（保持原默认）
    model: str = "gpt-4o",
    temperature: float = 0.0,
    top_p: float = 0.9,
    max_tokens: int = 2500,
    max_steps: int = 100,
    max_memory_length: int = 25,
    stop_token: Optional[str] = None,
    plan: bool = False,
    overwriting: bool = False,
    retry_failed: bool = False,
    dbt_only: bool = True,
) -> None:
    """
    把原 test(main) 改成可复用接口。保证：
      - project 等于调用者指定的项目名
      - suffix 强制为 project
      - 不改变任何输出 JSON 的结构或命名
    """
    # ====== 构造 experiment_id：模型名后缀 + '-' + project；plan 时附加 '-plan' ======
    model_tail = model.split("/")[-1]
    suffix = project
    experiment_id = f"{model_tail}-{suffix}"
    if plan:
        experiment_id = f"{experiment_id}-plan"

    # ====== 创建 Agent ======
    agent = PromptAgent(
        model=model,
        max_tokens=max_tokens,
        top_p=top_p,
        temperature=temperature,
        max_memory_length=max_memory_length,
        max_steps=max_steps,
        use_plan=plan,
    )

    # ====== 输入输出路径（与原脚本一致的目录组织）======
    projectname = project
    output_root_for_project = os.path.join(output_dir, experiment_id, projectname)
    in_proj_dir = os.path.join(input_dir, projectname)

    os.makedirs(output_root_for_project, exist_ok=True)
    os.makedirs(in_proj_dir, exist_ok=True)

    # 如无 edges 文件，尝试从组合结果转换（与原脚本注释一致，这里显式执行一次）
    # 会在 in_proj_dir 下生成 <project>_edges.json
    try:
        process_join_graph(in_proj_dir, in_proj_dir)
    except Exception as e:
        logger.debug(f"[{projectname}] process_join_graph 跳过或已存在: {e}")

    input_json = os.path.join(in_proj_dir, f"{projectname}_edges.json")
    with open(input_json, "r", encoding="utf-8") as f:
        projects = json.load(f)

    # ====== 逐条任务跑（与原 test() 一致）======
    for idx, one_proj in enumerate(projects):
        # 针对每个样本，单独的输出目录：<output>/<experiment_id>/<projectname-idx>
        exp_dir = os.path.join(output_dir, experiment_id, f"{projectname}-{idx}")
        instance_id = f"{projectname}-{idx}"
        result_json_path = os.path.join(exp_dir, "result", f"result-{idx}.json")

        # 覆盖/重试控制（与原一致）
        if not overwriting and os.path.exists(result_json_path):
            logger.info("Skipping %s", instance_id)
            continue
        elif os.path.exists(result_json_path) and overwriting:
            logger.info("Overwriting %s", instance_id)
        else:
            logger.info("Running %s", instance_id)
        if retry_failed and os.path.exists(result_json_path):
            with open(result_json_path, "r", encoding="utf-8") as f:
                result = json.load(f)
                if result.get("finished") and ("FAIL" not in result.get("result", "")) and ("error" not in result.get("result","").lower()):
                    logger.info("Skipping %s", instance_id)
                    continue
            logger.info("Retrying %s", instance_id)

        # 清理并重建输出目录
        if os.path.exists(exp_dir):
            os.system(f"rm -rf {exp_dir}")
            logger.info("Removed existing %s", exp_dir)
        os.makedirs(exp_dir, exist_ok=True)
        os.makedirs(os.path.join(exp_dir, instance_id), exist_ok=True)

        # ====== 环境配置（保持与原逻辑一致）======
        source_data_dir = input_dir
        env_config = {
            "init_args": {
                "name": f"{experiment_id}-{projectname}",
                "work_dir": "/workspace",
            },
            "image_name": "transform_agent-image",
        }
        task_config: Dict[str, Any] = {
            "instance_id": projectname,
            "instruction": one_proj.get("query"),
            "config": [{
                "type": "copy_all_subfiles",
                "parameters": {"dirs": [os.path.join(source_data_dir, projectname)]}
            }],
        }
        valid_types = set()
        if dbt_only:
            valid_types.add("dbt")

        env = Transform_Agent_Env(
            env_config=env_config,
            task_config=task_config,
            cache_dir="./cache",
            mnt_dir=exp_dir
        )
        agent.set_env_and_task(env)

        logger.info('Task input: ' + projectname)
        done, result_output = agent.run(in_proj_dir, exp_dir, one_proj, idx, projectname)
        trajectory = agent.get_trajectory()

        os.makedirs(os.path.join(exp_dir, "result"), exist_ok=True)
        result_files = env.post_process()
        transform_result = {
            "finished": done,
            "steps": len(trajectory["trajectory"]),
            "result": result_output,
            "result_files": result_files,
            **trajectory,
        }
        os.makedirs(os.path.join(exp_dir, "transform"), exist_ok=True)
        with open(os.path.join(exp_dir, "transform", "result.json"), "w", encoding="utf-8") as f:
            json.dump(transform_result, f, indent=2, ensure_ascii=False)

        logger.info("Finished %s #%d", projectname, idx)
        env.close()

def run_transform_agent_for_projects(
    *,
    projects: list,
    input_dir: str,
    output_dir: str = "output",
    model: str = "gpt-4o",
    temperature: float = 0.0,
    top_p: float = 0.9,
    max_tokens: int = 2500,
    max_steps: int = 100,
    max_memory_length: int = 25,
    plan: bool = False,
    overwriting: bool = False,
    retry_failed: bool = False,
    dbt_only: bool = True,
) -> None:
    """对给定的项目列表逐个调用上面的单项目接口；suffix=project。"""
    for proj in projects:
        run_transform_agent_for_project(
            project=proj,
            input_dir=input_dir,
            output_dir=output_dir,
            model=model,
            temperature=temperature,
            top_p=top_p,
            max_tokens=max_tokens,
            max_steps=max_steps,
            max_memory_length=max_memory_length,
            plan=plan,
            overwriting=overwriting,
            retry_failed=retry_failed,
            dbt_only=dbt_only,
        )
