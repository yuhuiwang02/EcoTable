from schema_linking_DL import run_singlepair_by_paths
from LLM_Validation import run_two_step_filter_by_paths
from join import run_join_graph_builder
from run import run_transform_agent_for_projects
import argparse, os, json

def main():
    import argparse
    import os

    parser = argparse.ArgumentParser(description="Full pipeline: Stage1→Stage2→Stage3→Stage4")
    # —— Stage 1 & 2 ——（Schema Linking + LLM 两步筛选）
    parser.add_argument("--data_file_path", type=str, required=True, help="Stage1 数据 JSON（schema linking 数据集）")
    parser.add_argument("--model_path", type=str, required=True, help="RoBERTa 模型路径")
    parser.add_argument("--save_dir", type=str, required=True, help="Stage1 输出目录")
    parser.add_argument("--original_data_file", type=str, required=True, help="Stage2 所需原始标注 JSON")
    parser.add_argument("--two_step_output_dir", type=str, required=True, help="Stage2 输出目录")

    # —— Stage 3 ——（Join Graph Builder：不改输出格式、无返回值）
    parser.add_argument("--dataset_json_path", type=str, required=True, help="Join Graph 用数据（如 dataair.json）")
    parser.add_argument("--csv_root_dir", type=str, required=True, help="CSV 根目录，按项目名分子目录")
    parser.add_argument("--project_json_root", type=str, required=True, help="JoinClassify 输出目录")
    parser.add_argument("--join_model_dir", type=str, required=True, help="JoinClassify 模型目录")
    parser.add_argument("--outputset_root_dir", type=str, required=True, help="组合 JSON 输出根目录")
    parser.add_argument("--llm_judge_root", type=str, required=True, help="两段大模型判别的中间产物目录")
    parser.add_argument("--stage3_log_dir", type=str, default="./newlogs", help="Stage3 日志目录")

    # —— Stage 4 ——（Transform Agent）
    parser.add_argument("--transform_input_root", type=str, required=True,
                        help="Transform 输入根目录（通常为 <outputset_root_dir>/withoutllm）")
    parser.add_argument("--transform_output_dir", type=str, default="output", help="Transform 输出根目录")
    parser.add_argument("--transform_projects", type=str, default="github,asana,fivetran_log",
                        help="要运行的项目列表，逗号分隔，例如 'github,asana,fivetran_log'")

    # Transform Agent 推理参数（沿用默认）
    parser.add_argument("--ta_model", type=str, default="gpt-4o")
    parser.add_argument("--ta_temperature", type=float, default=0.0)
    parser.add_argument("--ta_top_p", type=float, default=0.9)
    parser.add_argument("--ta_max_tokens", type=int, default=2500)
    parser.add_argument("--ta_max_steps", type=int, default=100)
    parser.add_argument("--ta_max_memory_length", type=int, default=25)
    parser.add_argument("--ta_plan", action="store_true", default=False)
    parser.add_argument("--ta_overwriting", action="store_true", default=False)
    parser.add_argument("--ta_retry_failed", action="store_true", default=False)
    parser.add_argument("--ta_dbt_only", action="store_true", default=True)

    args = parser.parse_args()

    # ========== Stage 1：Schema Linking ==========
    sp_out = run_singlepair_by_paths(
        data_file_path=args.data_file_path,
        model_path=args.model_path,
        save_dir=args.save_dir,
    )
    contrastive_json = sp_out["artifacts"]["contrastive_results_json"]
    if not os.path.exists(contrastive_json):
        fallback = os.path.join(args.save_dir, "full_dataset_detailed_results.json")
        if os.path.exists(fallback):
            contrastive_json = fallback
        else:
            raise FileNotFoundError(
                f"找不到对比阶段详情：{sp_out['artifacts']['contrastive_results_json']} 或 {fallback}"
            )
    print(f"[Stage1] contrastive_results_json: {contrastive_json}")

    # ========== Stage 2：LLM 两步筛选 ==========
    ts_out = run_two_step_filter_by_paths(
        contrastive_results_file=contrastive_json,
        original_data_file=args.original_data_file,
        output_dir=args.two_step_output_dir,
    )
    print(f"[Stage2] results_json: {ts_out['results_json']}")
    print(f"[Stage2] llm_interactions_json: {ts_out['llm_interactions_json']}")

    # ========== Stage 3：Join Graph Builder（无返回值；不改输出格式）==========
    run_join_graph_builder(
        dataset_json_path=args.dataset_json_path,
        csv_root_dir=args.csv_root_dir,
        project_json_root=args.project_json_root,
        model_dir=args.join_model_dir,
        outputset_root_dir=args.outputset_root_dir,
        llm_judge_root=args.llm_judge_root,
        log_dir=args.stage3_log_dir
    )
    print(f"[Stage3] Done. Outputs under: {args.outputset_root_dir}")

    # ========== Stage 4：Transform Agent（逐项目执行；suffix=project）==========
    projects = [p.strip() for p in args.transform_projects.split(",") if p.strip()]
    run_transform_agent_for_projects(
        projects=projects,
        input_dir=args.transform_input_root,
        output_dir=args.transform_output_dir,
        model=args.ta_model,
        temperature=args.ta_temperature,
        top_p=args.ta_top_p,
        max_tokens=args.ta_max_tokens,
        max_steps=args.ta_max_steps,
        max_memory_length=args.ta_max_memory_length,
        plan=args.ta_plan,
        overwriting=args.ta_overwriting,
        retry_failed=args.ta_retry_failed,
        dbt_only=args.ta_dbt_only,
    )
    print(f"[Stage4] Done. Outputs under: {args.transform_output_dir}")


if __name__ == "__main__":
    main()
