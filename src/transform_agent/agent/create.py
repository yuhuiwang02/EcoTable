import json
import os

def cast_text(expr: str) -> str:
    """确保表达式转化为字符串进行比较"""
    return f"CAST({expr} AS TEXT)"

def generate_total_sql(json_data):
    alias_idx = 1
    alias_of = {}  # 已加入 FROM 的 {table_name: alias}
    from_sql = ""  # 存放 SQL 连接部分
    pending_and = []  # 若遇到重复关系，积累 AND 条件

    def new_alias():
        """生成唯一的别名"""
        nonlocal alias_idx
        a = f"t{alias_idx}"
        alias_idx += 1
        return a

    def on_clause(a_left, col_left, a_right, col_right):
        """生成连接条件，强制转换为字符串比较"""
        return f"{cast_text(f'{a_left}.{col_left}')} = {cast_text(f'{a_right}.{col_right}')}"

    for i, item in enumerate(json_data):
        t1, t2 = item["tables"]
        join_type = item["join_relation"]["Join"]
        join_pairs = item["join_relation"]["JoinPair"].split(",")  # 支持多个连接条件

        in_from_t1 = t1 in alias_of
        in_from_t2 = t2 in alias_of

        # 如果两张表都不在 FROM 里，作为起点
        if not in_from_t1 and not in_from_t2:
            a1, a2 = new_alias(), new_alias()
            alias_of[t1], alias_of[t2] = a1, a2
            on = " AND ".join([on_clause(a1, c.split("-")[0], a2, c.split("-")[1]) for c in join_pairs])
            if not from_sql:
                from_sql = f"{t1} AS {a1} {join_type} {t2} AS {a2} ON {on}"
            else:
                from_sql += f" {join_type} {t1} AS {a1} {join_type} {t2} AS {a2} ON {on}"

        # 如果 t1 已在 FROM 中，只需加入 t2
        elif in_from_t1 and not in_from_t2:
            a1 = alias_of[t1]
            a2 = new_alias()
            alias_of[t2] = a2
            on = " AND ".join([on_clause(a1, c.split("-")[0], a2, c.split("-")[1]) for c in join_pairs])
            from_sql += f" {join_type} {t2} AS {a2} ON {on}"

        # 如果 t2 已在 FROM 中，只需加入 t1
        elif not in_from_t1 and in_from_t2:
            a2 = alias_of[t2]
            a1 = new_alias()
            alias_of[t1] = a1
            on = " AND ".join([on_clause(a1, c.split("-")[0], a2, c.split("-")[1]) for c in join_pairs])
            from_sql += f" {join_type} {t1} AS {a1} ON {on}"

        # 如果 t1 和 t2 都已经在 FROM 中，积累 AND 条件
        else:
            a1, a2 = alias_of[t1], alias_of[t2]
            on = " AND ".join([on_clause(a1, c.split("-")[0], a2, c.split("-")[1]) for c in join_pairs])
            pending_and.append(on)

    # 如果有重复关系，拼接到最后一次的 ON 子句里
    if pending_and:
        from_sql += " AND " + " AND ".join(pending_and)

    return f"SELECT * FROM {from_sql}"

def process_json_to_sql(json_file_path):
    # 获取文件所在目录
    dir_path = os.path.dirname(json_file_path) or "."
    with open(json_file_path, "r") as f:
        data = json.load(f)
    sql = generate_total_sql(data) + ";\n"
    out = os.path.join(dir_path, "output.sql")
    with open(out, "w") as f:
        f.write(sql)
    print(f"written: {out}")

