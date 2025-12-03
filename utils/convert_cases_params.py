import argparse
import ast
import json
import os
import re
from typing import Any, Dict, List


MATMUL_STRUCT_FIELDS = [
    "inputTotalNum",
    "case_name",
    "compile_info",
    "soc_version",
    "coreNum",
    "ubSize",
    "tilingDataSize",
    "x1_shape",
    "x2_shape",
    "bias_shape",
    "x3_shape",
    "antiquant_scale_shape",
    "antiquant_offset_shape",
    "dequant_scale_shape",
    "pertoken_scale_shape",
    "comm_quant_scale_1_shape",
    "comm_quant_scale_2_shape",
    "output_shape",
    "x1_dtype",
    "x2_dtype",
    "bias_dtype",
    "x3_dtype",
    "antiquant_scale_dtype",
    "antiquant_offset_dtype",
    "dequant_scale_dtype",
    "pertoken_scale_dtype",
    "comm_quant_scale_1_dtype",
    "comm_quant_scale_2_dtype",
    "output_dtype",
    "is_trans_a",
    "is_trans_b",
    "expectTilingKey",
]


def extract_compile_info(src: str) -> str:
    """
    提取 C++ 中 COMPILE_INFO 常量里的字符串内容。
    兼容以下两种形式：
    - const std::string COMPILE_INFO = R"({ ... })";
    - const std::string COMPILE_INFO = "8 8 20 196352 0 0 ";
    """
    # 先匹配原始字符串 R"( ... )"
    pattern_raw = re.compile(
        r'const\s+(?:std::)?string\s+COMPILE_INFO\s*=\s*R"\((.*?)\)";',
        re.DOTALL,
    )
    m = pattern_raw.search(src)
    if m:
        return m.group(1).strip()

    # 再匹配普通字符串
    pattern_str = re.compile(
        r'const\s+(?:std::)?string\s+COMPILE_INFO\s*=\s*"([^"]*)";',
        re.DOTALL,
    )
    m = pattern_str.search(src)
    if not m:
        return ""
    return m.group(1)


def detect_mode_and_cases_block(src: str) -> (str, str):
    """
    检测当前 UT 文件类型，并提取 cases_params[] 初始化整体文本（去掉最外层花括号）。

    返回 (mode, cases_block)，mode 取值：
    - "all_gather_matmul"
    - "matmul_all_reduce"
    - "distribute_barrier"
    """
    patterns = [
        ("all_gather_matmul", r"AllGatherMatmulTilingTestParam\s+cases_params\s*\[\]\s*=\s*\{(.*?)\};"),
        ("matmul_all_reduce", r"MatmulAllReduceTilingTestParam\s+cases_params\s*\[\]\s*=\s*\{(.*?)\};"),
        ("distribute_barrier", r"DistributeBarrierTilingTestParam\s+cases_params\s*\[\]\s*=\s*\{(.*?)\};"),
    ]

    for mode, pat in patterns:
        m = re.search(pat, src, re.DOTALL)
        if m:
            return mode, m.group(1).strip()

    raise ValueError("未能识别 UT 文件类型（未找到 *TilingTestParam cases_params[] 初始化块）")


def split_case_initializers(cases_block: str) -> List[str]:
    """
    按最外层花括号深度切分每个用例的初始化：
    { ... },\n{ ... }, ...
    """
    items: List[str] = []
    depth = 0
    start = None
    for i, ch in enumerate(cases_block):
        if ch == "{":
            if depth == 0:
                start = i
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0 and start is not None:
                # 包含最外层花括号
                items.append(cases_block[start : i + 1])
                start = None
    return [s.strip() for s in items if s.strip()]


def split_top_level_commas(s: str) -> List[str]:
    """
    在不进入字符串且最外层（不在额外花括号内）的位置按逗号切分。
    用于把单个 case 初始化里的 32 个字段拆开。
    """
    # 去掉最外层花括号
    s = s.strip()
    if s.startswith("{") and s.endswith("}"):
        s = s[1:-1]

    parts: List[str] = []
    cur: List[str] = []
    depth_brace = 0
    in_string = False
    prev_char = ""

    for ch in s:
        if ch == '"' and prev_char != "\\":
            in_string = not in_string
            cur.append(ch)
        elif not in_string:
            if ch == "{":
                depth_brace += 1
                cur.append(ch)
            elif ch == "}":
                depth_brace -= 1
                cur.append(ch)
            elif ch == "," and depth_brace == 0:
                part = "".join(cur).strip()
                parts.append(part)
                cur = []
            else:
                cur.append(ch)
        else:
            cur.append(ch)
        prev_char = ch

    if cur:
        parts.append("".join(cur).strip())
    return parts


def parse_int(token: str) -> int:
    token = token.strip()
    # 去掉 C++ 整形后缀，比如 110UL / 110U / 110L 等
    token = re.sub(r"[uU]?[lL]+$", "", token)
    return int(token, 0)


def parse_shape(token: str) -> List[int]:
    token = token.strip()
    if token == "{}":
        return []
    m = re.match(r"\{(.*)\}", token)
    if not m:
        raise ValueError(f"无法解析 shape 字段: {token}")
    inner = m.group(1).strip()
    if not inner:
        return []
    return [parse_int(x) for x in inner.split(",") if x.strip()]


def parse_bool(token: str) -> bool:
    token = token.strip()
    if token == "true":
        return True
    if token == "false":
        return False
    raise ValueError(f"无法解析布尔值: {token}")


def parse_string_or_const(token: str, const_value: str) -> str:
    """
    解析字符串字段，如果是 COMPILE_INFO 这样的常量名，则替换为 const_value。
    """
    token = token.strip()
    if const_value and token == "COMPILE_INFO":
        return const_value
    # 普通 C++ 字符串字面量
    if token.startswith('"'):
        return ast.literal_eval(token)
    return token


def parse_case_matmul_like(case_str: str, compile_info_value: str) -> Dict[str, Any]:
    """
    解析 AllGatherMatmul / MatmulAllReduce 这类结构体的用例。
    字段布局与 MATMUL_STRUCT_FIELDS 一致。
    """
    tokens = split_top_level_commas(case_str)
    if len(tokens) != len(MATMUL_STRUCT_FIELDS):
        raise ValueError(
            f"字段数量不匹配，期望 {len(MATMUL_STRUCT_FIELDS)} 个，实际 {len(tokens)} 个：{case_str}"
        )

    res: Dict[str, Any] = {}

    # 0: inputTotalNum
    res["inputTotalNum"] = parse_int(tokens[0])

    # 1: case_name（C++ 字符串字面量）
    res["case_name"] = ast.literal_eval(tokens[1].strip())

    # 2: compile_info（通常是 COMPILE_INFO 常量）
    if compile_info_value:
        res["compile_info"] = compile_info_value
    else:
        # 兜底：直接从初始化里的字符串解析
        res["compile_info"] = parse_string_or_const(tokens[2], "")

    # 3: soc_version
    res["soc_version"] = ast.literal_eval(tokens[3].strip())

    # 4,5,6: coreNum, ubSize, tilingDataSize
    res["coreNum"] = parse_int(tokens[4])
    res["ubSize"] = parse_int(tokens[5])
    res["tilingDataSize"] = parse_int(tokens[6])

    # 7-17: 各种 shape
    res["x1_shape"] = parse_shape(tokens[7])
    res["x2_shape"] = parse_shape(tokens[8])
    res["bias_shape"] = parse_shape(tokens[9])
    res["x3_shape"] = parse_shape(tokens[10])
    res["antiquant_scale_shape"] = parse_shape(tokens[11])
    res["antiquant_offset_shape"] = parse_shape(tokens[12])
    res["dequant_scale_shape"] = parse_shape(tokens[13])
    res["pertoken_scale_shape"] = parse_shape(tokens[14])
    res["comm_quant_scale_1_shape"] = parse_shape(tokens[15])
    res["comm_quant_scale_2_shape"] = parse_shape(tokens[16])
    res["output_shape"] = parse_shape(tokens[17])

    # 18-28: dtype 信息，直接保留 C++ 标识符字符串
    dtype_field_names = [
        "x1_dtype",
        "x2_dtype",
        "bias_dtype",
        "x3_dtype",
        "antiquant_scale_dtype",
        "antiquant_offset_dtype",
        "dequant_scale_dtype",
        "pertoken_scale_dtype",
        "comm_quant_scale_1_dtype",
        "comm_quant_scale_2_dtype",
        "output_dtype",
    ]
    for idx, name in enumerate(dtype_field_names, start=18):
        res[name] = tokens[idx].strip()

    # 29,30: is_trans_a, is_trans_b
    res["is_trans_a"] = parse_bool(tokens[29])
    res["is_trans_b"] = parse_bool(tokens[30])

    # 31: expectTilingKey
    res["expectTilingKey"] = parse_int(tokens[31])

    return res


def parse_size_t_vec(token: str) -> List[int]:
    """
    解析 std::vector<size_t> 形式的初始化 {16777216, ...}。
    """
    token = token.strip()
    if token == "{}":
        return []
    m = re.match(r"\{(.*)\}", token)
    if not m:
        raise ValueError(f"无法解析 size_t 向量字段: {token}")
    inner = m.group(1).strip()
    if not inner:
        return []
    return [parse_int(x) for x in inner.split(",") if x.strip()]


def parse_case_distribute_barrier(case_str: str, compile_info_value: str) -> Dict[str, Any]:
    """
    解析 DistributeBarrierTilingTestParam 用例。
    字段顺序：
    case_name, m, n, dtype, group, world_size, soc_version,
    coreNum, ubSize, expectTilingKey, expectTilingData, expectWorkspaces,
    mc2TilingDataReservedLen
    """
    tokens = split_top_level_commas(case_str)
    if len(tokens) != 13:
        raise ValueError(f"DistributeBarrier 用例字段数量不匹配，实际 {len(tokens)} 个：{case_str}")

    res: Dict[str, Any] = {}

    res["case_name"] = ast.literal_eval(tokens[0].strip())
    res["m"] = parse_int(tokens[1])
    res["n"] = parse_int(tokens[2])
    res["dtype"] = tokens[3].strip()
    res["group"] = ast.literal_eval(tokens[4].strip())
    res["world_size"] = parse_int(tokens[5])
    res["soc_version"] = ast.literal_eval(tokens[6].strip())
    res["coreNum"] = parse_int(tokens[7])
    res["ubSize"] = parse_int(tokens[8])
    res["expectTilingKey"] = parse_int(tokens[9])

    # expectTilingData：可能是字符串字面量，也可能是 COMPILE_INFO 常量
    res["expectTilingData"] = parse_string_or_const(tokens[10], compile_info_value)

    # std::vector<size_t>
    res["expectWorkspaces"] = parse_size_t_vec(tokens[11])

    res["mc2TilingDataReservedLen"] = parse_int(tokens[12])

    return res


def convert(src_path: str, dst_path: str) -> None:
    with open(src_path, "r", encoding="utf-8") as f:
        src = f.read()

    compile_info_value = extract_compile_info(src)
    mode, cases_block = detect_mode_and_cases_block(src)
    case_inits = split_case_initializers(cases_block)

    os.makedirs(os.path.dirname(dst_path), exist_ok=True)
    with open(dst_path, "w", encoding="utf-8") as out:
        for case_str in case_inits:
            if mode in ("all_gather_matmul", "matmul_all_reduce"):
                obj = parse_case_matmul_like(case_str, compile_info_value)
            elif mode == "distribute_barrier":
                obj = parse_case_distribute_barrier(case_str, compile_info_value)
            else:
                raise ValueError(f"不支持的模式: {mode}")

            out.write(json.dumps(obj, ensure_ascii=False))
            out.write("\n")


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "从 C++ UT 文件中抽取 cases_params 用例并生成 JSONL 文件，"
            "当前支持：AllGatherMatmul / MatmulAllReduce / DistributeBarrier"
        )
    )
    parser.add_argument(
        "--src",
        default="target/test_all_gather_matmul_tiling.cpp",
        help="源 C++ 测试文件路径（三种之一）",
    )
    parser.add_argument(
        "--dst",
        default="input/all_gather_matmul.jsonl",
        help="输出 JSONL 文件路径",
    )
    args = parser.parse_args()

    convert(args.src, args.dst)


if __name__ == "__main__":
    main()


