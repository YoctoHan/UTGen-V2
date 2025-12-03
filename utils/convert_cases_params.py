import argparse
import ast
import json
import os
import re
from typing import Any, Dict, List


STRUCT_FIELDS = [
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
    提取 C++ 中 COMPILE_INFO 常量里的 JSON 字符串内容。
    """
    pattern = re.compile(
        r'const\s+std::string\s+COMPILE_INFO\s*=\s*R"\((.*?)\)";',
        re.DOTALL,
    )
    m = pattern.search(src)
    if not m:
        return ""
    return m.group(1).strip()


def extract_cases_block(src: str) -> str:
    """
    提取 AllGatherMatmulTilingTestParam cases_params[] 的初始化整体文本（去掉最外层花括号）。
    """
    pattern = re.compile(
        r"AllGatherMatmulTilingTestParam\s+cases_params\s*\[\]\s*=\s*\{(.*?)\};",
        re.DOTALL,
    )
    m = pattern.search(src)
    if not m:
        raise ValueError("未找到 AllGatherMatmulTilingTestParam cases_params[] 初始化块")
    return m.group(1).strip()


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


def parse_case(case_str: str, compile_info_value: str) -> Dict[str, Any]:
    tokens = split_top_level_commas(case_str)
    if len(tokens) != len(STRUCT_FIELDS):
        raise ValueError(
            f"字段数量不匹配，期望 {len(STRUCT_FIELDS)} 个，实际 {len(tokens)} 个：{case_str}"
        )

    res: Dict[str, Any] = {}

    # 0: inputTotalNum
    res["inputTotalNum"] = parse_int(tokens[0])

    # 1: case_name（C++ 字符串字面量）
    res["case_name"] = ast.literal_eval(tokens[1].strip())

    # 2: compile_info（COMPILE_INFO 常量）
    res["compile_info"] = compile_info_value

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


def convert(src_path: str, dst_path: str) -> None:
    with open(src_path, "r", encoding="utf-8") as f:
        src = f.read()

    compile_info_value = extract_compile_info(src)
    cases_block = extract_cases_block(src)
    case_inits = split_case_initializers(cases_block)

    os.makedirs(os.path.dirname(dst_path), exist_ok=True)
    with open(dst_path, "w", encoding="utf-8") as out:
        for case_str in case_inits:
            obj = parse_case(case_str, compile_info_value)
            out.write(json.dumps(obj, ensure_ascii=False))
            out.write("\n")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="从 test_all_gather_matmul_tiling.cpp 中抽取 cases_params 用例并生成 JSONL 文件"
    )
    parser.add_argument(
        "--src",
        default="target/test_all_gather_matmul_tiling.cpp",
        help="源 C++ 测试文件路径",
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


