import argparse
import ast
import json
import os
import re
from typing import Any, Dict, List, Optional


# 完整版字段（有 compile_info, soc_version 等）
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
    # V1 只有 expectTilingKey，V2 在其前面新增了 expectSuccess
    "expectTilingKey",
]

# 简化版字段（无 compile_info, soc_version）
MATMUL_SIMPLE_STRUCT_FIELDS = [
    "inputTotalNum",
    "case_name",
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

# 默认硬件配置
DEFAULT_COMPILE_INFO = json.dumps({
    "hardware_info": {
        "BT_SIZE": 0,
        "load3d_constraints": "1",
        "Intrinsic_fix_pipe_l0c2out": False,
        "Intrinsic_data_move_l12ub": True,
        "Intrinsic_data_move_l0c2ub": True,
        "Intrinsic_data_move_out2l1_nd2nz": False,
        "UB_SIZE": 196608,
        "L2_SIZE": 33554432,
        "L1_SIZE": 524288,
        "L0A_SIZE": 65536,
        "L0B_SIZE": 65536,
        "L0C_SIZE": 131072,
        "CORE_NUM": 20,
        "socVersion": "Ascend910B"
    }
})

DEFAULT_SOC_VERSION = "Ascend910B"
DEFAULT_CORE_NUM = 20
DEFAULT_UB_SIZE = 196608
DEFAULT_TILING_DATA_SIZE = 4096


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

    兼容更多 Tiling UT：
    - 先通过通用的 "*TilingTestParam cases_params[]" 形式定位 cases 块；
    - 再根据结构体名 / 字段数量判定解析模式。

    返回 (mode, cases_block)，mode 取值示例：
    - "matmul_like"
    - "matmul_simple"
    - "distribute_barrier"
    - "matmul_reduce_scatter"
    - "matmul_reduce_scatter_v2"
    - "matmul_reduce_scatter_v2_new"  (新格式，含嵌套 TensorDescParam)
    - "grouped_matmul_all_reduce"
    - "batch_matmul_reduce_scatter_alltoall"
    - "allto_all_all_gather_bmm"
    - "moe_distribute_dispatch"
    - "moe_distribute_dispatch_v2"
    - "moe_distribute_combine"
    - "moe_distribute_combine_add_rms_norm"
    - "moe_distribute_combine_v2"
    - "allto_allv_grouped_matmul"  (使用 TestParam 结构体)
    - "matmul_all_reduce_add_rms_norm" (使用 TestParam 结构体的简化格式)
    """
    # 1. 通用匹配：任意 XXXTilingTestParam cases_params[] = { ... }; 或 const XXXParam cases_params[] = { ... };
    generic_pat = re.compile(
        r"(?:const\s+)?(\w+(?:TilingTestParam|TestParam|Param))\s+(?:cases_params|test_params)\s*\[\]\s*=\s*\{(.*?)\};",
        re.DOTALL,
    )
    m = generic_pat.search(src)

    # 如果没找到，尝试匹配 static TestParam casesParamsQuant[] = { ... }; 或类似的变体
    if not m:
        alt_pat = re.compile(
            r"(?:static\s+)?(\w+(?:TestParam|Param))\s+(\w+)\s*\[\]\s*=\s*\{(.*?)\};",
            re.DOTALL,
        )
        # 查找所有匹配的数组
        all_matches = alt_pat.findall(src)
        if all_matches:
            # 合并所有同类型数组的内容
            combined_blocks = []
            struct_name = None
            for match in all_matches:
                s_name = match[0]
                var_name = match[1]
                block = match[2].strip()
                if struct_name is None:
                    struct_name = s_name
                if s_name == struct_name:
                    combined_blocks.append(block)

            if combined_blocks:
                cases_block = ", ".join(combined_blocks)
                # 检测是否是 MatmulAllReduceAddRmsNorm 的简化 TestParam
                if struct_name == "TestParam":
                    case_inits = split_case_initializers(cases_block)
                    if case_inits:
                        first_tokens = split_top_level_commas(case_inits[0])
                        # MatmulAllReduceAddRmsNorm 的 TestParam 有 3 个字段
                        if len(first_tokens) == 3:
                            return "matmul_all_reduce_add_rms_norm", cases_block

    if not m:
        raise ValueError("未能识别 UT 文件类型（未找到 *TilingTestParam cases_params[] 或 test_params[] 初始化块）")

    struct_name = m.group(1)
    cases_block = m.group(2).strip()

    # 2. 针对部分特殊 struct_name，直接指定解析模式
    if struct_name == "MatmulReduceScatterTilingTestParam":
        return "matmul_reduce_scatter", cases_block
    if struct_name == "MatmulReduceScatterV2TilingTestParam":
        # 检测是新格式（含 TensorDescParam）还是旧格式
        case_inits = split_case_initializers(cases_block)
        if case_inits:
            first_tokens = split_top_level_commas(case_inits[0])
            # 新格式有 17 个字段（包含 inputs/outputs 嵌套结构）
            if len(first_tokens) == 17:
                return "matmul_reduce_scatter_v2_new", cases_block
        return "matmul_reduce_scatter_v2", cases_block
    if struct_name == "GroupedMatMulAllReduceTilingTestParam":
        return "grouped_matmul_all_reduce", cases_block
    if struct_name == "BatchMatMulReduceScatterAlltoAllTilingTestParam":
        return "batch_matmul_reduce_scatter_alltoall", cases_block
    if struct_name == "AlltoAllAllGatherBmmTilingTestParam":
        return "allto_all_all_gather_bmm", cases_block
    if struct_name == "MoeDistributeDispatchTilingTestParam":
        return "moe_distribute_dispatch", cases_block
    if struct_name == "MoeDistributeDispatchV2TilingTestParam":
        return "moe_distribute_dispatch_v2", cases_block
    if struct_name == "MoeDistributeCombineTilingTestParam":
        return "moe_distribute_combine", cases_block
    if struct_name == "MoeDistributeCombineAddRmsNormTilingTestParam":
        return "moe_distribute_combine_add_rms_norm", cases_block
    if struct_name == "MoeDistributeCombineV2TilingTestParam":
        return "moe_distribute_combine_v2", cases_block
    if struct_name == "TestParam":
        # AlltoAllvGroupedMatMul 使用 TestParam 结构体
        return "allto_allv_grouped_matmul", cases_block

    # 3. 其它情况：拆出第一个 case，基于字段数量推断是 matmul-like / distribute_barrier / matmul_simple
    case_inits = split_case_initializers(cases_block)
    if not case_inits:
        raise ValueError("未在 cases_params[] 中解析到任何用例初始化")

    first_tokens = split_top_level_commas(case_inits[0])
    field_cnt = len(first_tokens)

    # matmul-like 完整版：AllGatherMatmulV2 / MatmulAllReduce 等（有 compile_info, soc_version）
    if field_cnt in (len(MATMUL_STRUCT_FIELDS), len(MATMUL_STRUCT_FIELDS) + 1):
        return "matmul_like", cases_block

    # matmul_simple 简化版：AllGatherMatmul 等（无 compile_info, soc_version）
    if field_cnt in (len(MATMUL_SIMPLE_STRUCT_FIELDS), len(MATMUL_SIMPLE_STRUCT_FIELDS) + 1):
        return "matmul_simple", cases_block

    # distribute_barrier：固定 13 个字段
    if field_cnt == 13:
        return "distribute_barrier", cases_block

    raise ValueError(
        f"暂不支持的用例结构体 {struct_name}，单个用例字段数为 {field_cnt}，"
        f"既不是 matmul-like({len(MATMUL_STRUCT_FIELDS)} 或 "
        f"{len(MATMUL_STRUCT_FIELDS) + 1})，"
        f"也不是 matmul_simple({len(MATMUL_SIMPLE_STRUCT_FIELDS)} 或 "
        f"{len(MATMUL_SIMPLE_STRUCT_FIELDS) + 1})，"
        f"也不是 distribute_barrier(13)"
    )


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
    token = _strip_cpp_line_comment_prefix(token).strip()
    # 去掉 C++ 整形后缀，比如 110UL / 110U / 110L 等
    token = re.sub(r"[uU]?[lL]+$", "", token)
    try:
        return int(token, 0)
    except ValueError:
        # 兼容简单编译期常量表达式（例如 "32 * 8"），退化为安全 eval
        try:
            return int(eval(token, {"__builtins__": {}}, {}))
        except Exception as e:  # noqa: BLE001
            raise ValueError(f"无法解析整型字段: {token}") from e


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


def _strip_cpp_line_comment_prefix(token: str) -> str:
    """
    去掉形如 `// xxx` 的单行注释前缀（只处理出现在最前面的情况）。
    """
    return re.sub(r"^\s*//[^\n]*\n", "", token)


def parse_cpp_string_literal(token: str) -> str:
    """
    解析 C++ 字符串字面量，自动去除前置行注释。
    """
    token = _strip_cpp_line_comment_prefix(token).strip()
    return ast.literal_eval(token)


def parse_shape_2d(token: str) -> List[List[int]]:
    """
    解析形如 {{16, 128, 64}, {4, 64, 128}} 的二维 shape 列表。
    """
    token = token.strip()
    if token == "{}":
        return []
    if not (token.startswith("{") and token.endswith("}")):
        raise ValueError(f"无法解析二维 shape 字段: {token}")
    inner = token[1:-1].strip()
    if not inner:
        return []

    shapes: List[List[int]] = []
    depth = 0
    start = None
    for i, ch in enumerate(inner):
        if ch == "{":
            if depth == 0:
                start = i
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0 and start is not None:
                sub = inner[start : i + 1]
                shapes.append(parse_shape(sub))
                start = None
    return shapes


def parse_int_vec(token: str) -> List[int]:
    """
    解析 {1, 2, 3} 这类一维整型 vector。
    """
    return parse_shape(token)


def parse_dtype_vec(token: str) -> List[str]:
    """
    解析 {ge::DT_FLOAT16, ge::DT_FLOAT16} 这类 DataType 向量。
    """
    token = token.strip()
    if token == "{}":
        return []
    m = re.match(r"\{(.*)\}", token)
    if not m:
        raise ValueError(f"无法解析 dtype 向量字段: {token}")
    inner = m.group(1).strip()
    if not inner:
        return []
    return [part.strip() for part in inner.split(",") if part.strip()]


def parse_case_matmul_like(case_str: str, compile_info_value: str) -> Dict[str, Any]:
    """
    解析 AllGatherMatmul / MatmulAllReduce 这类结构体的用例。
    字段布局与 MATMUL_STRUCT_FIELDS 一致。
    """
    tokens = split_top_level_commas(case_str)
    # V1：字段数与 MATMUL_STRUCT_FIELDS 相同；
    # V2：在 expectTilingKey 前多了一个 bool expectSuccess 字段。
    if len(tokens) not in (len(MATMUL_STRUCT_FIELDS), len(MATMUL_STRUCT_FIELDS) + 1):
        raise ValueError(
            f"字段数量不匹配，期望 {len(MATMUL_STRUCT_FIELDS)} 或 "
            f"{len(MATMUL_STRUCT_FIELDS) + 1} 个，实际 {len(tokens)} 个：{case_str}"
        )

    has_expect_success = len(tokens) == len(MATMUL_STRUCT_FIELDS) + 1

    res: Dict[str, Any] = {}

    # 0: inputTotalNum
    res["inputTotalNum"] = parse_int(tokens[0])

    # 1: case_name（C++ 字符串字面量）
    res["case_name"] = parse_cpp_string_literal(tokens[1])

    # 2: compile_info（通常是 COMPILE_INFO 常量）
    if compile_info_value:
        res["compile_info"] = compile_info_value
    else:
        # 兜底：直接从初始化里的字符串解析
        res["compile_info"] = parse_string_or_const(tokens[2], "")

    # 3: soc_version
    res["soc_version"] = parse_cpp_string_literal(tokens[3])

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

    # V1: 31 为 expectTilingKey
    # V2: 31 为 expectSuccess，32 为 expectTilingKey
    if has_expect_success:
        res["expectSuccess"] = parse_bool(tokens[31])
        expect_key_idx = 32
    else:
        expect_key_idx = 31

    res["expectTilingKey"] = parse_int(tokens[expect_key_idx])

    return res


def parse_case_matmul_simple(case_str: str, default_compile_info: str = DEFAULT_COMPILE_INFO,
                              default_soc_version: str = DEFAULT_SOC_VERSION,
                              default_core_num: int = DEFAULT_CORE_NUM,
                              default_ub_size: int = DEFAULT_UB_SIZE,
                              default_tiling_data_size: int = DEFAULT_TILING_DATA_SIZE) -> Dict[str, Any]:
    """
    解析简化版 AllGatherMatmul 结构体的用例。
    该版本没有 compile_info 和 soc_version 字段，需要填充默认值。
    字段布局与 MATMUL_SIMPLE_STRUCT_FIELDS 一致。
    """
    tokens = split_top_level_commas(case_str)
    # 简化版字段数
    if len(tokens) not in (len(MATMUL_SIMPLE_STRUCT_FIELDS), len(MATMUL_SIMPLE_STRUCT_FIELDS) + 1):
        raise ValueError(
            f"简化版字段数量不匹配，期望 {len(MATMUL_SIMPLE_STRUCT_FIELDS)} 或 "
            f"{len(MATMUL_SIMPLE_STRUCT_FIELDS) + 1} 个，实际 {len(tokens)} 个：{case_str}"
        )

    has_expect_success = len(tokens) == len(MATMUL_SIMPLE_STRUCT_FIELDS) + 1

    res: Dict[str, Any] = {}

    # 0: inputTotalNum
    res["inputTotalNum"] = parse_int(tokens[0])

    # 1: case_name（C++ 字符串字面量）
    res["case_name"] = parse_cpp_string_literal(tokens[1])

    # 填充默认值
    res["compile_info"] = default_compile_info
    res["soc_version"] = default_soc_version

    # 2,3,4: coreNum, ubSize, tilingDataSize（简化版中值可能为 0，需要用默认值替换）
    parsed_core_num = parse_int(tokens[2])
    parsed_ub_size = parse_int(tokens[3])
    parsed_tiling_data_size = parse_int(tokens[4])

    res["coreNum"] = parsed_core_num if parsed_core_num > 0 else default_core_num
    res["ubSize"] = parsed_ub_size if parsed_ub_size > 0 else default_ub_size
    res["tilingDataSize"] = parsed_tiling_data_size if parsed_tiling_data_size > 0 else default_tiling_data_size

    # 5-15: 各种 shape
    res["x1_shape"] = parse_shape(tokens[5])
    res["x2_shape"] = parse_shape(tokens[6])
    res["bias_shape"] = parse_shape(tokens[7])
    res["x3_shape"] = parse_shape(tokens[8])
    res["antiquant_scale_shape"] = parse_shape(tokens[9])
    res["antiquant_offset_shape"] = parse_shape(tokens[10])
    res["dequant_scale_shape"] = parse_shape(tokens[11])
    res["pertoken_scale_shape"] = parse_shape(tokens[12])
    res["comm_quant_scale_1_shape"] = parse_shape(tokens[13])
    res["comm_quant_scale_2_shape"] = parse_shape(tokens[14])
    res["output_shape"] = parse_shape(tokens[15])

    # 16-26: dtype 信息，直接保留 C++ 标识符字符串
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
    for idx, name in enumerate(dtype_field_names, start=16):
        res[name] = tokens[idx].strip()

    # 27,28: is_trans_a, is_trans_b
    res["is_trans_a"] = parse_bool(tokens[27])
    res["is_trans_b"] = parse_bool(tokens[28])

    # 简化版: 29 为 expectTilingKey 或 30（若有 expectSuccess）
    if has_expect_success:
        res["expectSuccess"] = parse_bool(tokens[29])
        expect_key_idx = 30
    else:
        expect_key_idx = 29

    res["expectTilingKey"] = parse_int(tokens[expect_key_idx])

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

    res["case_name"] = parse_cpp_string_literal(tokens[0])
    res["m"] = parse_int(tokens[1])
    res["n"] = parse_int(tokens[2])
    res["dtype"] = tokens[3].strip()
    res["group"] = parse_cpp_string_literal(tokens[4])
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


def parse_case_matmul_reduce_scatter(case_str: str) -> Dict[str, Any]:
    """
    解析 MatmulReduceScatterTilingTestParam 用例。
    字段顺序：
    inputTotalNum, case_name, soc_version, coreNum, ubSize,
    x1_shape, x2_shape, x3_shape, x4_shape,
    y_shape,
    x1_dtype, x2_dtype, x3_dtype, x4_dtype, y_dtype,
    is_trans_a, is_trans_b,
    expectTilingKey
    """
    tokens = split_top_level_commas(case_str)
    if len(tokens) != 18:
        raise ValueError(f"MatmulReduceScatter 用例字段数量不匹配，实际 {len(tokens)} 个：{case_str}")

    res: Dict[str, Any] = {}

    res["inputTotalNum"] = parse_int(tokens[0])
    res["case_name"] = parse_cpp_string_literal(tokens[1])
    res["soc_version"] = parse_cpp_string_literal(tokens[2])
    res["coreNum"] = parse_int(tokens[3])
    res["ubSize"] = parse_int(tokens[4])

    res["x1_shape"] = parse_shape(tokens[5])
    res["x2_shape"] = parse_shape(tokens[6])
    res["x3_shape"] = parse_shape(tokens[7])
    res["x4_shape"] = parse_shape(tokens[8])
    res["y_shape"] = parse_shape(tokens[9])

    res["x1_dtype"] = tokens[10].strip()
    res["x2_dtype"] = tokens[11].strip()
    res["x3_dtype"] = tokens[12].strip()
    res["x4_dtype"] = tokens[13].strip()
    res["y_dtype"] = tokens[14].strip()

    res["is_trans_a"] = parse_bool(tokens[15])
    res["is_trans_b"] = parse_bool(tokens[16])

    res["expectTilingKey"] = parse_int(tokens[17])
    return res


def parse_case_matmul_reduce_scatter_v2(case_str: str, compile_info_value: str) -> Dict[str, Any]:
    """
    解析 MatmulReduceScatterV2TilingTestParam 用例。
    字段顺序：
    inputTotalNum, case_name, soc_version, coreNum, ubSize, tilingDataSize, compile_info,
    x1_shape, x2_shape,
    y_shape,
    x1_dtype, x2_dtype, y_dtype,
    is_trans_a, is_trans_b,
    expectTilingKey
    """
    tokens = split_top_level_commas(case_str)
    if len(tokens) != 16:
        raise ValueError(f"MatmulReduceScatterV2 用例字段数量不匹配，实际 {len(tokens)} 个：{case_str}")

    res: Dict[str, Any] = {}

    res["inputTotalNum"] = parse_int(tokens[0])
    res["case_name"] = parse_cpp_string_literal(tokens[1])
    res["soc_version"] = parse_cpp_string_literal(tokens[2])
    res["coreNum"] = parse_int(tokens[3])
    res["ubSize"] = parse_int(tokens[4])
    res["tilingDataSize"] = parse_int(tokens[5])

    # compile_info
    res["compile_info"] = parse_string_or_const(tokens[6], compile_info_value)

    res["x1_shape"] = parse_shape(tokens[7])
    res["x2_shape"] = parse_shape(tokens[8])
    res["y_shape"] = parse_shape(tokens[9])

    res["x1_dtype"] = tokens[10].strip()
    res["x2_dtype"] = tokens[11].strip()
    res["y_dtype"] = tokens[12].strip()

    res["is_trans_a"] = parse_bool(tokens[13])
    res["is_trans_b"] = parse_bool(tokens[14])

    res["expectTilingKey"] = parse_int(tokens[15])
    return res


def parse_case_grouped_matmul_all_reduce(case_str: str) -> Dict[str, Any]:
    """
    解析 GroupedMatMulAllReduceTilingTestParam 用例。
    字段顺序：
    case_name, soc_version, coreNum, ubSize, tilingDataSize,
    x1_shape, x2_shape, output_shape,
    x1_dtype, x2_dtype, output_dtype,
    rankNum, expectTilingKey
    """
    tokens = split_top_level_commas(case_str)
    # 当前结构体共有 13 个字段，如果未来有扩展，只要前 13 个顺序不变即可复用。
    if len(tokens) < 13:
        raise ValueError(f"GroupedMatMulAllReduce 用例字段数量不匹配，实际 {len(tokens)} 个：{case_str}")

    res: Dict[str, Any] = {}

    res["case_name"] = parse_cpp_string_literal(tokens[0])
    res["soc_version"] = parse_cpp_string_literal(tokens[1])
    res["coreNum"] = parse_int(tokens[2])
    res["ubSize"] = parse_int(tokens[3])
    res["tilingDataSize"] = parse_int(tokens[4])

    res["x1_shape"] = parse_shape(tokens[5])
    res["x2_shape"] = parse_shape(tokens[6])
    res["output_shape"] = parse_shape(tokens[7])

    res["x1_dtype"] = tokens[8].strip()
    res["x2_dtype"] = tokens[9].strip()
    res["output_dtype"] = tokens[10].strip()

    res["rankNum"] = parse_int(tokens[11])
    res["expectTilingKey"] = parse_int(tokens[12])

    return res


def parse_case_batch_matmul_reduce_scatter_alltoall(case_str: str) -> Dict[str, Any]:
    """
    解析 BatchMatMulReduceScatterAlltoAllTilingTestParam 用例。
    字段顺序：
    inputTotalNum, case_name, coreNum, ubSize,
    x_shape, w_shape, bias_shape,
    y_shape,
    x_dtype, w_dtype, bias_dtype, y_dtype,
    group_ep, group_tp, ep_world_size, tp_world_size, y_shard_type, transpose_weight,
    hasExpectTilingKey, expectTilingKey
    """
    tokens = split_top_level_commas(case_str)
    if len(tokens) != 20:
        raise ValueError(
            f"BatchMatMulReduceScatterAlltoAll 用例字段数量不匹配，实际 {len(tokens)} 个：{case_str}"
        )

    res: Dict[str, Any] = {}

    res["inputTotalNum"] = parse_int(tokens[0])
    res["case_name"] = parse_cpp_string_literal(tokens[1])
    res["coreNum"] = parse_int(tokens[2])
    res["ubSize"] = parse_int(tokens[3])

    res["x_shape"] = parse_shape(tokens[4])
    res["w_shape"] = parse_shape(tokens[5])
    res["bias_shape"] = parse_shape(tokens[6])
    res["y_shape"] = parse_shape(tokens[7])

    res["x_dtype"] = tokens[8].strip()
    res["w_dtype"] = tokens[9].strip()
    res["bias_dtype"] = tokens[10].strip()
    res["y_dtype"] = tokens[11].strip()

    res["group_ep"] = parse_cpp_string_literal(tokens[12])
    res["group_tp"] = parse_cpp_string_literal(tokens[13])
    res["ep_world_size"] = parse_int(tokens[14])
    res["tp_world_size"] = parse_int(tokens[15])
    res["y_shard_type"] = parse_int(tokens[16])
    res["transpose_weight"] = parse_bool(tokens[17])

    res["hasExpectTilingKey"] = parse_bool(tokens[18])
    res["expectTilingKey"] = parse_int(tokens[19])
    return res


def parse_case_allto_all_all_gather_bmm(case_str: str) -> Dict[str, Any]:
    """
    解析 AlltoAllAllGatherBmmTilingTestParam 用例。
    字段顺序：
    case_name, soc_version, coreNum, ubSize, tilingDataSize,
    input_shapes (vector<vector<int64_t>>),
    input_dtypes (vector<ge::DataType>),
    output_shape (vector<int64_t>),
    output_dtype,
    group_ep, group_tp, ep_world_size, tp_world_size,
    x_shard_type, act_type,
    transpose_weight, output_y2_flag, output_y3_flag,
    has_expect_tiling_key, expect_tiling_key
    """
    tokens = split_top_level_commas(case_str)
    # 结构体定义包含 20 个字段，如果实际更多，则忽略多余字段（通常是尾随注释等造成）。
    if len(tokens) < 20:
        raise ValueError(
            f"AlltoAllAllGatherBmm 用例字段数量不匹配，实际 {len(tokens)} 个：{case_str}"
        )

    res: Dict[str, Any] = {}

    res["case_name"] = parse_cpp_string_literal(tokens[0])
    res["soc_version"] = parse_cpp_string_literal(tokens[1])
    res["coreNum"] = parse_int(tokens[2])
    res["ubSize"] = parse_int(tokens[3])
    res["tilingDataSize"] = parse_int(tokens[4])

    res["input_shapes"] = parse_shape_2d(tokens[5])
    res["input_dtypes"] = parse_dtype_vec(tokens[6])
    res["output_shape"] = parse_int_vec(tokens[7])
    res["output_dtype"] = tokens[8].strip()

    res["group_ep"] = parse_cpp_string_literal(tokens[9])
    res["group_tp"] = parse_cpp_string_literal(tokens[10])
    res["ep_world_size"] = parse_int(tokens[11])
    res["tp_world_size"] = parse_int(tokens[12])
    res["x_shard_type"] = parse_int(tokens[13])
    res["act_type"] = parse_int(tokens[14])
    res["transpose_weight"] = parse_bool(tokens[15])
    res["output_y2_flag"] = parse_bool(tokens[16])
    res["output_y3_flag"] = parse_bool(tokens[17])
    res["has_expect_tiling_key"] = parse_bool(tokens[18])
    res["expect_tiling_key"] = parse_int(tokens[19])
    return res


def parse_case_moe_distribute_dispatch(case_str: str) -> Dict[str, Any]:
    """
    解析 MoeDistributeDispatchTilingTestParam 用例。
    字段按照结构体定义顺序一一对应。
    """
    tokens = split_top_level_commas(case_str)
    # 结构体当前共有 38 个字段，若未来新增字段则只要前 38 个顺序不变即可复用。
    if len(tokens) < 38:
        raise ValueError(
            f"MoeDistributeDispatch 用例字段数量不匹配，实际 {len(tokens)} 个：{case_str}"
        )

    res: Dict[str, Any] = {}

    idx = 0
    res["inputTotalNum"] = parse_int(tokens[idx]); idx += 1
    res["case_name"] = parse_cpp_string_literal(tokens[idx]); idx += 1
    res["soc_version"] = parse_cpp_string_literal(tokens[idx]); idx += 1
    res["coreNum"] = parse_int(tokens[idx]); idx += 1
    res["ubSize"] = parse_int(tokens[idx]); idx += 1

    # 输入 / 输出 shape
    res["input0_shape"] = parse_shape(tokens[idx]); idx += 1
    res["input1_shape"] = parse_shape(tokens[idx]); idx += 1
    res["input2_shape"] = parse_shape(tokens[idx]); idx += 1

    res["output0_shape"] = parse_shape(tokens[idx]); idx += 1
    res["output1_shape"] = parse_shape(tokens[idx]); idx += 1
    res["output2_shape"] = parse_shape(tokens[idx]); idx += 1
    res["output3_shape"] = parse_shape(tokens[idx]); idx += 1
    res["output4_shape"] = parse_shape(tokens[idx]); idx += 1
    res["output5_shape"] = parse_shape(tokens[idx]); idx += 1

    # dtype
    res["input0_dtype"] = tokens[idx].strip(); idx += 1
    res["input1_dtype"] = tokens[idx].strip(); idx += 1
    res["input2_dtype"] = tokens[idx].strip(); idx += 1

    res["output0_dtype"] = tokens[idx].strip(); idx += 1
    res["output1_dtype"] = tokens[idx].strip(); idx += 1
    res["output2_dtype"] = tokens[idx].strip(); idx += 1
    res["output3_dtype"] = tokens[idx].strip(); idx += 1
    res["output4_dtype"] = tokens[idx].strip(); idx += 1
    res["output5_dtype"] = tokens[idx].strip(); idx += 1

    # attrs
    res["ep_group"] = parse_cpp_string_literal(tokens[idx]); idx += 1
    res["tp_group"] = parse_cpp_string_literal(tokens[idx]); idx += 1
    res["ep_world_size"] = parse_int(tokens[idx]); idx += 1
    res["tp_world_size"] = parse_int(tokens[idx]); idx += 1
    res["ep_rank_id"] = parse_int(tokens[idx]); idx += 1
    res["tp_rank_id"] = parse_int(tokens[idx]); idx += 1
    res["expert_shard_type"] = parse_int(tokens[idx]); idx += 1
    res["shared_expert_num"] = parse_int(tokens[idx]); idx += 1
    res["shared_expert_rank_num"] = parse_int(tokens[idx]); idx += 1
    res["moe_expert_num"] = parse_int(tokens[idx]); idx += 1
    res["quant_mode"] = parse_int(tokens[idx]); idx += 1
    res["global_bs"] = parse_int(tokens[idx]); idx += 1
    res["expert_token_nums_type"] = parse_int(tokens[idx]); idx += 1

    res["has_expect_tiling_key"] = parse_bool(tokens[idx]); idx += 1
    res["expect_tiling_key"] = parse_int(tokens[idx]); idx += 1
    return res


def parse_case_moe_distribute_dispatch_v2(case_str: str) -> Dict[str, Any]:
    """
    解析 MoeDistributeDispatchV2TilingTestParam 用例。
    字段按照结构体定义顺序一一对应。
    """
    tokens = split_top_level_commas(case_str)
    # 结构体当前共有 52 个字段，若未来新增字段则只要前 52 个顺序不变即可复用。
    if len(tokens) < 52:
        raise ValueError(
            f"MoeDistributeDispatchV2 用例字段数量不匹配，实际 {len(tokens)} 个：{case_str}"
        )

    res: Dict[str, Any] = {}
    idx = 0

    res["inputTotalNum"] = parse_int(tokens[idx]); idx += 1
    res["outputTotalNum"] = parse_int(tokens[idx]); idx += 1
    res["case_name"] = parse_cpp_string_literal(tokens[idx]); idx += 1
    res["soc_version"] = parse_cpp_string_literal(tokens[idx]); idx += 1
    res["coreNum"] = parse_int(tokens[idx]); idx += 1
    res["ubSize"] = parse_int(tokens[idx]); idx += 1

    # 输入 shape（最多 6 个）
    for name in [
        "input0_shape",
        "input1_shape",
        "input2_shape",
        "input3_shape",
        "input4_shape",
        "input5_shape",
    ]:
        res[name] = parse_shape(tokens[idx]); idx += 1

    # 输出 shape（最多 7 个）
    for name in [
        "output0_shape",
        "output1_shape",
        "output2_shape",
        "output3_shape",
        "output4_shape",
        "output5_shape",
        "output6_shape",
    ]:
        res[name] = parse_shape(tokens[idx]); idx += 1

    # 输入 dtype 6 个
    for name in [
        "input0_dtype",
        "input1_dtype",
        "input2_dtype",
        "input3_dtype",
        "input4_dtype",
        "input5_dtype",
    ]:
        res[name] = tokens[idx].strip()
        idx += 1

    # 输出 dtype 7 个
    for name in [
        "output0_dtype",
        "output1_dtype",
        "output2_dtype",
        "output3_dtype",
        "output4_dtype",
        "output5_dtype",
        "output6_dtype",
    ]:
        res[name] = tokens[idx].strip()
        idx += 1

    # attrs
    res["ep_group"] = parse_cpp_string_literal(tokens[idx]); idx += 1
    res["tp_group"] = parse_cpp_string_literal(tokens[idx]); idx += 1
    res["ep_world_size"] = parse_int(tokens[idx]); idx += 1
    res["tp_world_size"] = parse_int(tokens[idx]); idx += 1
    res["ep_rank_id"] = parse_int(tokens[idx]); idx += 1
    res["tp_rank_id"] = parse_int(tokens[idx]); idx += 1
    res["expert_shard_type"] = parse_int(tokens[idx]); idx += 1
    res["shared_expert_num"] = parse_int(tokens[idx]); idx += 1
    res["shared_expert_rank_num"] = parse_int(tokens[idx]); idx += 1
    res["moe_expert_num"] = parse_int(tokens[idx]); idx += 1
    res["quant_mode"] = parse_int(tokens[idx]); idx += 1
    res["global_bs"] = parse_int(tokens[idx]); idx += 1
    res["expert_token_nums_type"] = parse_int(tokens[idx]); idx += 1
    res["comm_alg"] = ast.literal_eval(tokens[idx].strip()); idx += 1
    res["zero_expert_num"] = parse_int(tokens[idx]); idx += 1
    res["copy_expert_num"] = parse_int(tokens[idx]); idx += 1
    res["const_expert_num"] = parse_int(tokens[idx]); idx += 1

    res["has_expect_tiling_key"] = parse_bool(tokens[idx]); idx += 1
    res["expect_tiling_key"] = parse_int(tokens[idx]); idx += 1
    return res


def parse_case_moe_distribute_combine(case_str: str) -> Dict[str, Any]:
    """
    解析 MoeDistributeCombineTilingTestParam 用例。
    """
    tokens = split_top_level_commas(case_str)
    # 结构体当前共有 35 个字段，若未来新增字段则只要前 35 个顺序不变即可复用。
    if len(tokens) < 35:
        raise ValueError(
            f"MoeDistributeCombine 用例字段数量不匹配，实际 {len(tokens)} 个：{case_str}"
        )

    res: Dict[str, Any] = {}
    idx = 0

    res["case_name"] = parse_cpp_string_literal(tokens[idx]); idx += 1
    res["soc_version"] = parse_cpp_string_literal(tokens[idx]); idx += 1
    res["coreNum"] = parse_int(tokens[idx]); idx += 1
    res["ubSize"] = parse_int(tokens[idx]); idx += 1

    # 输入 shape 6 个
    for name in [
        "input0_shape",
        "input1_shape",
        "input2_shape",
        "input3_shape",
        "input4_shape",
        "input5_shape",
    ]:
        res[name] = parse_shape(tokens[idx]); idx += 1

    # 输出 shape
    res["output_shape"] = parse_shape(tokens[idx]); idx += 1

    # 输入 dtype 6 个
    for name in [
        "input0_dtype",
        "input1_dtype",
        "input2_dtype",
        "input3_dtype",
        "input4_dtype",
        "input5_dtype",
    ]:
        res[name] = tokens[idx].strip()
        idx += 1

    # 输出 dtype
    res["output_dtype"] = tokens[idx].strip(); idx += 1

    # attrs
    res["ep_group"] = parse_cpp_string_literal(tokens[idx]); idx += 1
    res["tp_group"] = parse_cpp_string_literal(tokens[idx]); idx += 1
    res["ep_world_size"] = parse_int(tokens[idx]); idx += 1
    res["tp_world_size"] = parse_int(tokens[idx]); idx += 1
    res["ep_rank_id"] = parse_int(tokens[idx]); idx += 1
    res["tp_rank_id"] = parse_int(tokens[idx]); idx += 1
    res["expert_shard_type"] = parse_int(tokens[idx]); idx += 1
    res["shared_expert_num"] = parse_int(tokens[idx]); idx += 1
    res["shared_expert_rank_num"] = parse_int(tokens[idx]); idx += 1
    res["moe_expert_num"] = parse_int(tokens[idx]); idx += 1
    res["global_bs"] = parse_int(tokens[idx]); idx += 1
    res["out_dtype"] = parse_int(tokens[idx]); idx += 1
    res["comm_quant_mode"] = parse_int(tokens[idx]); idx += 1
    res["group_list_type"] = parse_int(tokens[idx]); idx += 1

    res["has_expect_tiling_key"] = parse_bool(tokens[idx]); idx += 1
    res["expect_tiling_key"] = parse_int(tokens[idx]); idx += 1
    return res


def parse_case_moe_distribute_combine_add_rms_norm(case_str: str) -> Dict[str, Any]:
    """
    解析 MoeDistributeCombineAddRmsNormTilingTestParam 用例。
    该结构体只暴露 case_name / tiling_params_str_pair / tiling_dTypes_pair / status，
    其中后两者直接以原始字符串形式保留。
    """
    tokens = split_top_level_commas(case_str)
    if len(tokens) != 4:
        raise ValueError(
            f"MoeDistributeCombineAddRmsNorm 用例字段数量不匹配，实际 {len(tokens)} 个：{case_str}"
        )

    res: Dict[str, Any] = {}
    res["case_name"] = parse_cpp_string_literal(tokens[0])
    res["tiling_params_str_pair"] = tokens[1].strip()
    res["tiling_dTypes_pair"] = tokens[2].strip()
    res["status"] = tokens[3].strip()
    return res


def parse_tensor_description_v2(token: str) -> List[Dict[str, Any]]:
    """
    解析 TensorDescription 列表，格式：
    {
        {{{64, 7168}, {64, 7168}}, ge::DT_FLOAT16, ge::FORMAT_ND},
        ...
    }
    返回 [{"storage_shape": [64, 7168], "origin_shape": [64, 7168], "dtype": "ge::DT_FLOAT16", "format": "ge::FORMAT_ND"}, ...]
    """
    token = token.strip()
    if token == "{}":
        return []
    if not (token.startswith("{") and token.endswith("}")):
        raise ValueError(f"无法解析 TensorDescription 列表: {token}")

    inner = token[1:-1].strip()
    if not inner:
        return []

    result: List[Dict[str, Any]] = []
    # 按最外层花括号切分每个 TensorDescription
    depth = 0
    start = None
    for i, ch in enumerate(inner):
        if ch == "{":
            if depth == 0:
                start = i
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0 and start is not None:
                desc_str = inner[start:i + 1].strip()
                # 解析单个 TensorDescription: {{{storage}, {origin}}, dtype, format}
                # 或者简化形式: {{}, dtype, format}
                desc = parse_single_tensor_description(desc_str)
                if desc:
                    result.append(desc)
                start = None

    return result


def parse_single_tensor_description(desc_str: str) -> Optional[Dict[str, Any]]:
    """
    解析单个 TensorDescription：
    - {{{64, 7168}, {64, 7168}}, ge::DT_FLOAT16, ge::FORMAT_ND}
    - {{}, ge::DT_INT32, ge::FORMAT_ND}  (空 shape)
    """
    parts = split_top_level_commas(desc_str)
    if len(parts) < 3:
        return None

    res: Dict[str, Any] = {}
    shape_pair = parts[0].strip()
    res["dtype"] = parts[1].strip()
    res["format"] = parts[2].strip()

    # 解析 shape_pair: {{64, 7168}, {64, 7168}} 或 {}
    if shape_pair == "{}":
        res["storage_shape"] = []
        res["origin_shape"] = []
    else:
        # 找到两个内部 shape
        inner_shapes = []
        depth = 0
        start = None
        for i, ch in enumerate(shape_pair):
            if ch == "{":
                if depth == 1:
                    start = i
                depth += 1
            elif ch == "}":
                depth -= 1
                if depth == 1 and start is not None:
                    inner_shapes.append(shape_pair[start:i + 1])
                    start = None

        if len(inner_shapes) >= 2:
            res["storage_shape"] = parse_shape_with_expr(inner_shapes[0])
            res["origin_shape"] = parse_shape_with_expr(inner_shapes[1])
        elif len(inner_shapes) == 1:
            res["storage_shape"] = parse_shape_with_expr(inner_shapes[0])
            res["origin_shape"] = res["storage_shape"]
        else:
            res["storage_shape"] = []
            res["origin_shape"] = []

    return res


def parse_shape_with_expr(token: str) -> List[int]:
    """
    解析 shape，支持表达式如 {8 * 8 * 32, 7168}
    """
    token = token.strip()
    if token == "{}":
        return []
    m = re.match(r"\{(.*)\}", token)
    if not m:
        return []
    inner = m.group(1).strip()
    if not inner:
        return []
    return [parse_int(x) for x in inner.split(",") if x.strip()]


def parse_op_attr_list(token: str) -> List[Dict[str, Any]]:
    """
    解析 OpAttr 列表，格式：
    {
        {"group_ep", build_from<std::string>("ep_group")},
        {"ep_world_size", build_from<int64_t>(8)},
        ...
    }
    返回 [{"name": "group_ep", "type": "string", "value": "ep_group"}, ...]
    """
    token = token.strip()
    if token == "{}":
        return []
    if not (token.startswith("{") and token.endswith("}")):
        return []

    inner = token[1:-1].strip()
    if not inner:
        return []

    result: List[Dict[str, Any]] = []
    # 按最外层花括号切分每个 OpAttr
    depth = 0
    start = None
    in_angle = 0
    for i, ch in enumerate(inner):
        if ch == "<":
            in_angle += 1
        elif ch == ">":
            in_angle -= 1
        elif ch == "{" and in_angle == 0:
            if depth == 0:
                start = i
            depth += 1
        elif ch == "}" and in_angle == 0:
            depth -= 1
            if depth == 0 and start is not None:
                attr_str = inner[start:i + 1].strip()
                attr = parse_single_op_attr(attr_str)
                if attr:
                    result.append(attr)
                start = None

    return result


def parse_single_op_attr(attr_str: str) -> Optional[Dict[str, Any]]:
    """
    解析单个 OpAttr：
    - {"group_ep", build_from<std::string>("ep_group")}
    - {"ep_world_size", build_from<int64_t>(8)}
    """
    # 去掉外层花括号
    attr_str = attr_str.strip()
    if attr_str.startswith("{") and attr_str.endswith("}"):
        attr_str = attr_str[1:-1].strip()

    # 找到第一个逗号分隔 name 和 value
    first_comma = -1
    in_string = False
    for i, ch in enumerate(attr_str):
        if ch == '"' and (i == 0 or attr_str[i-1] != '\\'):
            in_string = not in_string
        elif ch == ',' and not in_string:
            first_comma = i
            break

    if first_comma == -1:
        return None

    name_part = attr_str[:first_comma].strip()
    value_part = attr_str[first_comma + 1:].strip()

    # 解析 name
    try:
        name = parse_cpp_string_literal(name_part)
    except Exception:
        return None

    res: Dict[str, Any] = {"name": name}

    # 解析 value: build_from<type>(value)
    build_from_match = re.match(r'build_from<([^>]+)>\((.+)\)$', value_part)
    if build_from_match:
        type_str = build_from_match.group(1).strip()
        val_str = build_from_match.group(2).strip()

        if type_str == "std::string":
            res["type"] = "string"
            try:
                res["value"] = parse_cpp_string_literal(val_str)
            except Exception:
                res["value"] = val_str
        elif type_str == "int64_t":
            res["type"] = "int64"
            res["value"] = parse_int(val_str)
        elif type_str == "bool":
            res["type"] = "bool"
            res["value"] = parse_bool(val_str)
        else:
            res["type"] = type_str
            res["value"] = val_str
    else:
        res["value"] = value_part

    return res


def parse_case_moe_distribute_combine_v2(case_str: str) -> Dict[str, Any]:
    """
    解析 MoeDistributeCombineV2TilingTestParam 用例。
    inputs / outputs / attrs 进行结构化解析。
    """
    tokens = split_top_level_commas(case_str)
    if len(tokens) != 9:
        raise ValueError(
            f"MoeDistributeCombineV2 用例字段数量不匹配，实际 {len(tokens)} 个：{case_str}"
        )

    res: Dict[str, Any] = {}
    res["case_name"] = parse_cpp_string_literal(tokens[0])
    res["soc_version"] = parse_cpp_string_literal(tokens[1])
    res["coreNum"] = parse_int(tokens[2])
    res["ubSize"] = parse_int(tokens[3])
    res["inputs"] = parse_tensor_description_v2(tokens[4])
    res["outputs"] = parse_tensor_description_v2(tokens[5])
    res["attrs"] = parse_op_attr_list(tokens[6])
    res["hasExpectTilingKey"] = parse_bool(tokens[7])
    res["expectTilingKey"] = parse_int(tokens[8])
    return res


def parse_tensor_desc_param_list(token: str) -> List[Dict[str, Any]]:
    """
    解析 TensorDescParam 列表，例如：
    {
        {{8192, 1536}, ge::DT_FLOAT16},
        {{1536, 12288}, ge::DT_FLOAT16},
    }
    返回 [{"shape": [8192, 1536], "dtype": "ge::DT_FLOAT16"}, ...]
    """
    token = token.strip()
    if token == "{}":
        return []
    if not (token.startswith("{") and token.endswith("}")):
        raise ValueError(f"无法解析 TensorDescParam 列表: {token}")

    inner = token[1:-1].strip()
    if not inner:
        return []

    result: List[Dict[str, Any]] = []
    # 按最外层花括号切分每个 TensorDescParam
    depth = 0
    start = None
    for i, ch in enumerate(inner):
        if ch == "{":
            if depth == 0:
                start = i
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0 and start is not None:
                param_str = inner[start:i + 1].strip()
                # 解析单个 TensorDescParam: {{shape}, dtype}
                parts = split_top_level_commas(param_str)
                if len(parts) >= 2:
                    shape = parse_shape(parts[0])
                    dtype = parts[1].strip()
                    result.append({"shape": shape, "dtype": dtype})
                start = None

    return result


def parse_case_matmul_reduce_scatter_v2_new(case_str: str) -> Dict[str, Any]:
    """
    解析新格式的 MatmulReduceScatterV2TilingTestParam 用例。
    字段顺序：
    case_name, inputs (TensorDescParam列表), outputs (TensorDescParam列表),
    is_trans_a, is_trans_b, rank_size, group_size, is_amax_out, y_dtype_attr,
    comm_turn, group, reduce_op, comm_mode, core_num, mock_rank_num,
    check_tiling_key, expect_tiling_key
    """
    tokens = split_top_level_commas(case_str)
    if len(tokens) != 17:
        raise ValueError(
            f"MatmulReduceScatterV2(新格式) 用例字段数量不匹配，实际 {len(tokens)} 个：{case_str}"
        )

    res: Dict[str, Any] = {}

    res["case_name"] = parse_cpp_string_literal(tokens[0])
    res["inputs"] = parse_tensor_desc_param_list(tokens[1])
    res["outputs"] = parse_tensor_desc_param_list(tokens[2])
    res["is_trans_a"] = parse_bool(tokens[3])
    res["is_trans_b"] = parse_bool(tokens[4])
    res["rank_size"] = parse_int(tokens[5])
    res["group_size"] = parse_int(tokens[6])
    res["is_amax_out"] = parse_bool(tokens[7])
    res["y_dtype_attr"] = parse_int(tokens[8])
    res["comm_turn"] = parse_int(tokens[9])
    res["group"] = parse_cpp_string_literal(tokens[10])
    res["reduce_op"] = parse_cpp_string_literal(tokens[11])
    res["comm_mode"] = parse_cpp_string_literal(tokens[12])
    res["core_num"] = parse_int(tokens[13])
    res["mock_rank_num"] = parse_int(tokens[14])
    res["check_tiling_key"] = parse_bool(tokens[15])
    res["expect_tiling_key"] = parse_int(tokens[16])

    return res


def parse_kv_pair_list(token: str) -> List[Dict[str, str]]:
    """
    解析 key-value pair 列表，例如：
    {{"e", "64"}, {"permute_out_flag", "true"}}
    返回 [{"key": "e", "value": "64"}, ...]
    """
    token = token.strip()
    if token == "{}":
        return []
    if not (token.startswith("{") and token.endswith("}")):
        raise ValueError(f"无法解析 kv pair 列表: {token}")

    inner = token[1:-1].strip()
    if not inner:
        return []

    result: List[Dict[str, str]] = []
    depth = 0
    start = None
    for i, ch in enumerate(inner):
        if ch == "{":
            if depth == 0:
                start = i
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0 and start is not None:
                pair_str = inner[start:i + 1].strip()
                parts = split_top_level_commas(pair_str)
                if len(parts) >= 2:
                    key = parse_cpp_string_literal(parts[0])
                    value = parse_cpp_string_literal(parts[1])
                    result.append({"key": key, "value": value})
                start = None

    return result


def parse_vec_from_str(value_str: str) -> List[int]:
    """
    解析向量字符串，支持多种格式：
    - std::vector<int64_t>{128, 128, ...}
    - std::vector<int64_t>{128, 128, ...
           128, 128}  (多行)
    - {1, 2, 3}
    """
    value_str = value_str.strip()
    # 去掉 std::vector<int64_t> 前缀
    vec_match = re.match(r"std::vector<[^>]+>\s*\{(.*)\}", value_str, re.DOTALL)
    if vec_match:
        inner = vec_match.group(1).strip()
        # 去掉换行和多余空格，分割成数字
        numbers = [x.strip() for x in inner.split(",") if x.strip()]
        return [parse_int(n) for n in numbers if n]
    elif value_str.startswith("{"):
        return parse_int_vec(value_str)
    return []


def parse_kv_vec_pair_list(token: str) -> List[Dict[str, Any]]:
    """
    解析 key-vector pair 列表，例如：
    {{"send_counts", std::vector<int64_t>{128, 128, ...}}}
    返回 [{"key": "send_counts", "value": [128, 128, ...]}, ...]
    """
    token = token.strip()
    if token == "{}":
        return []
    if not (token.startswith("{") and token.endswith("}")):
        return []

    inner = token[1:-1].strip()
    if not inner:
        return []

    result: List[Dict[str, Any]] = []
    depth = 0
    start = None
    in_angle_bracket = 0
    for i, ch in enumerate(inner):
        if ch == "<":
            in_angle_bracket += 1
        elif ch == ">":
            in_angle_bracket -= 1
        elif ch == "{" and in_angle_bracket == 0:
            if depth == 0:
                start = i
            depth += 1
        elif ch == "}" and in_angle_bracket == 0:
            depth -= 1
            if depth == 0 and start is not None:
                pair_str = inner[start:i + 1].strip()
                parts = split_top_level_commas(pair_str)
                if len(parts) >= 2:
                    key = parse_cpp_string_literal(parts[0])
                    # 重新组合剩余部分（可能跨多个逗号分割的片段）
                    value_str = ", ".join(parts[1:])
                    value = parse_vec_from_str(value_str)
                    result.append({"key": key, "value": value})
                start = None

    return result


def parse_size_dtype_pair_list(token: str) -> List[Dict[str, Any]]:
    """
    解析 size-dtype pair 列表，例如：
    {{0, ge::DT_FLOAT16}, {1, ge::DT_FLOAT}}
    返回 [{"size": 0, "dtype": "ge::DT_FLOAT16"}, ...]
    """
    token = token.strip()
    if token == "{}":
        return []
    if not (token.startswith("{") and token.endswith("}")):
        return []

    inner = token[1:-1].strip()
    if not inner:
        return []

    result: List[Dict[str, Any]] = []
    depth = 0
    start = None
    for i, ch in enumerate(inner):
        if ch == "{":
            if depth == 0:
                start = i
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0 and start is not None:
                pair_str = inner[start:i + 1].strip()
                parts = split_top_level_commas(pair_str)
                if len(parts) >= 2:
                    size = parse_int(parts[0])
                    dtype = parts[1].strip()
                    result.append({"size": size, "dtype": dtype})
                start = None

    return result


def parse_case_allto_allv_grouped_matmul(case_str: str) -> Dict[str, Any]:
    """
    解析 AlltoAllvGroupedMatMul 的 TestParam 用例。
    字段顺序：
    test_name, tiling_params_str_pair, tiling_params_vec_pair, tiling_dTypes_pair, status
    """
    tokens = split_top_level_commas(case_str)
    if len(tokens) != 5:
        raise ValueError(
            f"AlltoAllvGroupedMatMul TestParam 用例字段数量不匹配，实际 {len(tokens)} 个：{case_str}"
        )

    res: Dict[str, Any] = {}

    res["test_name"] = parse_cpp_string_literal(tokens[0])
    res["tiling_params_str_pair"] = parse_kv_pair_list(tokens[1])
    res["tiling_params_vec_pair"] = parse_kv_vec_pair_list(tokens[2])
    res["tiling_dTypes_pair"] = parse_size_dtype_pair_list(tokens[3])
    res["status"] = tokens[4].strip()

    return res


def parse_case_matmul_all_reduce_add_rms_norm(case_str: str) -> Dict[str, Any]:
    """
    解析 MatmulAllReduceAddRmsNorm 的 TestParam 用例。
    字段顺序：
    caseName, blockDim, tilingKey

    caseName 中编码了所有参数，格式如：
    "MODEL0_group_sum_4096_688_4096_1_0_0_0_-1_0_0_1_0_0.1_INT8_INT8_INT32_BF16"
    """
    tokens = split_top_level_commas(case_str)
    if len(tokens) != 3:
        raise ValueError(
            f"MatmulAllReduceAddRmsNorm TestParam 用例字段数量不匹配，实际 {len(tokens)} 个：{case_str}"
        )

    res: Dict[str, Any] = {}

    case_name = parse_cpp_string_literal(tokens[0])
    res["case_name"] = case_name
    res["blockDim"] = parse_int(tokens[1])
    res["tilingKey"] = parse_int(tokens[2])

    # 解析 caseName 中的参数
    parts = case_name.split("_")
    if len(parts) >= 20:
        idx = 0
        res["model_name"] = parts[idx]; idx += 1
        res["group_name"] = parts[idx]; idx += 1
        res["reduce_op"] = parts[idx]; idx += 1
        res["m"] = int(parts[idx]); idx += 1
        res["k"] = int(parts[idx]); idx += 1
        res["n"] = int(parts[idx]); idx += 1
        res["biasFlag"] = int(parts[idx]); idx += 1
        res["x3Flag"] = int(parts[idx]); idx += 1
        res["transA"] = int(parts[idx]); idx += 1
        res["transB"] = int(parts[idx]); idx += 1
        res["group"] = int(parts[idx]); idx += 1
        res["antiquant_offsetExistFlag"] = int(parts[idx]); idx += 1
        res["antiquant_scaleExistFlag"] = int(parts[idx]); idx += 1
        res["dequant_scaleExistFlag"] = int(parts[idx]); idx += 1
        res["antigroupSize"] = int(parts[idx]); idx += 1
        res["epsilon"] = float(parts[idx]); idx += 1
        res["xDtype"] = parts[idx]; idx += 1
        res["weightDtype"] = parts[idx]; idx += 1
        res["biasDtype"] = parts[idx]; idx += 1
        res["yDtype"] = parts[idx]; idx += 1

    return res


def convert(src_path: str, dst_path: str,
            default_compile_info: str = DEFAULT_COMPILE_INFO,
            default_soc_version: str = DEFAULT_SOC_VERSION,
            default_core_num: int = DEFAULT_CORE_NUM,
            default_ub_size: int = DEFAULT_UB_SIZE,
            default_tiling_data_size: int = DEFAULT_TILING_DATA_SIZE) -> None:
    with open(src_path, "r", encoding="utf-8") as f:
        src = f.read()

    compile_info_value = extract_compile_info(src)
    mode, cases_block = detect_mode_and_cases_block(src)
    case_inits = split_case_initializers(cases_block)

    dst_dir = os.path.dirname(dst_path)
    if dst_dir:
        os.makedirs(dst_dir, exist_ok=True)
    with open(dst_path, "w", encoding="utf-8") as out:
        for case_str in case_inits:
            if mode == "matmul_like":
                obj = parse_case_matmul_like(case_str, compile_info_value)
            elif mode == "matmul_simple":
                obj = parse_case_matmul_simple(
                    case_str, default_compile_info, default_soc_version,
                    default_core_num, default_ub_size, default_tiling_data_size
                )
            elif mode == "distribute_barrier":
                obj = parse_case_distribute_barrier(case_str, compile_info_value)
            elif mode == "matmul_reduce_scatter":
                obj = parse_case_matmul_reduce_scatter(case_str)
            elif mode == "matmul_reduce_scatter_v2":
                obj = parse_case_matmul_reduce_scatter_v2(case_str, compile_info_value)
            elif mode == "grouped_matmul_all_reduce":
                obj = parse_case_grouped_matmul_all_reduce(case_str)
            elif mode == "batch_matmul_reduce_scatter_alltoall":
                obj = parse_case_batch_matmul_reduce_scatter_alltoall(case_str)
            elif mode == "allto_all_all_gather_bmm":
                obj = parse_case_allto_all_all_gather_bmm(case_str)
            elif mode == "moe_distribute_dispatch":
                obj = parse_case_moe_distribute_dispatch(case_str)
            elif mode == "moe_distribute_dispatch_v2":
                obj = parse_case_moe_distribute_dispatch_v2(case_str)
            elif mode == "moe_distribute_combine":
                obj = parse_case_moe_distribute_combine(case_str)
            elif mode == "moe_distribute_combine_add_rms_norm":
                obj = parse_case_moe_distribute_combine_add_rms_norm(case_str)
            elif mode == "moe_distribute_combine_v2":
                obj = parse_case_moe_distribute_combine_v2(case_str)
            elif mode == "matmul_reduce_scatter_v2_new":
                obj = parse_case_matmul_reduce_scatter_v2_new(case_str)
            elif mode == "allto_allv_grouped_matmul":
                obj = parse_case_allto_allv_grouped_matmul(case_str)
            elif mode == "matmul_all_reduce_add_rms_norm":
                obj = parse_case_matmul_all_reduce_add_rms_norm(case_str)
            else:
                raise ValueError(f"不支持的模式: {mode}")

            out.write(json.dumps(obj, ensure_ascii=False))
            out.write("\n")


def get_output_filename(src_filename: str) -> str:
    """
    根据源文件名生成输出文件名。
    例如：test_all_gather_matmul_tiling.cpp -> all_gather_matmul.jsonl
    """
    # 去掉 test_ 前缀和 _tiling.cpp 后缀
    name = src_filename
    if name.startswith("test_"):
        name = name[5:]
    if name.endswith("_tiling.cpp"):
        name = name[:-11]
    elif name.endswith(".cpp"):
        name = name[:-4]
    return name + ".jsonl"


def batch_convert(src_dir: str, dst_dir: str,
                  default_compile_info: str = DEFAULT_COMPILE_INFO,
                  default_soc_version: str = DEFAULT_SOC_VERSION,
                  default_core_num: int = DEFAULT_CORE_NUM,
                  default_ub_size: int = DEFAULT_UB_SIZE,
                  default_tiling_data_size: int = DEFAULT_TILING_DATA_SIZE) -> None:
    """
    批量转换 src_dir 目录下的所有 .cpp 文件到 dst_dir 目录。
    """
    os.makedirs(dst_dir, exist_ok=True)
    cpp_files = [f for f in os.listdir(src_dir) if f.endswith(".cpp")]

    for cpp_file in sorted(cpp_files):
        src_path = os.path.join(src_dir, cpp_file)
        out_name = get_output_filename(cpp_file)
        dst_path = os.path.join(dst_dir, out_name)

        try:
            convert(src_path, dst_path, default_compile_info, default_soc_version,
                    default_core_num, default_ub_size, default_tiling_data_size)
            print(f"✓ 已转换: {cpp_file} -> {out_name}")
        except Exception as e:
            print(f"✗ 转换失败: {cpp_file}: {e}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "从 C++ UT 文件中抽取 cases_params 用例并生成 JSONL 文件，"
            "当前支持：AllGatherMatmul / MatmulAllReduce / DistributeBarrier 及多种特殊并行 MatMul / MoE / AlltoAll 等 Tiling UT"
        )
    )
    parser.add_argument(
        "--src",
        default=None,
        help="源 C++ 测试文件路径（单文件模式）",
    )
    parser.add_argument(
        "--dst",
        default=None,
        help="输出 JSONL 文件路径（单文件模式）",
    )
    parser.add_argument(
        "--batch",
        action="store_true",
        help="批量转换模式：转换 --src-dir 中的所有 .cpp 文件",
    )
    parser.add_argument(
        "--src-dir",
        default="target",
        help="批量模式的源目录（默认为 target）",
    )
    parser.add_argument(
        "--dst-dir",
        default="input",
        help="批量模式的输出目录（默认为 input）",
    )
    parser.add_argument(
        "--soc-version",
        default=DEFAULT_SOC_VERSION,
        help=f"默认 SoC 版本（用于简化版结构体，默认为 {DEFAULT_SOC_VERSION}）",
    )
    parser.add_argument(
        "--core-num",
        type=int,
        default=DEFAULT_CORE_NUM,
        help=f"默认核数（用于简化版结构体，默认为 {DEFAULT_CORE_NUM}）",
    )
    parser.add_argument(
        "--ub-size",
        type=int,
        default=DEFAULT_UB_SIZE,
        help=f"默认 UB 大小（用于简化版结构体，默认为 {DEFAULT_UB_SIZE}）",
    )
    parser.add_argument(
        "--tiling-data-size",
        type=int,
        default=DEFAULT_TILING_DATA_SIZE,
        help=f"默认 tiling data 大小（用于简化版结构体，默认为 {DEFAULT_TILING_DATA_SIZE}）",
    )
    args = parser.parse_args()

    # 根据参数中的值动态生成 compile_info
    compile_info = json.dumps({
        "hardware_info": {
            "BT_SIZE": 0,
            "load3d_constraints": "1",
            "Intrinsic_fix_pipe_l0c2out": False,
            "Intrinsic_data_move_l12ub": True,
            "Intrinsic_data_move_l0c2ub": True,
            "Intrinsic_data_move_out2l1_nd2nz": False,
            "UB_SIZE": args.ub_size,
            "L2_SIZE": 33554432,
            "L1_SIZE": 524288,
            "L0A_SIZE": 65536,
            "L0B_SIZE": 65536,
            "L0C_SIZE": 131072,
            "CORE_NUM": args.core_num,
            "socVersion": args.soc_version
        }
    })

    if args.batch:
        # 批量转换模式
        batch_convert(
            args.src_dir, args.dst_dir,
            compile_info, args.soc_version,
            args.core_num, args.ub_size, args.tiling_data_size
        )
    else:
        # 单文件模式
        src_path = args.src or "target/test_all_gather_matmul_tiling.cpp"
        dst_path = args.dst or "input/all_gather_matmul.jsonl"
        convert(
            src_path, dst_path,
            compile_info, args.soc_version,
            args.core_num, args.ub_size, args.tiling_data_size
        )
        print(f"✓ 已转换: {src_path} -> {dst_path}")


if __name__ == "__main__":
    main()


