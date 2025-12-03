"""
从 JSONL 输入文件和模板文件生成完整的 C++ 单元测试文件。

支持的算子类型：
- all_gather_matmul (多行格式)
- matmul_all_reduce (单行格式)
- distribute_barrier (2行格式)
"""
import json
import re
from pathlib import Path
from typing import Any, Dict, List, Tuple

from state import WorkflowState


# AllGatherMatmul / MatmulAllReduce 使用的字段顺序
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

# DistributeBarrier 使用的字段顺序
DISTRIBUTE_BARRIER_FIELDS = [
    "case_name",
    "m",
    "n",
    "dtype",
    "group",
    "world_size",
    "soc_version",
    "coreNum",
    "ubSize",
    "expectTilingKey",
    "expectTilingData",
    "expectWorkspaces",
    "mc2TilingDataReservedLen",
]

# 需要用十六进制表示的大数值阈值 
# 268435455 (0xFFFFFFF) 使用十六进制，16777216 保持十进制
HEX_THRESHOLD = 100000000  # 大于 1亿 的数用十六进制


def detect_mode(template_content: str, operator_name: str) -> str:
    """根据模板内容和算子名检测模式"""
    if "AllGatherMatmulTilingTestParam" in template_content:
        return "all_gather_matmul"
    elif "MatmulAllReduceTilingTestParam" in template_content:
        return "matmul_all_reduce"
    elif "DistributeBarrierTilingTestParam" in template_content:
        return "distribute_barrier"
    return operator_name


def get_struct_name(mode: str) -> str:
    """获取结构体名称"""
    mapping = {
        "all_gather_matmul": "AllGatherMatmulTilingTestParam",
        "matmul_all_reduce": "MatmulAllReduceTilingTestParam",
        "distribute_barrier": "DistributeBarrierTilingTestParam",
    }
    return mapping.get(mode, "")


def read_jsonl(file_path: Path) -> List[Dict[str, Any]]:
    """读取 JSONL 文件，支持格式化的 JSON（每行不一定是一个完整对象）"""
    cases = []
    content = file_path.read_text(encoding="utf-8")
    
    decoder = json.JSONDecoder()
    pos = 0
    while pos < len(content):
        content_stripped = content[pos:].lstrip()
        if not content_stripped:
            break
        try:
            obj, idx = decoder.raw_decode(content_stripped)
            cases.append(obj)
            pos = len(content) - len(content_stripped) + idx
        except json.JSONDecodeError:
            break
    
    return cases


def format_int(value: int, key: str = "") -> str:
    """格式化整数，对于 expectTilingKey 加 UL 后缀，对于大数使用十六进制"""
    if key == "expectTilingKey":
        return f"{value}UL"
    # 对于超大数使用十六进制格式 (用于 268435455 等)
    if value > HEX_THRESHOLD:
        return f"0x{value:X}"
    return str(value)


def convert_value_to_cpp(key: str, value: Any, use_compile_info: bool = False) -> str:
    """将 JSON 值转换为 C++ 格式"""
    if isinstance(value, bool):
        return "true" if value else "false"
    elif isinstance(value, str):
        if value.startswith("ge::"):
            return value
        if key in ("compile_info", "expectTilingData") and use_compile_info:
            return "COMPILE_INFO"
        return f'"{value}"'
    elif isinstance(value, list):
        if len(value) == 0:
            return "{}"
        items = ", ".join(format_int(v) if isinstance(v, int) else str(v) for v in value)
        return f"{{{items}}}"
    elif isinstance(value, int):
        return format_int(value, key)
    elif isinstance(value, float):
        return str(value)
    else:
        return str(value)


def find_most_common_value(cases: List[Dict[str, Any]], key: str) -> Tuple[str, int]:
    """找出最常见的值及其出现次数"""
    from collections import Counter
    values = [case.get(key, "") for case in cases if key in case]
    if not values:
        return "", 0
    counter = Counter(values)
    most_common = counter.most_common(1)[0]
    return most_common[0], most_common[1]


def generate_compile_info_const(mode: str, cases: List[Dict[str, Any]]) -> Tuple[str, str]:
    """生成 COMPILE_INFO 常量定义，返回 (常量定义代码, 常量值)"""
    if mode in ("all_gather_matmul", "matmul_all_reduce"):
        key = "compile_info"
    else:
        key = "expectTilingData"
    
    common_value, count = find_most_common_value(cases, key)
    
    if not common_value or count < 2:
        return "", ""
    
    if mode == "all_gather_matmul":
        const_def = f'const std::string COMPILE_INFO = R"({common_value})";'
    elif mode == "matmul_all_reduce":
        # matmul_all_reduce 使用 string 而不是 std::string
        const_def = f'const string COMPILE_INFO = R"({common_value})";'
    else:
        const_def = f'const std::string COMPILE_INFO = "{common_value}";'
    
    return const_def, common_value


def find_insert_position(template_content: str) -> int:
    """找到插入测试数据的位置（在 TEST_P 之前）"""
    match = re.search(r'\nTEST_P\(', template_content)
    if match:
        return match.start()
    return len(template_content)


# ============== all_gather_matmul 多行格式 ==============
def generate_all_gather_matmul_case(case: Dict[str, Any], common_value: str) -> str:
    """生成 AllGatherMatmul 用例 (多行格式)"""
    use_compile_info = (case.get("compile_info", "") == common_value)
    
    # 第一行: 平台信息
    line1_fields = ["inputTotalNum", "case_name", "compile_info", "soc_version", 
                    "coreNum", "ubSize", "tilingDataSize"]
    line1_vals = [convert_value_to_cpp(f, case.get(f, 0), use_compile_info) for f in line1_fields]
    line1 = "    {" + ", ".join(line1_vals) + ","
    
    # 第二行: shape 信息
    shape_fields = ["x1_shape", "x2_shape", "bias_shape", "x3_shape",
                    "antiquant_scale_shape", "antiquant_offset_shape",
                    "dequant_scale_shape", "pertoken_scale_shape",
                    "comm_quant_scale_1_shape", "comm_quant_scale_2_shape"]
    shape_vals = [convert_value_to_cpp(f, case.get(f, []), use_compile_info) for f in shape_fields]
    line2 = "        " + ", ".join(shape_vals) + ","
    
    # 第三行: output_shape + 前4个dtype
    line3_fields = ["output_shape", "x1_dtype", "x2_dtype", "bias_dtype", "x3_dtype"]
    line3_vals = [convert_value_to_cpp(f, case.get(f, "ge::DT_FLOAT16" if "dtype" in f else []), use_compile_info) for f in line3_fields]
    line3 = "        " + ", ".join(line3_vals) + ","
    
    # 第四行: 中间的6个dtype
    line4_fields = ["antiquant_scale_dtype", "antiquant_offset_dtype",
                    "dequant_scale_dtype", "pertoken_scale_dtype",
                    "comm_quant_scale_1_dtype", "comm_quant_scale_2_dtype"]
    line4_vals = [convert_value_to_cpp(f, case.get(f, "ge::DT_FLOAT"), use_compile_info) for f in line4_fields]
    line4 = "        " + ", ".join(line4_vals) + ","
    
    # 第五行: output_dtype + bool + expectTilingKey
    line5_fields = ["output_dtype", "is_trans_a", "is_trans_b", "expectTilingKey"]
    line5_vals = [convert_value_to_cpp(f, case.get(f, False if "trans" in f else 0), use_compile_info) for f in line5_fields]
    line5 = "        " + ", ".join(line5_vals) + "},"
    
    return "\n".join([line1, line2, line3, line4, line5])


# ============== matmul_all_reduce 单行格式 ==============
def generate_matmul_all_reduce_case(case: Dict[str, Any], common_value: str) -> str:
    """生成 MatmulAllReduce 用例 (单行格式)"""
    use_compile_info = (case.get("compile_info", "") == common_value)
    
    values = []
    for field in MATMUL_STRUCT_FIELDS:
        if field in case:
            val = convert_value_to_cpp(field, case[field], use_compile_info)
        else:
            if "shape" in field:
                val = "{}"
            elif "dtype" in field:
                val = "ge::DT_FLOAT16"
            elif "trans" in field:
                val = "false"
            else:
                val = "0"
        values.append(val)
    
    return "{" + ",".join(values) + "},"


# ============== distribute_barrier 2行格式 ==============
def generate_distribute_barrier_case(case: Dict[str, Any], common_value: str) -> str:
    """生成 DistributeBarrier 用例 (2行格式)"""
    use_compile_info = (case.get("expectTilingData", "") == common_value)
    
    # 第一行: case_name到world_size
    line1_fields = ["case_name", "m", "n", "dtype", "group", "world_size"]
    line1_vals = [convert_value_to_cpp(f, case.get(f, ""), use_compile_info) for f in line1_fields]
    line1 = "    {" + ", ".join(line1_vals) + ","
    
    # 第二行: soc_version到mc2TilingDataReservedLen
    line2_fields = ["soc_version", "coreNum", "ubSize", "expectTilingKey",
                    "expectTilingData", "expectWorkspaces", "mc2TilingDataReservedLen"]
    line2_vals = [convert_value_to_cpp(f, case.get(f, [] if f == "expectWorkspaces" else 0), use_compile_info) for f in line2_fields]
    line2 = "        " + ", ".join(line2_vals) + "},"
    
    return "\n".join([line1, line2])


def generate_cases_params(mode: str, cases: List[Dict[str, Any]], 
                          struct_name: str, common_value: str) -> str:
    """生成 cases_params 数组代码"""
    lines = []
    # 只对 matmul 类型添加注释
    if mode in ("all_gather_matmul", "matmul_all_reduce"):
        lines.append("// 用例列表集")
    lines.append(f"{struct_name} cases_params[] = {{")
    
    for i, case in enumerate(cases):
        if mode == "all_gather_matmul":
            case_code = generate_all_gather_matmul_case(case, common_value)
            lines.append(case_code)
            # 每个用例后加空行（最后一个除外）
            if i < len(cases) - 1:
                lines.append("")
        elif mode == "matmul_all_reduce":
            case_code = generate_matmul_all_reduce_case(case, common_value)
            lines.append(case_code)
        else:  # distribute_barrier
            case_code = generate_distribute_barrier_case(case, common_value)
            lines.append(case_code)
    
    lines.append("};")
    return "\n".join(lines)


def generate_unit_test(state: WorkflowState) -> WorkflowState:
    """生成单元测试文件的主函数"""
    template_path = Path(state["template_file_path"])
    input_path = Path(state.get("input_path", ""))
    output_path = Path(state["output_path"])
    operator_name = state.get("operator_name", "")
    
    template_content = template_path.read_text(encoding="utf-8")
    mode = detect_mode(template_content, operator_name)
    struct_name = get_struct_name(mode)
    
    if not input_path or not input_path.exists():
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(template_content, encoding="utf-8")
        print(f"生成完成（无输入数据）: {output_path}")
        state["output_path"] = str(output_path)
        return state
    
    cases = read_jsonl(input_path)
    
    if not cases:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(template_content, encoding="utf-8")
        print(f"生成完成（空数据）: {output_path}")
        state["output_path"] = str(output_path)
        return state
    
    const_def, common_value = generate_compile_info_const(mode, cases)
    cases_code = generate_cases_params(mode, cases, struct_name, common_value)
    
    data_code_parts = []
    if const_def:
        data_code_parts.append(const_def)
        data_code_parts.append("")
    data_code_parts.append(cases_code)
    data_code_parts.append("")
    
    data_code = "\n".join(data_code_parts)
    insert_pos = find_insert_position(template_content)
    output_content = template_content[:insert_pos] + "\n" + data_code + template_content[insert_pos:]
    
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(output_content, encoding="utf-8")
    print(f"生成完成: {output_path}")
    
    state["output_path"] = str(output_path)
    return state
