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


def extract_struct_name(template_content: str) -> str:
    """从模板内容中提取结构体名称"""
    # 首先尝试匹配 XXXTilingTestParam 模式
    match = re.search(r'struct\s+(\w+TilingTestParam)\s*\{', template_content)
    if match:
        return match.group(1)
    # 其次尝试匹配简单的 TestParam 结构体
    match = re.search(r'struct\s+(TestParam)\s*\{', template_content)
    if match:
        return match.group(1)
    return ""


def extract_param_array_names(template_content: str) -> List[str]:
    """
    从模板内容中提取所有测试参数数组名称。
    通过查找 testing::ValuesIn(xxx) 模式来确定。
    """
    # 查找所有 testing::ValuesIn(array_name) 或 ::testing::ValuesIn(array_name)
    matches = re.findall(r'testing::ValuesIn\((\w+)\)', template_content)
    if matches:
        return list(dict.fromkeys(matches))  # 去重保持顺序
    return ["cases_params"]  # 默认值


def extract_param_array_name(template_content: str) -> str:
    """
    从模板内容中提取第一个测试参数数组名称。
    向后兼容接口。
    """
    names = extract_param_array_names(template_content)
    return names[0] if names else "cases_params"


def detect_mode(template_content: str, struct_name: str) -> str:
    """
    根据结构体名称和模板特征检测生成模式。
    返回: 'moe_tensor_desc' | 'all_gather_matmul_v2' | 'all_gather_matmul' | 'matmul_all_reduce' | 'distribute_barrier' | 'allto_allv_complex'
    """
    # 检查是否是使用 vector<TensorDescription> 的复杂模式
    if "std::vector<gert::TilingContextPara::TensorDescription> inputs;" in template_content:
        return "moe_tensor_desc"
    
    # 检查是否是 allto_allv_grouped_mat_mul 的特殊结构 (有 tiling_params_str_pair 字段)
    if "std::vector<std::pair<string, string>> tiling_params_str_pair" in template_content:
        return "allto_allv_complex"
    
    # 优先根据结构体名称判断特殊类型
    if "AllGatherMatmul" in struct_name:
        # 区分 V1 和 V2: V2 有 expectSuccess 字段
        if "bool expectSuccess;" in template_content:
            return "all_gather_matmul_v2"
        return "all_gather_matmul"
    
    # 检查 DistributeBarrier 特征 (含有 m, n 字段)
    if "int64_t m;" in template_content and "int64_t n;" in template_content:
        return "distribute_barrier"
        
    # 默认为通用 Matmul 结构 (mc2 matmul like)
    return "matmul_all_reduce"


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


def clean_ge_type_string(value: str) -> str:
    """清理 ge:: 类型字符串中的注释和换行符"""
    # 移除 "// 未使用" 等注释
    if "//" in value:
        # 提取所有以 ge:: 开头的部分
        import re
        ge_match = re.search(r'ge::\w+', value)
        if ge_match:
            return ge_match.group()
    return value.strip()


def convert_value_to_cpp(key: str, value: Any, use_compile_info: bool = False) -> str:
    """将 JSON 值转换为 C++ 格式"""
    if isinstance(value, bool):
        return "true" if value else "false"
    elif isinstance(value, str):
        # 处理 ge:: 类型字符串
        if "ge::" in value:
            return clean_ge_type_string(value)
        if key in ("compile_info", "expectTilingData") and use_compile_info:
            return "COMPILE_INFO"
        return f'"{value}"'
    elif isinstance(value, list):
        if len(value) == 0:
            return "{}"
        items = ", ".join(convert_value_to_cpp(key, v, use_compile_info) for v in value)
        return f"{{{items}}}"
    elif isinstance(value, int):
        return format_int(value, key)
    elif isinstance(value, float):
        return str(value)
    else:
        return str(value)


# ============== moe_tensor_desc 模式的辅助函数 ==============
def generate_tensor_desc_cpp(tensor: Dict[str, Any]) -> str:
    """
    将单个 TensorDescription 转换为 C++ 格式
    格式: {{{shape}, {shape}}, dtype, format}
    或空 shape: {{}, dtype, format}
    """
    storage_shape = tensor.get("storage_shape", [])
    origin_shape = tensor.get("origin_shape", [])
    dtype = tensor.get("dtype", "ge::DT_FLOAT16")
    fmt = tensor.get("format", "ge::FORMAT_ND")
    
    if not storage_shape and not origin_shape:
        # 空 shape
        return f"{{{{}}, {dtype}, {fmt}}}"
    else:
        storage_str = "{" + ", ".join(str(v) for v in storage_shape) + "}"
        origin_str = "{" + ", ".join(str(v) for v in origin_shape) + "}"
        return f"{{{{{storage_str}, {origin_str}}}, {dtype}, {fmt}}}"


def generate_tensor_desc_list_cpp(tensors: List[Dict[str, Any]], indent: str = "        ") -> str:
    """生成 TensorDescription 列表的 C++ 代码"""
    if not tensors:
        return "{}"
    
    lines = ["{"]
    for i, tensor in enumerate(tensors):
        # 所有元素都加逗号，保持与 target 一致
        lines.append(f"{indent}    {generate_tensor_desc_cpp(tensor)},")
    lines.append(f"{indent}}}")
    return "\n".join(lines)


def generate_op_attr_cpp(attr: Dict[str, Any]) -> str:
    """
    将单个 OpAttr 转换为 C++ 格式
    格式: {"name", build_from<type>(value)}
    """
    name = attr.get("name", "")
    attr_type = attr.get("type", "int64")
    value = attr.get("value", 0)
    
    # 类型映射
    type_map = {
        "string": "std::string",
        "int64": "int64_t",
        "int32": "int32_t",
        "bool": "bool",
        "float": "float",
        "double": "double",
    }
    
    cpp_type = type_map.get(attr_type, "int64_t")
    
    # 格式化值
    if attr_type == "string":
        cpp_value = f'"{value}"'
    elif attr_type == "bool":
        cpp_value = "true" if value else "false"
    else:
        cpp_value = str(value)
    
    return f'{{"{name}", build_from<{cpp_type}>({cpp_value})}}'


def generate_op_attr_list_cpp(attrs: List[Dict[str, Any]], indent: str = "        ") -> str:
    """生成 OpAttr 列表的 C++ 代码"""
    if not attrs:
        return "{}"
    
    lines = ["{"]
    for i, attr in enumerate(attrs):
        suffix = "," if i < len(attrs) - 1 else ""
        lines.append(f"{indent}    {generate_op_attr_cpp(attr)}{suffix}")
    lines.append(f"{indent}}}")
    return "\n".join(lines)


def generate_moe_tensor_desc_case(case: Dict[str, Any]) -> str:
    """
    生成 moe_tensor_desc 模式的测试用例（复杂 TensorDescription 格式）
    """
    lines = []
    
    # 开始
    lines.append("    {")
    
    # 基本字段
    case_name = case.get("case_name", "")
    soc_version = case.get("soc_version", "")
    core_num = case.get("coreNum", 0)
    ub_size = case.get("ubSize", 0)
    
    lines.append(f'        "{case_name}",')
    lines.append(f'        "{soc_version}",')
    lines.append(f'        {core_num},')
    lines.append(f'        {ub_size},')
    
    # inputs
    inputs = case.get("inputs", [])
    inputs_cpp = generate_tensor_desc_list_cpp(inputs, "        ")
    lines.append(f"        {inputs_cpp},")
    
    # outputs
    outputs = case.get("outputs", [])
    outputs_cpp = generate_tensor_desc_list_cpp(outputs, "        ")
    lines.append(f"        {outputs_cpp},")
    
    # attrs
    attrs = case.get("attrs", [])
    attrs_cpp = generate_op_attr_list_cpp(attrs, "        ")
    lines.append(f"        {attrs_cpp},")
    
    # hasExpectTilingKey 和 expectTilingKey
    has_expect = case.get("hasExpectTilingKey", False)
    expect_key = case.get("expectTilingKey", 0)
    
    lines.append(f'        {"true" if has_expect else "false"},')
    lines.append(f'        {expect_key}UL')
    
    lines.append("    },")
    
    return "\n".join(lines)


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
    if mode in ("all_gather_matmul", "all_gather_matmul_v2", "matmul_all_reduce"):
        key = "compile_info"
    else:
        key = "expectTilingData"
    
    common_value, count = find_most_common_value(cases, key)
    
    if not common_value or count < 2:
        return "", ""
    
    if mode in ("all_gather_matmul", "all_gather_matmul_v2"):
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
def generate_all_gather_matmul_case(case: Dict[str, Any], common_value: str, include_expect_success: bool = True) -> str:
    """
    生成 AllGatherMatmul 用例 (多行格式)
    
    Args:
        case: 测试用例数据
        common_value: COMPILE_INFO 常量值
        include_expect_success: 是否包含 expectSuccess 字段 (V2 有, V1 没有)
    """
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
    
    # 第五行: output_dtype + bool + expectTilingKey (V2 包含 expectSuccess，V1 不包含)
    if include_expect_success:
        line5_fields = ["output_dtype", "is_trans_a", "is_trans_b", "expectSuccess", "expectTilingKey"]
    else:
        line5_fields = ["output_dtype", "is_trans_a", "is_trans_b", "expectTilingKey"]
    
    line5_vals = []
    for f in line5_fields:
        if f == "expectSuccess":
            # 默认期望成功(True)，除非数据中显式指定为 False
            val = case.get(f, True)
        elif "trans" in f:
            val = case.get(f, False)
        elif f == "expectTilingKey":
            val = case.get(f, 0)
        else:
            val = case.get(f, "ge::DT_FLOAT16" if "dtype" in f else 0)
        
        line5_vals.append(convert_value_to_cpp(f, val, use_compile_info))

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


# 字段名映射表： {结构体名: {生成的字段名: JSON中的字段名}}
FIELD_NAME_MAPPING = {
    "GroupedMatMulAllReduceTilingTestParam": {
        "x_shape": "x1_shape",
        "x_dtype": "x1_dtype",
        "weight_shape": "x2_shape",
        "weight_dtype": "x2_dtype"
    },
    "BatchMatMulReduceScatterAlltoAllTilingTestParam": {
        "weight_shape": "w_shape",
        "weight_dtype": "w_dtype"
    },
    "AlltoAllAllGatherBatchMatMulTilingTestParam": {
        "weight_shape": "w_shape",
        "weight_dtype": "w_dtype"
    }
    # 注意: MatmulReduceScatterTilingTestParam 不需要特殊映射，
    # 因为模板字段名与输入 JSONL 字段名一致 (x3_shape, x4_shape 等)
}
# 通用映射（对所有算子生效，如果字段名匹配）
COMMON_FIELD_MAPPING = {
}


def parse_struct_fields(template_content: str, struct_name: str) -> List[Tuple[str, str]]:
    """
    解析结构体定义，返回 [(字段名, 类型), ...]列表。
    """
    # 找到结构体定义的开始
    start_pattern = f"struct {struct_name} {{"
    start_idx = template_content.find(start_pattern)
    if start_idx == -1:
        return []
    
    # 找到结构体定义的结束
    end_idx = template_content.find("};", start_idx)
    if end_idx == -1:
        return []
        
    struct_body = template_content[start_idx + len(start_pattern):end_idx]
    
    fields = []
    # 逐行解析
    for line in struct_body.split('\n'):
        line = line.strip()
        if not line or line.startswith('//'):
            continue
            
        # 移除分号及之后的内容
        code_part = line.split(';')[0].strip()
        if not code_part:
            continue

        # 分离类型和名称 (取最后一个空格作为分隔符)
        parts = code_part.rsplit(' ', 1)
        if len(parts) == 2:
            field_type = parts[0].strip()
            field_name = parts[1].strip()
            # 清理可能的指针或引用符号 (虽然在结构体成员中不常见，但以防万一)
            field_name = field_name.replace('*', '').replace('&', '')
            fields.append((field_name, field_type))
            
    return fields


def generate_generic_case(case: Dict[str, Any], fields: List[Tuple[str, str]], common_value: str, struct_name: str = "") -> str:
    """基于结构体字段生成通用初始化列表"""
    use_compile_info = False
    # 检查是否有 compile_info 字段且值匹配
    if "compile_info" in case and case["compile_info"] == common_value:
        use_compile_info = True
    elif "expectTilingData" in case and case["expectTilingData"] == common_value:
        use_compile_info = True

    mapping = FIELD_NAME_MAPPING.get(struct_name, {})

    values = []
    for field_name, field_type in fields:
        # 获取值，如果不存在则使用默认值
        # 优先使用特定算子的映射，然后是通用映射，最后是字段原名
        lookup_name = mapping.get(field_name, COMMON_FIELD_MAPPING.get(field_name, field_name))
        val = case.get(lookup_name)
        
        if val is None:
            # 根据类型提供默认值
            if "string" in field_type:
                val = ""
            elif "vector" in field_type or "initializer_list" in field_type:
                val = []
            elif "bool" in field_type:
                val = False
            else: # int, etc
                val = 0
        
        # 转换值
        cpp_val = convert_value_to_cpp(field_name, val, use_compile_info)
        
        # 特殊处理 ge::DataType 默认值
        if "ge::DataType" in field_type and (val is None or val == 0 or val == ""):
             if "ge::" not in str(cpp_val):
                 cpp_val = "ge::DT_FLOAT"
        
        values.append(cpp_val)
        
    return "    {" + ", ".join(values) + "},"


def generate_simple_test_param_case(case: Dict[str, Any], struct_name: str) -> str:
    """
    生成简单的 TestParam 用例（只有 caseName, blockDim, tilingKey）。
    用于 matmul_all_reduce_add_rms_norm 等特殊模板。
    """
    case_name = case.get("case_name", "")
    block_dim = case.get("blockDim", 8)
    tiling_key = case.get("tilingKey", 0)
    
    # 处理大整数，使用 UL 后缀
    if tiling_key > 65535:
        tiling_key_str = f"{tiling_key}"
    else:
        tiling_key_str = str(tiling_key)
    
    return f'        {{"{case_name}", {block_dim}, {tiling_key_str}}},'


def generate_allto_allv_complex_case(case: Dict[str, Any]) -> str:
    """
    生成 allto_allv_grouped_mat_mul 的复杂 TestParam 用例。
    格式: {"test_name", {{pairs...}}, {{vec_pairs...}}, {{dtype_pairs...}}, status}
    """
    test_name = case.get("test_name", "")
    status = case.get("status", "ge::GRAPH_FAILED")
    
    # 处理 tiling_params_str_pair: [{"key": "e", "value": "64"}, ...]
    str_pairs = case.get("tiling_params_str_pair", [])
    if str_pairs:
        pairs_str = ", ".join(f'{{"{p.get("key", "")}", "{p.get("value", "")}"}}' for p in str_pairs)
        str_pair_cpp = "{" + pairs_str + "}"
    else:
        str_pair_cpp = "{}"
    
    # 处理 tiling_params_vec_pair: [{"key": "send_counts", "value": [...]}, ...]
    vec_pairs = case.get("tiling_params_vec_pair", [])
    if vec_pairs:
        vec_parts = []
        for p in vec_pairs:
            key = p.get("key", "")
            values = p.get("value", [])
            if values:
                vec_str = ", ".join(str(v) for v in values)
                vec_parts.append(f'{{"{key}", {{{vec_str}}}}}')
            else:
                vec_parts.append(f'{{"{key}", {{}}}}')
        vec_pair_cpp = "{" + ", ".join(vec_parts) + "}"
    else:
        vec_pair_cpp = "{}"
    
    # 处理 tiling_dTypes_pair: [{"idx": 0, "dtype": "ge::DT_FLOAT16"}, ...]
    dtype_pairs = case.get("tiling_dTypes_pair", [])
    if dtype_pairs:
        dtype_parts = []
        for p in dtype_pairs:
            idx = p.get("idx", 0)
            dtype = p.get("dtype", "ge::DT_FLOAT16")
            dtype_parts.append(f'{{{idx}, {dtype}}}')
        dtype_pair_cpp = "{" + ", ".join(dtype_parts) + "}"
    else:
        dtype_pair_cpp = "{}"
    
    return f'    {{"{test_name}", {str_pair_cpp}, {vec_pair_cpp}, {dtype_pair_cpp}, {status}}},'


def generate_cases_params(mode: str, cases: List[Dict[str, Any]], 
                          struct_name: str, common_value: str,
                          template_content: str = "") -> str:
    """生成 cases_params 数组代码"""
    lines = []
    
    # 从模板中提取参数数组名称
    param_array_names = extract_param_array_names(template_content)
    
    # 检测是否有多数组模式（如 casesParamsQuant + InValidCheckcasesParamsQuant）
    has_invalid_array = any("InValid" in name for name in param_array_names)
    
    if has_invalid_array and len(param_array_names) >= 2:
        # 多数组模式：将用例按前缀分组
        valid_cases = []
        invalid_cases = []
        for case in cases:
            case_name = case.get("case_name", "")
            if case_name.startswith("InValid"):
                invalid_cases.append(case)
            else:
                valid_cases.append(case)
        
        # 找出有效和无效数组的名称
        valid_array = [n for n in param_array_names if "InValid" not in n][0]
        invalid_array = [n for n in param_array_names if "InValid" in n][0]
        
        # 生成两个数组
        result_lines = []
        for array_name, array_cases in [(valid_array, valid_cases), (invalid_array, invalid_cases)]:
            result_lines.append(f"static {struct_name} {array_name}[] = {{")
            for case in array_cases:
                case_code = generate_simple_test_param_case(case, struct_name)
                result_lines.append(case_code)
            result_lines.append("")
            result_lines.append("};")
        
        return "\n".join(result_lines)
    
    param_array_name = param_array_names[0] if param_array_names else "cases_params"
    
    # 某些模式需要添加注释 (只有 V1 版本需要，V2 不需要)
    if mode == "all_gather_matmul":
        lines.append("// 用例列表集")
    
    lines.append(f"{struct_name} {param_array_name}[] = {{")
    
    # 尝试解析结构体字段以使用通用生成逻辑
    struct_fields = parse_struct_fields(template_content, struct_name)
    
    for i, case in enumerate(cases):
        if mode == "moe_tensor_desc":
            # 复杂的 TensorDescription 模式
            case_code = generate_moe_tensor_desc_case(case)
            lines.append(case_code)
        elif mode == "allto_allv_complex":
            # allto_allv_grouped_mat_mul 的复杂结构
            case_code = generate_allto_allv_complex_case(case)
            lines.append(case_code)
        elif mode == "all_gather_matmul_v2":
            # AllGatherMatmul V2 (带 expectSuccess)
            case_code = generate_all_gather_matmul_case(case, common_value, include_expect_success=True)
            lines.append(case_code)
            if i < len(cases) - 1:
                lines.append("")
        elif mode == "all_gather_matmul":
            # AllGatherMatmul V1 (不带 expectSuccess)
            case_code = generate_all_gather_matmul_case(case, common_value, include_expect_success=False)
            lines.append(case_code)
            if i < len(cases) - 1:
                lines.append("")
        elif struct_fields:
            # 如果成功解析了结构体字段，使用通用生成逻辑
            case_code = generate_generic_case(case, struct_fields, common_value, struct_name)
            lines.append(case_code)
        elif mode == "matmul_all_reduce":
            # 降级到旧的 matmul 逻辑
            case_code = generate_matmul_all_reduce_case(case, common_value)
            lines.append(case_code)
        else:  # distribute_barrier (legacy)
            case_code = generate_distribute_barrier_case(case, common_value)
            lines.append(case_code)
    
    lines.append("};")
    return "\n".join(lines)


def generate_unit_test(state: WorkflowState) -> WorkflowState:
    """生成单元测试文件的主函数"""
    template_path = Path(state["template_file_path"])
    input_path = Path(state.get("input_path", ""))
    output_path = Path(state["output_path"])
    
    template_content = template_path.read_text(encoding="utf-8")
    
    # 动态提取结构体名称和检测模式
    struct_name = extract_struct_name(template_content)
    if not struct_name:
        print(f"警告: 无法从模板 {template_path} 中提取结构体名称，生成的代码可能不完整。")
    
    mode = detect_mode(template_content, struct_name)
    print(f"检测到生成模式: {mode} (Struct: {struct_name})")
    
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
    # 使用提取到的 struct_name 和 template_content
    cases_code = generate_cases_params(mode, cases, struct_name, common_value, template_content)
    
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
