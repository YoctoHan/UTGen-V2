#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
根据 op_host 下的 *_def.cpp 自动解析并生成对应的 UT 公共代码骨架。

用法:
    python3 gen_param_struct_from_def.py <op_def.cpp> <out.cpp>

示例:
    python3 gen_param_struct_from_def.py mc2/matmul_all_reduce/op_host/matmul_all_reduce_def.cpp workspace/results/matmul_all_reduce.cpp
"""

import re
import sys
from pathlib import Path
from typing import List, Tuple
from state import WorkflowState


def parse_op_def(src: str) -> Tuple[str, List[Tuple[str, str]], List[Tuple[str, str]], List[Tuple[str, str, str]]]:
    """
    解析 *_def.cpp，提取：
    - op_class_name: 算子类名（如 MatmulAllReduce, AllGatherMatmul, DistributeBarrier）
    - inputs: [(name, param_type), ...] 其中 param_type 为 REQUIRED 或 OPTIONAL
    - outputs: [(name, param_type), ...]
    - attrs: [(name, attr_type, default_value), ...] 其中 attr_type 为 String/Bool/Int 等
    """
    # 提取类名
    class_match = re.search(r"class\s+(\w+)\s*:\s*public\s+OpDef", src)
    if not class_match:
        raise RuntimeError("未找到 OpDef 派生类定义")
    op_class_name = class_match.group(1)

    # 提取 Input（只取第一个 AICore config 之前的定义，避免重复）
    # 找到 OpAICoreConfig 的位置作为截止点
    aicore_pos = src.find("OpAICoreConfig")
    if aicore_pos == -1:
        main_src = src
    else:
        main_src = src[:aicore_pos]

    inputs = []
    for m in re.finditer(
        r'this->Input\("(\w+)"\)\s*\n?\s*\.ParamType\((REQUIRED|OPTIONAL)\)',
        main_src,
        re.S,
    ):
        name = m.group(1)
        param_type = m.group(2)
        if name not in [i[0] for i in inputs]:  # 去重
            inputs.append((name, param_type))

    # 提取 Output
    outputs = []
    for m in re.finditer(
        r'this->Output\("(\w+)"\)\s*\n?\s*\.ParamType\((REQUIRED|OPTIONAL)\)',
        main_src,
        re.S,
    ):
        name = m.group(1)
        param_type = m.group(2)
        if name not in [o[0] for o in outputs]:  # 去重
            outputs.append((name, param_type))

    # 提取 Attr
    attrs = []
    for m in re.finditer(
        r'this->Attr\("(\w+)"\)\.AttrType\((REQUIRED|OPTIONAL)\)\.(\w+)\(([^)]*)\)',
        main_src,
    ):
        attr_name = m.group(1)
        attr_type = m.group(3)  # String, Bool, Int
        attr_default = m.group(4).strip()  # 可能为空
        if attr_name not in [a[0] for a in attrs]:  # 去重
            attrs.append((attr_name, attr_type, attr_default))

    return op_class_name, inputs, outputs, attrs


def is_mc2_matmul_like(attrs: List[Tuple[str, str, str]]) -> bool:
    """
    判断是否是 mc2 matmul 类算子（如 MatmulAllReduce, AllGatherMatmul）
    特征：有 is_trans_a, is_trans_b 属性
    """
    attr_names = {a[0] for a in attrs}
    return "is_trans_a" in attr_names and "is_trans_b" in attr_names


def is_all_gather_matmul(op_class_name: str) -> bool:
    """判断是否是 AllGatherMatmul 类算子"""
    return "AllGatherMatmul" in op_class_name


def gen_mc2_matmul_like_code(op_class_name: str, inputs: List, outputs: List, attrs: List) -> str:
    """
    生成 mc2 matmul 类算子的 UT 代码骨架，精确匹配源文件格式
    """
    namespace_name = f"{op_class_name}UT"
    struct_name = f"{op_class_name}TilingTestParam"
    param_class_name = f"{op_class_name}TilingParam"
    compile_info_name = f"{op_class_name}CompileInfo"

    # 固定的 10 个输入字段（mc2 matmul 通用模板）
    input_fields = [
        "x1", "x2", "bias", "x3",
        "antiquant_scale", "antiquant_offset",
        "dequant_scale", "pertoken_scale",
        "comm_quant_scale_1", "comm_quant_scale_2",
    ]

    # 构建 attr 列表代码
    attr_code_lines = []
    for attr_name, attr_type, attr_default in attrs:
        if attr_type == "String":
            if attr_name == "group":
                attr_code_lines.append(f'            {{"{attr_name}", build_from<std::string>("group")}}')
            elif attr_name == "reduce_op":
                attr_code_lines.append(f'            {{"{attr_name}", build_from<std::string>("sum")}}')
            else:
                attr_code_lines.append(f'            {{"{attr_name}", build_from<std::string>("")}}')
        elif attr_type == "Bool":
            if attr_name in ("is_trans_a", "is_trans_b"):
                attr_code_lines.append(f'            {{"{attr_name}", build_from<bool>(param.{attr_name})}}')
            else:
                default_val = "true" if attr_default == "true" else "false"
                attr_code_lines.append(f'            {{"{attr_name}", build_from<bool>({default_val})}}')
        elif attr_type == "Int":
            attr_code_lines.append(f'            {{"{attr_name}", build_from<int64_t>(0)}}')
    attr_code = ",\n".join(attr_code_lines)

    # 判断是否有多个输出（如 AllGatherMatmul 有 y 和 gather_out）
    has_multiple_outputs = len(outputs) > 1

    lines = []

    # 版权头
    lines.append('/**')
    lines.append(' * Copyright (c) Huawei Technologies Co., Ltd. 2025. All rights reserved.')
    lines.append(' * This file is a part of the CANN Open Software.')
    lines.append(' * Licensed under CANN Open Software License Agreement Version 2.0 (the "License").')
    lines.append(' * Please refer to the License for details. You may not use this file except in compliance with the License.')
    lines.append(' * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,')
    lines.append(' * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.')
    lines.append(' * See LICENSE in the root of the software repository for the full text of the License.')
    lines.append(' */')
    lines.append('')
    lines.append('#include <iostream>')
    lines.append('#include <cctype>')
    lines.append('#include <gtest/gtest.h>')
    lines.append('#include "mc2_tiling_case_executor.h"')
    lines.append('')
    lines.append(f'namespace {namespace_name} {{')
    lines.append('template <typename T>')
    lines.append('auto build_from(const T& value){')
    lines.append('    return Ops::Transformer::AnyValue::CreateFrom<T>(value);')
    lines.append('}')
    lines.append('')
    lines.append('// 定义用例信息结构体')
    lines.append(f'struct {struct_name} {{')
    lines.append('    // 平台信息')
    lines.append('    uint64_t inputTotalNum;')
    lines.append('    string case_name;')
    lines.append('    string compile_info;')
    lines.append('    string soc_version;')
    lines.append('    uint64_t coreNum;')
    lines.append('    uint64_t ubSize;')
    lines.append('    uint64_t tilingDataSize;')
    lines.append('')
    lines.append('    // 输入信息shape')
    for field in input_fields:
        lines.append(f'    std::initializer_list<int64_t> {field}_shape;')
    lines.append('    std::initializer_list<int64_t> output_shape; // 输出信息')
    lines.append('')
    lines.append('    // 输入信息类型')
    for field in input_fields:
        lines.append(f'    ge::DataType {field}_dtype;')
    lines.append('    ge::DataType output_dtype; // 输出信息')
    lines.append('')
    lines.append('    bool is_trans_a;')
    lines.append('    bool is_trans_b;')
    lines.append('')
    lines.append('    // 结果')
    lines.append('    uint64_t expectTilingKey;')
    lines.append('};')
    lines.append('')
    lines.append(f'class {param_class_name} : public ::testing::TestWithParam<{struct_name}> {{')
    lines.append('protected:')
    lines.append('    static void SetUpTestCase()')
    lines.append('    {')
    lines.append(f'        std::cout << "{op_class_name}Tiling SetUp" << std::endl;')
    lines.append('    }')
    lines.append('')
    lines.append('    static void TearDownTestCase()')
    lines.append('    {')
    lines.append(f'        std::cout << "{op_class_name}Tiling TearDown" << std::endl;')
    lines.append('    }')
    lines.append('};')
    lines.append('')
    lines.append('gert::StorageShape make_shape(const std::initializer_list<int64_t>& input_shape){')
    lines.append('    if (input_shape.size() == 0){')
    lines.append('        return gert::StorageShape{};')
    lines.append('    }')
    lines.append('    return gert::StorageShape{input_shape, input_shape};')
    lines.append('}')
    lines.append('')
    lines.append(f'void TestOneParamCase(const {struct_name}& param){{')
    lines.append(f'    struct {compile_info_name} {{}};')
    lines.append(f'    {compile_info_name} compileInfo;')
    lines.append('')
    lines.append('    // 存取用户输入的用例信息')
    lines.append('    std::vector<pair<std::initializer_list<int64_t>, ge::DataType>> shapeDtypeList = {')
    for i, field in enumerate(input_fields):
        comma = ',' if i < len(input_fields) - 1 else ''
        trailing = ' ' if i < len(input_fields) - 1 else ''
        lines.append(f'    {{param.{field}_shape, param.{field}_dtype}}{comma}{trailing}')
    lines.append('    };')
    lines.append('')
    lines.append('    // 按需提取后传入构造')
    lines.append('    std::vector<gert::TilingContextPara::TensorDescription> inputList;')
    lines.append('    for (int i = 0; i < param.inputTotalNum; i++){')
    lines.append('        inputList.push_back({make_shape(shapeDtypeList[i].first), shapeDtypeList[i].second, ge::FORMAT_ND});')
    lines.append('    }')
    lines.append('')

    if has_multiple_outputs:
        lines.append('    std::vector<gert::TilingContextPara::TensorDescription> outputList = {')
        lines.append('        {{param.output_shape, param.output_shape}, param.output_dtype, ge::FORMAT_ND},')
        lines.append('        {make_shape(param.x1_shape), param.x1_dtype, ge::FORMAT_ND},')
        lines.append('    };')
        lines.append('')
        lines.append(f'    gert::TilingContextPara tilingContextPara("{op_class_name}", inputList, outputList,')
        lines.append('        {')
        lines.append(attr_code)
        lines.append('        },')
        lines.append('        &compileInfo, param.soc_version, param.compile_info, param.tilingDataSize);')
    else:
        lines.append(f'    gert::TilingContextPara tilingContextPara("{op_class_name}", inputList,')
        lines.append('        {')
        lines.append('            {{param.output_shape, param.output_shape}, param.output_dtype, ge::FORMAT_ND},')
        lines.append('        },')
        lines.append('        {')
        lines.append(attr_code)
        lines.append('        },')
        lines.append('        &compileInfo, param.soc_version, param.compile_info, param.tilingDataSize);')

    lines.append('')
    lines.append('    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, param.expectTilingKey);')
    lines.append('}')
    lines.append('')
    lines.append(f'TEST_P({param_class_name}, general_case)')
    lines.append('{')
    lines.append('    Mc2Hcom::MockValues hcomTopologyMockValues{{"rankNum", 8}};')
    lines.append('    Mc2Hcom::MC2HcomTopologyMocker::GetInstance().SetValues(hcomTopologyMockValues);')
    lines.append('')
    lines.append('    const auto &param = GetParam();')
    lines.append('    TestOneParamCase(param);')
    lines.append('')
    lines.append('    Mc2Hcom::MC2HcomTopologyMocker::GetInstance().Reset();')
    lines.append('}')
    lines.append('')
    lines.append('INSTANTIATE_TEST_SUITE_P(')
    lines.append('    general_cases_params,')
    lines.append(f'    {param_class_name},')
    lines.append('    ::testing::ValuesIn(cases_params),')
    lines.append(f'    [](const ::testing::TestParamInfo<{struct_name}> &info) {{')
    lines.append('        std::string name = info.param.case_name;')
    lines.append('        for (char &c : name) {')
    lines.append("            if (!std::isalnum(static_cast<unsigned char>(c)) && c != '_') {")
    lines.append("                c = '_';")
    lines.append('            }')
    lines.append('        }')
    lines.append('        return name;')
    lines.append('    });')
    lines.append('')
    lines.append(f'}} // namespace {namespace_name}')

    return '\n'.join(lines)


def gen_all_gather_matmul_like_code(op_class_name: str, inputs: List, outputs: List, attrs: List) -> str:
    """
    生成 AllGatherMatmul 类算子的 UT 代码骨架，精确匹配源文件格式
    """
    namespace_name = f"{op_class_name}UT"
    struct_name = f"{op_class_name}TilingTestParam"
    param_class_name = f"{op_class_name}TilingParam"
    compile_info_name = f"{op_class_name}CompileInfo"

    # 固定的 10 个输入字段
    input_fields = [
        "x1", "x2", "bias", "x3",
        "antiquant_scale", "antiquant_offset",
        "dequant_scale", "pertoken_scale",
        "comm_quant_scale_1", "comm_quant_scale_2",
    ]

    # AllGatherMatmul 只使用这 5 个 attr（与源 UT 文件一致）
    allowed_attrs = ["group", "is_trans_a", "is_trans_b", "gather_index", "comm_turn"]
    
    # 构建 attr 列表代码
    attr_code_lines = []
    for attr_name, attr_type, attr_default in attrs:
        if attr_name not in allowed_attrs:
            continue
        if attr_type == "String":
            if attr_name == "group":
                attr_code_lines.append(f'            {{"{attr_name}", build_from<std::string>("group")}}')
            else:
                attr_code_lines.append(f'            {{"{attr_name}", build_from<std::string>("")}}')
        elif attr_type == "Bool":
            if attr_name in ("is_trans_a", "is_trans_b"):
                attr_code_lines.append(f'            {{"{attr_name}", build_from<bool>(param.{attr_name})}}')
            else:
                default_val = "true" if attr_default == "true" else "false"
                attr_code_lines.append(f'            {{"{attr_name}", build_from<bool>({default_val})}}')
        elif attr_type == "Int":
            attr_code_lines.append(f'            {{"{attr_name}", build_from<int64_t>(0)}}')
    # 最后一个 attr 后面也要有逗号（与源文件一致）
    attr_code = ",\n".join(attr_code_lines) + ","

    lines = []

    # 版权头（AllGatherMatmul 风格）
    lines.append('/**')
    lines.append(' * This program is free software, you can redistribute it and/or modify.')
    lines.append(' * Copyright (c) 2025 Huawei Technologies Co., Ltd.')
    lines.append(' * This file is a part of the CANN Open Software.')
    lines.append(' * Licensed under CANN Open Software License Agreement Version 2.0 (the "License").')
    lines.append(' * Please refer to the License for details. You may not use this file except in compliance with the License.')
    lines.append(' * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.')
    lines.append(' * See LICENSE in the root of the software repository for the full text of the License.')
    lines.append(' */')
    lines.append('')
    lines.append('#include <iostream>')
    lines.append('#include <thread>')
    lines.append('#include <cctype>')
    lines.append('#include <gtest/gtest.h>')
    lines.append('#include "mc2_tiling_case_executor.h"')
    lines.append('')
    lines.append(f'namespace {namespace_name} {{')
    lines.append('')
    lines.append('namespace {')
    lines.append('')
    lines.append('template <typename T>')
    lines.append('auto build_from(const T &value)')
    lines.append('{')
    lines.append('    return Ops::Transformer::AnyValue::CreateFrom<T>(value);')
    lines.append('}')
    lines.append('')
    lines.append('// 定义用例信息结构体')
    lines.append(f'struct {struct_name} {{')
    lines.append('    // 平台信息')
    lines.append('    uint64_t inputTotalNum;')
    lines.append('    std::string case_name;')
    lines.append('    std::string compile_info;')
    lines.append('    std::string soc_version;')
    lines.append('    uint64_t coreNum;')
    lines.append('    uint64_t ubSize;')
    lines.append('    uint64_t tilingDataSize;')
    lines.append('')
    lines.append('    // 输入信息shape')
    for field in input_fields:
        lines.append(f'    std::initializer_list<int64_t> {field}_shape;')
    lines.append('    std::initializer_list<int64_t> output_shape; // 输出信息')
    lines.append('')
    lines.append('    // 输入信息类型')
    for field in input_fields:
        lines.append(f'    ge::DataType {field}_dtype;')
    lines.append('    ge::DataType output_dtype; // 输出信息')
    lines.append('')
    lines.append('    bool is_trans_a;')
    lines.append('    bool is_trans_b;')
    lines.append('')
    lines.append('    // 结果')
    lines.append('    uint64_t expectTilingKey;')
    lines.append('};')
    lines.append('')
    lines.append(f'class {param_class_name} : public ::testing::TestWithParam<{struct_name}> {{')
    lines.append('protected:')
    lines.append('    static void SetUpTestCase()')
    lines.append('    {')
    lines.append(f'        std::cout << "{op_class_name}Tiling SetUp" << std::endl;')
    lines.append('    }')
    lines.append('')
    lines.append('    static void TearDownTestCase()')
    lines.append('    {')
    lines.append(f'        std::cout << "{op_class_name}Tiling TearDown" << std::endl;')
    lines.append('    }')
    lines.append('};')
    lines.append('')
    lines.append('gert::StorageShape make_shape(const std::initializer_list<int64_t> &input_shape)')
    lines.append('{')
    lines.append('    if (input_shape.size() == 0) {')
    lines.append('        return gert::StorageShape{};')
    lines.append('    }')
    lines.append('    return gert::StorageShape{input_shape, input_shape};')
    lines.append('}')
    lines.append('')
    lines.append(f'void TestOneParamCase(const {struct_name} &param)')
    lines.append('{')
    lines.append(f'    struct {compile_info_name} {{}};')
    lines.append(f'    {compile_info_name} compileInfo;')
    lines.append('')
    lines.append('    // 存取用户输入的用例信息')
    lines.append('    std::vector<std::pair<std::initializer_list<int64_t>, ge::DataType>> shapeDtypeList = {')
    for i, field in enumerate(input_fields):
        comma = '' if i == len(input_fields) - 1 else ','
        lines.append(f'        {{param.{field}_shape, param.{field}_dtype}}{comma}')
    lines.append('    };')
    lines.append('')
    lines.append('    // 按需提取后传入构造')
    lines.append('    std::vector<gert::TilingContextPara::TensorDescription> inputList;')
    lines.append('    for (uint64_t i = 0; i < param.inputTotalNum; ++i) {')
    lines.append('        inputList.push_back({make_shape(shapeDtypeList[i].first), shapeDtypeList[i].second, ge::FORMAT_ND});')
    lines.append('    }')
    lines.append('')
    lines.append('    std::vector<gert::TilingContextPara::TensorDescription> outputList = {')
    lines.append('        {{param.output_shape, param.output_shape}, param.output_dtype, ge::FORMAT_ND},')
    lines.append('        {make_shape(param.x1_shape), param.x1_dtype, ge::FORMAT_ND},')
    lines.append('    };')
    lines.append('')
    lines.append(f'    gert::TilingContextPara tilingContextPara("{op_class_name}", inputList, outputList,')
    lines.append('        {')
    lines.append(attr_code)
    lines.append('        },')
    lines.append('        &compileInfo, param.soc_version, param.compile_info, param.tilingDataSize);')
    lines.append('')
    lines.append('    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, param.expectTilingKey);')
    lines.append('}')
    lines.append('')
    lines.append(f'TEST_P({param_class_name}, general_case)')
    lines.append('{')
    lines.append('    Mc2Hcom::MockValues hcomTopologyMockValues{{"rankNum", 8}};')
    lines.append('    Mc2Hcom::MC2HcomTopologyMocker::GetInstance().SetValues(hcomTopologyMockValues);')
    lines.append('')
    lines.append('    const auto &param = GetParam();')
    lines.append('    TestOneParamCase(param);')
    lines.append('')
    lines.append('    Mc2Hcom::MC2HcomTopologyMocker::GetInstance().Reset();')
    lines.append('}')
    lines.append('')
    lines.append('INSTANTIATE_TEST_SUITE_P(')
    lines.append('    general_cases_params,')
    lines.append(f'    {param_class_name},')
    lines.append('    ::testing::ValuesIn(cases_params),')
    lines.append(f'    [](const ::testing::TestParamInfo<{struct_name}> &info) {{')
    lines.append('        std::string name = info.param.case_name;')
    lines.append('        for (char &c : name) {')
    lines.append("            if (!std::isalnum(static_cast<unsigned char>(c)) && c != '_') {")
    lines.append("                c = '_';")
    lines.append('            }')
    lines.append('        }')
    lines.append('        return name;')
    lines.append('    });')
    lines.append('')
    lines.append('} // anonymous namespace')
    lines.append('')
    lines.append(f'}} // namespace {namespace_name}')

    return '\n'.join(lines)


def gen_distribute_barrier_like_code(op_class_name: str, inputs: List, outputs: List, attrs: List) -> str:
    """
    生成 DistributeBarrier 类算子的 UT 代码骨架
    """
    struct_name = f"{op_class_name}TilingTestParam"
    param_class_name = f"{op_class_name}TilingParam"
    compile_info_name = f"{op_class_name}CompileInfo"

    # 构建 attr 代码
    attr_code_lines = []
    for idx, (attr_name, attr_type, attr_default) in enumerate(attrs):
        if attr_type == "String":
            attr_code_lines.append(f'{{"{attr_name}", Ops::Transformer::AnyValue::CreateFrom<std::string>(param.{attr_name})}}')
        elif attr_type == "Int":
            attr_code_lines.append(f'{{"{attr_name}", Ops::Transformer::AnyValue::CreateFrom<int64_t>(param.{attr_name})}}')
        elif attr_type == "Bool":
            attr_code_lines.append(f'{{"{attr_name}", Ops::Transformer::AnyValue::CreateFrom<bool>(param.{attr_name})}}')
    attr_code = ",\n         ".join(attr_code_lines)

    # 构建 struct 字段（基于 attrs）
    struct_fields = []
    struct_fields.append("    std::string case_name;")
    struct_fields.append("    int64_t m;")
    struct_fields.append("    int64_t n;")
    struct_fields.append("    ge::DataType dtype;")
    for attr_name, attr_type, _ in attrs:
        if attr_type == "String":
            struct_fields.append(f"    std::string {attr_name};")
        elif attr_type == "Int":
            struct_fields.append(f"    int64_t {attr_name};")
        elif attr_type == "Bool":
            struct_fields.append(f"    bool {attr_name};")
    struct_fields.append("    std::string soc_version;")
    struct_fields.append("    uint64_t coreNum;")
    struct_fields.append("    uint64_t ubSize;")
    struct_fields.append("    uint64_t expectTilingKey;")
    struct_fields.append("    std::string expectTilingData;")
    struct_fields.append("    std::vector<size_t> expectWorkspaces;")
    struct_fields.append("    uint64_t mc2TilingDataReservedLen;")
    struct_fields_code = "\n".join(struct_fields)

    lines = []
    lines.append('/**')
    lines.append(' * This program is free software, you can redistribute it and/or modify.')
    lines.append(' * Copyright (c) 2025 Huawei Technologies Co., Ltd.')
    lines.append(' * This file is a part of the CANN Open Software.')
    lines.append(' * Licensed under CANN Open Software License Agreement Version 2.0 (the "License").')
    lines.append(' * Please refer to the License for details. You may not use this file except in compliance with the License.')
    lines.append(' * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.')
    lines.append(' * See LICENSE in the root of the software repository for the full text of the License.')
    lines.append(' */')
    lines.append('')
    lines.append('#include <iostream>')
    lines.append('#include <vector>')
    lines.append('#include <string>')
    lines.append('#include <gtest/gtest.h>')
    lines.append('#include "mc2_tiling_case_executor.h"')
    lines.append('')
    lines.append('using namespace std;')
    lines.append('')
    lines.append('namespace {')
    lines.append('')
    lines.append(f'struct {struct_name} {{')
    lines.append(struct_fields_code)
    lines.append('};')
    lines.append('')
    lines.append(f'class {param_class_name} : public ::testing::TestWithParam<{struct_name}> {{')
    lines.append('protected:')
    lines.append('    static void SetUpTestCase()')
    lines.append('    {')
    lines.append(f'        std::cout << "{op_class_name}Tiling SetUp" << std::endl;')
    lines.append('    }')
    lines.append('')
    lines.append('    static void TearDownTestCase()')
    lines.append('    {')
    lines.append(f'        std::cout << "{op_class_name}Tiling TearDown" << std::endl;')
    lines.append('    }')
    lines.append('};')
    lines.append('')
    lines.append(f'void TestOneParamCase(const {struct_name} &param)')
    lines.append('{')
    lines.append(f'    struct {compile_info_name} {{}} compileInfo;')
    lines.append(f'    gert::TilingContextPara tilingContextPara("{op_class_name}",')
    lines.append('        {{{{param.m, param.n}, {param.m, param.n}}, param.dtype, ge::FORMAT_ND},},')
    lines.append('        {{{{param.m, param.n}, {param.m, param.n}}, param.dtype, ge::FORMAT_ND},},')
    lines.append(f'        {{{attr_code}}},')
    lines.append('        &compileInfo, param.soc_version, param.coreNum, param.ubSize);')
    lines.append('')
    lines.append('    Mc2Hcom::MockValues hcomTopologyMockValues{{"rankNum", 8}};')
    lines.append('    Mc2ExecuteTestCase(tilingContextPara, hcomTopologyMockValues, ge::GRAPH_SUCCESS, param.expectTilingKey,')
    lines.append('        param.expectTilingData, param.expectWorkspaces, param.mc2TilingDataReservedLen);')
    lines.append('}')
    lines.append('')
    lines.append(f'TEST_P({param_class_name}, general_case)')
    lines.append('{')
    lines.append('    const auto &param = GetParam();')
    lines.append('    TestOneParamCase(param);')
    lines.append('}')
    lines.append('')
    lines.append('INSTANTIATE_TEST_SUITE_P(')
    lines.append('    general_cases_params,')
    lines.append(f'    {param_class_name},')
    lines.append('    ::testing::ValuesIn(cases_params),')
    lines.append(f'    [](const ::testing::TestParamInfo<{struct_name}> &info) {{')
    lines.append('        std::string name = info.param.case_name;')
    lines.append('        for (char &c : name) {')
    lines.append("            if (!std::isalnum(static_cast<unsigned char>(c)) && c != '_') {")
    lines.append("                c = '_';")
    lines.append('            }')
    lines.append('        }')
    lines.append('        return name;')
    lines.append('    });')
    lines.append('')
    lines.append('} // anonymous namespace')

    return '\n'.join(lines)


def _generate_template_core(def_path: Path, out_path: Path) -> None:
    """
    核心生成逻辑：
    - 从 def_path 读取 *_def.cpp
    - 解析为 op_class_name / inputs / outputs / attrs
    - 根据算子类型选择对应的代码生成函数
    - 写入 out_path
    """
    src = def_path.read_text(encoding="utf-8")

    # 解析 def.cpp
    op_class_name, inputs, outputs, attrs = parse_op_def(src)
    print(f"解析到算子: {op_class_name}")
    print(f"  Inputs: {[i[0] for i in inputs]}")
    print(f"  Outputs: {[o[0] for o in outputs]}")
    print(f"  Attrs: {[a[0] for a in attrs]}")

    # 根据算子类型选择生成函数
    if is_all_gather_matmul(op_class_name):
        print("  类型: AllGatherMatmul 类算子")
        code = gen_all_gather_matmul_like_code(op_class_name, inputs, outputs, attrs)
    elif is_mc2_matmul_like(attrs):
        print("  类型: MC2 Matmul 类算子")
        code = gen_mc2_matmul_like_code(op_class_name, inputs, outputs, attrs)
    else:
        print("  类型: DistributeBarrier 类算子")
        code = gen_distribute_barrier_like_code(op_class_name, inputs, outputs, attrs)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(code, encoding="utf-8")
    print(f"生成完成: {out_path}")


def generate_template(state: WorkflowState) -> WorkflowState:
    """
    Workflow 节点包装：
    - 使用 state["def_file_path"] 作为输入 def.cpp
    - 使用 state["template_file_path"] 作为输出模板 cpp
    - 复用与命令行工具完全相同的生成逻辑
    """
    def_path = Path(state["def_file_path"])
    out_path = Path(state["template_file_path"])

    _generate_template_core(def_path, out_path)

    # 把规范化后的路径字符串写回状态，便于后续节点继续使用
    state["template_file_path"] = str(out_path)
    return state


def main():
    if len(sys.argv) != 3:
        print("用法: gen_param_struct_from_def.py <op_def.cpp> <out.cpp>")
        sys.exit(1)

    def_path = Path(sys.argv[1]).resolve()
    out_path = Path(sys.argv[2]).resolve()

    _generate_template_core(def_path, out_path)


if __name__ == "__main__":
    main()
