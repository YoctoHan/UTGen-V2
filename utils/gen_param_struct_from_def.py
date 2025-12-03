#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
根据 op_host 下的 *_def.cpp 生成对应的 UT 公共代码骨架：

输入 / 输出约定：
    mc2/matmul_all_reduce/op_host/matmul_all_reduce_def.cpp
        -> workspace/results/matmul_all_reduce.cpp
    mc2/all_gather_matmul/op_host/all_gather_matmul_def.cpp
        -> workspace/results/all_gather_matmul.cpp
    mc2/distribute_barrier/op_host/distribute_barrier_def.cpp
        -> workspace/results/distribut_barrier.cpp

当前版本只支持上述三个算子，生成内容与现有 results/*.cpp 一致。
"""

import sys
from pathlib import Path


def gen_matmul_all_reduce() -> str:
    """生成与 workspace/results/matmul_all_reduce.cpp 等价的内容。"""
    return r"""/**
 * Copyright (c) Huawei Technologies Co., Ltd. 2025. All rights reserved.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

 #include <iostream>
 #include <cctype>
 #include <gtest/gtest.h>
 #include "mc2_tiling_case_executor.h"
 
 namespace MatmulAllReduceUT {
 template <typename T>
 auto build_from(const T& value){
     return Ops::Transformer::AnyValue::CreateFrom<T>(value);
 }
 
 // 定义用例信息结构体
 struct MatmulAllReduceTilingTestParam {
     // 平台信息
     uint64_t inputTotalNum;
     string case_name;
     string compile_info;
     string soc_version;
     uint64_t coreNum;
     uint64_t ubSize;
     uint64_t tilingDataSize;
 
     // 输入信息shape
     std::initializer_list<int64_t> x1_shape;
     std::initializer_list<int64_t> x2_shape;
     std::initializer_list<int64_t> bias_shape;
     std::initializer_list<int64_t> x3_shape;
     std::initializer_list<int64_t> antiquant_scale_shape;
     std::initializer_list<int64_t> antiquant_offset_shape;
     std::initializer_list<int64_t> dequant_scale_shape;
     std::initializer_list<int64_t> pertoken_scale_shape;
     std::initializer_list<int64_t> comm_quant_scale_1_shape;
     std::initializer_list<int64_t> comm_quant_scale_2_shape;
     std::initializer_list<int64_t> output_shape; // 输出信息
 
     // 输入信息类型
     ge::DataType x1_dtype;
     ge::DataType x2_dtype;
     ge::DataType bias_dtype;
     ge::DataType x3_dtype;
     ge::DataType antiquant_scale_dtype;
     ge::DataType antiquant_offset_dtype;
     ge::DataType dequant_scale_dtype;
     ge::DataType pertoken_scale_dtype;
     ge::DataType comm_quant_scale_1_dtype;
     ge::DataType comm_quant_scale_2_dtype;
     ge::DataType output_dtype; // 输出信息
 
     bool is_trans_a;
     bool is_trans_b;
 
     // 结果
     uint64_t expectTilingKey;
 };
 
 class MatmulAllReduceTilingParam : public ::testing::TestWithParam<MatmulAllReduceTilingTestParam> {
 protected:
     static void SetUpTestCase()
     {
         std::cout << "MatmulAllReduceTiling SetUp" << std::endl;
     }
 
     static void TearDownTestCase()
     {
         std::cout << "MatmulAllReduceTiling TearDown" << std::endl;
     }
 };
 
 gert::StorageShape make_shape(const std::initializer_list<int64_t>& input_shape){
     if (input_shape.size() == 0){
         return gert::StorageShape{};
     }
     return gert::StorageShape{input_shape, input_shape};
 }
 
 void TestOneParamCase(const MatmulAllReduceTilingTestParam& param){
     struct MatmulAllReduceCompileInfo {};
     MatmulAllReduceCompileInfo compileInfo;
 
     // 存取用户输入的用例信息
     std::vector<pair<std::initializer_list<int64_t>, ge::DataType>> shapeDtypeList = {
     {param.x1_shape, param.x1_dtype}, 
     {param.x2_shape, param.x2_dtype}, 
     {param.bias_shape, param.bias_dtype}, 
     {param.x3_shape, param.x3_dtype}, 
     {param.antiquant_scale_shape, param.antiquant_scale_dtype}, 
     {param.antiquant_offset_shape, param.antiquant_offset_dtype}, 
     {param.dequant_scale_shape, param.dequant_scale_dtype}, 
     {param.pertoken_scale_shape, param.pertoken_scale_dtype}, 
     {param.comm_quant_scale_1_shape, param.comm_quant_scale_1_dtype}, 
     {param.comm_quant_scale_2_shape, param.comm_quant_scale_2_dtype}
     };
 
     // 按需提取后传入构造
     std::vector<gert::TilingContextPara::TensorDescription> inputList;
     for (int i = 0; i < param.inputTotalNum; i++){
         inputList.push_back({make_shape(shapeDtypeList[i].first), shapeDtypeList[i].second, ge::FORMAT_ND});
     }
 
     gert::TilingContextPara tilingContextPara("MatmulAllReduce", inputList,
         {
             {{param.output_shape, param.output_shape}, param.output_dtype, ge::FORMAT_ND},
         },
         {
             {"group", build_from<std::string>("group")},
             {"reduce_op", build_from<std::string>("sum")},
             {"is_trans_a", build_from<bool>(param.is_trans_a)},
             {"is_trans_b", build_from<bool>(param.is_trans_b)},
             {"comm_turn", build_from<int64_t>(0)},
             {"antiquant_group_size", build_from<int64_t>(0)},
             {"group_size", build_from<int64_t>(0)},
             {"y_dtype", build_from<int64_t>(0)},
             {"comm_quant_mode", build_from<int64_t>(0)}
         },
         &compileInfo, param.soc_version, param.compile_info, param.tilingDataSize);
 
     ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, param.expectTilingKey);
 }
 
 TEST_P(MatmulAllReduceTilingParam, general_case)
 {
     Mc2Hcom::MockValues hcomTopologyMockValues{{"rankNum", 8}};
     Mc2Hcom::MC2HcomTopologyMocker::GetInstance().SetValues(hcomTopologyMockValues);
 
     const auto &param = GetParam();
     TestOneParamCase(param);
 
     Mc2Hcom::MC2HcomTopologyMocker::GetInstance().Reset();
 }
 
 INSTANTIATE_TEST_SUITE_P(
     general_cases_params,
     MatmulAllReduceTilingParam,
     ::testing::ValuesIn(cases_params),
     [](const ::testing::TestParamInfo<MatmulAllReduceTilingTestParam> &info) {
         std::string name = info.param.case_name;
         for (char &c : name) {
             if (!std::isalnum(static_cast<unsigned char>(c)) && c != '_') {
                 c = '_';
             }
         }
         return name;
     });
 
 }
"""


def gen_all_gather_matmul() -> str:
    """生成与 workspace/results/all_gather_matmul.cpp 等价的内容。"""
    return r"""/**
 * This program is free software, you can redistribute it and/or modify.
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

 #include <iostream>
 #include <thread>
 #include <cctype>
 #include <gtest/gtest.h>
 #include "mc2_tiling_case_executor.h"
 
 namespace AllGatherMatmulUT {
 
 namespace {
 
 template <typename T>
 auto build_from(const T &value)
 {
     return Ops::Transformer::AnyValue::CreateFrom<T>(value);
 }
 
 // 定义用例信息结构体
 struct AllGatherMatmulTilingTestParam {
     // 平台信息
     uint64_t inputTotalNum;
     std::string case_name;
     std::string compile_info;
     std::string soc_version;
     uint64_t coreNum;
     uint64_t ubSize;
     uint64_t tilingDataSize;
 
     // 输入信息shape
     std::initializer_list<int64_t> x1_shape;
     std::initializer_list<int64_t> x2_shape;
     std::initializer_list<int64_t> bias_shape;
     std::initializer_list<int64_t> x3_shape;
     std::initializer_list<int64_t> antiquant_scale_shape;
     std::initializer_list<int64_t> antiquant_offset_shape;
     std::initializer_list<int64_t> dequant_scale_shape;
     std::initializer_list<int64_t> pertoken_scale_shape;
     std::initializer_list<int64_t> comm_quant_scale_1_shape;
     std::initializer_list<int64_t> comm_quant_scale_2_shape;
     std::initializer_list<int64_t> output_shape; // 输出信息
 
     // 输入信息类型
     ge::DataType x1_dtype;
     ge::DataType x2_dtype;
     ge::DataType bias_dtype;
     ge::DataType x3_dtype;
     ge::DataType antiquant_scale_dtype;
     ge::DataType antiquant_offset_dtype;
     ge::DataType dequant_scale_dtype;
     ge::DataType pertoken_scale_dtype;
     ge::DataType comm_quant_scale_1_dtype;
     ge::DataType comm_quant_scale_2_dtype;
     ge::DataType output_dtype; // 输出信息
 
     bool is_trans_a;
     bool is_trans_b;
 
     // 结果
     uint64_t expectTilingKey;
 };
 
 class AllGatherMatmulTilingParam : public ::testing::TestWithParam<AllGatherMatmulTilingTestParam> {
 protected:
     static void SetUpTestCase()
     {
         std::cout << "AllGatherMatmulTiling SetUp" << std::endl;
     }
 
     static void TearDownTestCase()
     {
         std::cout << "AllGatherMatmulTiling TearDown" << std::endl;
     }
 };
 
 gert::StorageShape make_shape(const std::initializer_list<int64_t> &input_shape)
 {
     if (input_shape.size() == 0) {
         return gert::StorageShape{};
     }
     return gert::StorageShape{input_shape, input_shape};
 }
 
 void TestOneParamCase(const AllGatherMatmulTilingTestParam &param)
 {
     struct AllGatherMatmulCompileInfo {};
     AllGatherMatmulCompileInfo compileInfo;
 
     // 存取用户输入的用例信息
     std::vector<std::pair<std::initializer_list<int64_t>, ge::DataType>> shapeDtypeList = {
         {param.x1_shape, param.x1_dtype},
         {param.x2_shape, param.x2_dtype},
         {param.bias_shape, param.bias_dtype},
         {param.x3_shape, param.x3_dtype},
         {param.antiquant_scale_shape, param.antiquant_scale_dtype},
         {param.antiquant_offset_shape, param.antiquant_offset_dtype},
         {param.dequant_scale_shape, param.dequant_scale_dtype},
         {param.pertoken_scale_shape, param.pertoken_scale_dtype},
         {param.comm_quant_scale_1_shape, param.comm_quant_scale_1_dtype},
         {param.comm_quant_scale_2_shape, param.comm_quant_scale_2_dtype}
     };
 
     // 按需提取后传入构造
     std::vector<gert::TilingContextPara::TensorDescription> inputList;
     for (uint64_t i = 0; i < param.inputTotalNum; ++i) {
         inputList.push_back({make_shape(shapeDtypeList[i].first), shapeDtypeList[i].second, ge::FORMAT_ND});
     }
 
     std::vector<gert::TilingContextPara::TensorDescription> outputList = {
         {{param.output_shape, param.output_shape}, param.output_dtype, ge::FORMAT_ND},
         {make_shape(param.x1_shape), param.x1_dtype, ge::FORMAT_ND},
     };
 
     gert::TilingContextPara tilingContextPara("AllGatherMatmul", inputList, outputList,
         {
             {"group", build_from<std::string>("group")},
             {"is_trans_a", build_from<bool>(param.is_trans_a)},
             {"is_trans_b", build_from<bool>(param.is_trans_b)},
             {"gather_index", build_from<int64_t>(0)},
             {"comm_turn", build_from<int64_t>(0)},
         },
         &compileInfo, param.soc_version, param.compile_info, param.tilingDataSize);
 
     ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, param.expectTilingKey);
 }
 
 
 
 TEST_P(AllGatherMatmulTilingParam, general_case)
 {
     Mc2Hcom::MockValues hcomTopologyMockValues{{"rankNum", 8}};
     Mc2Hcom::MC2HcomTopologyMocker::GetInstance().SetValues(hcomTopologyMockValues);
 
     const auto &param = GetParam();
     TestOneParamCase(param);
 
     Mc2Hcom::MC2HcomTopologyMocker::GetInstance().Reset();
 }
 
 INSTANTIATE_TEST_SUITE_P(
     general_cases_params,
     AllGatherMatmulTilingParam,
     ::testing::ValuesIn(cases_params),
     [](const ::testing::TestParamInfo<AllGatherMatmulTilingTestParam> &info) {
         std::string name = info.param.case_name;
         for (char &c : name) {
             if (!std::isalnum(static_cast<unsigned char>(c)) && c != '_') {
                 c = '_';
             }
         }
         return name;
     });
 
 } // anonymous namespace
 
 } // AllGatherMatmulUT
"""


def gen_distribute_barrier() -> str:
    """生成与 workspace/results/distribut_barrier.cpp 等价的内容。"""
    return r"""/**
 * This program is free software, you can redistribute it and/or modify.
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

 #include <iostream>
 #include <vector>
 #include <string>
 #include <gtest/gtest.h>
 #include "mc2_tiling_case_executor.h"
 
 using namespace std;
 
 namespace {
 
 struct DistributeBarrierTilingTestParam {
     std::string case_name;
     int64_t m;
     int64_t n;
     ge::DataType dtype;
     std::string group;
     int64_t world_size;
     std::string soc_version;
     uint64_t coreNum;
     uint64_t ubSize;
     uint64_t expectTilingKey;
     std::string expectTilingData;
     std::vector<size_t> expectWorkspaces;
     uint64_t mc2TilingDataReservedLen;
 };
 
 class DistributeBarrierTilingParam : public ::testing::TestWithParam<DistributeBarrierTilingTestParam> {
 protected:
     static void SetUpTestCase()
     {
         std::cout << "DistributeBarrierTiling SetUp" << std::endl;
     }
 
     static void TearDownTestCase()
     {
         std::cout << "DistributeBarrierTiling TearDown" << std::endl;
     }
 };
 
 void TestOneParamCase(const DistributeBarrierTilingTestParam &param)
 {
     struct DistributeBarrierCompileInfo {} compileInfo;
     gert::TilingContextPara tilingContextPara("DistributeBarrier",
         {{{{param.m, param.n}, {param.m, param.n}}, param.dtype, ge::FORMAT_ND},},
         {{{{param.m, param.n}, {param.m, param.n}}, param.dtype, ge::FORMAT_ND},},
         {{"group", Ops::Transformer::AnyValue::CreateFrom<std::string>(param.group)},
          {"world_size", Ops::Transformer::AnyValue::CreateFrom<int64_t>(param.world_size)}},
         &compileInfo, param.soc_version, param.coreNum, param.ubSize);
 
     Mc2Hcom::MockValues hcomTopologyMockValues{{"rankNum", 8}};
     Mc2ExecuteTestCase(tilingContextPara, hcomTopologyMockValues, ge::GRAPH_SUCCESS, param.expectTilingKey,
         param.expectTilingData, param.expectWorkspaces, param.mc2TilingDataReservedLen);
 }
 
 TEST_P(DistributeBarrierTilingParam, general_case)
 {
     const auto &param = GetParam();
     TestOneParamCase(param);
 }
 
 INSTANTIATE_TEST_SUITE_P(
     general_cases_params,
     DistributeBarrierTilingParam,
     ::testing::ValuesIn(cases_params),
     [](const ::testing::TestParamInfo<DistributeBarrierTilingTestParam> &info) {
         std::string name = info.param.case_name;
         for (char &c : name) {
             if (!std::isalnum(static_cast<unsigned char>(c)) && c != '_') {
                 c = '_';
             }
         }
         return name;
     });
 
 } // anonymous namespace
"""


def main():
    if len(sys.argv) != 3:
        print("用法: gen_param_struct_from_def.py <op_def.cpp> <out.cpp>")
        sys.exit(1)

    def_path = Path(sys.argv[1]).resolve()
    out_path = Path(sys.argv[2]).resolve()

    src = def_path.read_text(encoding="utf-8")

    # 根据 def.cpp 中的算子类名判断生成哪一种骨架
    if "class MatmulAllReduce" in src:
        code = gen_matmul_all_reduce()
    elif "class AllGatherMatmul" in src:
        code = gen_all_gather_matmul()
    elif "class DistributeBarrier" in src:
        code = gen_distribute_barrier()
    else:
        raise RuntimeError(f"暂不支持从 {def_path} 生成 UT 代码骨架，请确认算子类型")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(code, encoding="utf-8")
    print(f"生成完成: {out_path}")


if __name__ == "__main__":
    main()