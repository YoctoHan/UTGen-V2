/**
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

const string COMPILE_INFO = R"({"hardware_info": {"BT_SIZE": 0, "load3d_constraints": "1", "Intrinsic_fix_pipe_l0c2out": false, "Intrinsic_data_move_l12ub": true, "Intrinsic_data_move_l0c2ub": true, "Intrinsic_data_move_out2l1_nd2nz": false, "UB_SIZE": 196608, "L2_SIZE": 33554432, "L1_SIZE": 524288, "L0A_SIZE": 65536, "L0B_SIZE": 65536, "L0C_SIZE": 131072, "CORE_NUM": 20, "socVersion": "Ascend910B"}})";

// 用例列表集
MatmulAllReduceTilingTestParam cases_params[] = {
{4,"matmul_all_reduce_test_tiling_float16_empty_k",COMPILE_INFO,"Ascend910B",20,196608,4096,{256, 0},{0, 8192},{},{},{},{},{},{},{},{},{256, 8192},ge::DT_FLOAT16,ge::DT_FLOAT16,ge::DT_FLOAT16,ge::DT_FLOAT16,ge::DT_FLOAT,ge::DT_FLOAT,ge::DT_FLOAT,ge::DT_FLOAT,ge::DT_FLOAT,ge::DT_FLOAT,ge::DT_FLOAT16,false,false,10000000000000000009UL},
{4,"matmul_all_reduce_test_tiling_bfloat16",COMPILE_INFO,"Ascend910B",20,196608,4096,{8192, 1536},{1536, 12288},{12288},{},{},{},{},{},{},{},{8192, 12288},ge::DT_FLOAT16,ge::DT_FLOAT16,ge::DT_FLOAT16,ge::DT_FLOAT16,ge::DT_FLOAT,ge::DT_FLOAT,ge::DT_FLOAT,ge::DT_FLOAT,ge::DT_FLOAT,ge::DT_FLOAT,ge::DT_FLOAT16,false,false,10000000000000001100UL},
{4,"matmul_all_reduce_test_tiling_float16_support_3_dim",COMPILE_INFO,"Ascend910B",20,196608,4096,{1, 8192, 1536},{1536, 12288},{12288},{},{},{},{},{},{},{},{1, 8192, 12288},ge::DT_FLOAT16,ge::DT_FLOAT16,ge::DT_FLOAT16,ge::DT_FLOAT16,ge::DT_FLOAT,ge::DT_FLOAT,ge::DT_FLOAT,ge::DT_FLOAT,ge::DT_FLOAT,ge::DT_FLOAT,ge::DT_FLOAT16,false,false,10000000000000001100UL},
{4,"matmul_all_reduce_test_tiling_float16_5",COMPILE_INFO,"Ascend910B",20,196608,4096,{256, 1536},{1536, 8192},{},{},{},{},{},{},{},{},{8192, 12288},ge::DT_FLOAT16,ge::DT_FLOAT16,ge::DT_FLOAT16,ge::DT_FLOAT16,ge::DT_FLOAT,ge::DT_FLOAT,ge::DT_FLOAT,ge::DT_FLOAT,ge::DT_FLOAT,ge::DT_FLOAT,ge::DT_FLOAT16,false,false,10000000000000001100UL},
{4,"matmul_all_reduce_test_tiling_float16_4",COMPILE_INFO,"Ascend910B",20,196608,4096,{1024, 1536},{1536, 8192},{},{},{},{},{},{},{},{},{8192, 12288},ge::DT_FLOAT16,ge::DT_FLOAT16,ge::DT_FLOAT16,ge::DT_FLOAT16,ge::DT_FLOAT,ge::DT_FLOAT,ge::DT_FLOAT,ge::DT_FLOAT,ge::DT_FLOAT,ge::DT_FLOAT,ge::DT_FLOAT16,false,false,10000000000000001100UL},
{4,"matmul_all_reduce_test_tiling_float16_3",COMPILE_INFO,"Ascend910B",20,196608,4096,{128, 1536},{1536, 8192},{},{},{},{},{},{},{},{},{8192, 12288},ge::DT_FLOAT16,ge::DT_FLOAT16,ge::DT_FLOAT16,ge::DT_FLOAT16,ge::DT_FLOAT,ge::DT_FLOAT,ge::DT_FLOAT,ge::DT_FLOAT,ge::DT_FLOAT,ge::DT_FLOAT,ge::DT_FLOAT16,false,false,10000000000000001100UL},
{4,"matmul_all_reduce_test_tiling_float16_2",COMPILE_INFO,"Ascend910B",20,196608,4096,{8192, 1536},{1536, 12288},{},{},{},{},{},{},{},{},{8192, 12288},ge::DT_FLOAT16,ge::DT_FLOAT16,ge::DT_FLOAT16,ge::DT_FLOAT16,ge::DT_FLOAT,ge::DT_FLOAT,ge::DT_FLOAT,ge::DT_FLOAT,ge::DT_FLOAT,ge::DT_FLOAT,ge::DT_FLOAT16,true,true,10000000000000001100UL},
{4,"matmul_all_reduce_test_mcut_float16_910B_win2win",COMPILE_INFO,"Ascend910B",20,196608,4096,{12290, 15360},{15360, 12288},{},{},{},{},{},{},{},{},{12290, 12288},ge::DT_FLOAT16,ge::DT_FLOAT16,ge::DT_FLOAT16,ge::DT_FLOAT16,ge::DT_FLOAT,ge::DT_FLOAT,ge::DT_FLOAT,ge::DT_FLOAT,ge::DT_FLOAT,ge::DT_FLOAT,ge::DT_FLOAT16,false,false,10000000000000001100UL},
{4,"matmul_all_reduce_test_tiling_big_K",COMPILE_INFO,"Ascend910B",20,196608,4096,{8192, 0xFFFFFFF},{0xFFFFFFF, 12288},{},{8192, 12288},{},{},{},{},{},{},{8192, 12288},ge::DT_FLOAT16,ge::DT_FLOAT16,ge::DT_FLOAT16,ge::DT_FLOAT16,ge::DT_FLOAT,ge::DT_FLOAT,ge::DT_FLOAT,ge::DT_FLOAT,ge::DT_FLOAT,ge::DT_FLOAT,ge::DT_FLOAT16,false,false,65536UL},
{4,"matmul_all_reduce_test_tiling_big_N",COMPILE_INFO,"Ascend910B",20,196608,4096,{8192, 1536},{1536, 0xFFFFFFF},{},{8192, 0xFFFFFFF},{},{},{},{},{},{},{8192, 0xFFFFFFF},ge::DT_FLOAT16,ge::DT_FLOAT16,ge::DT_FLOAT16,ge::DT_FLOAT16,ge::DT_FLOAT,ge::DT_FLOAT,ge::DT_FLOAT,ge::DT_FLOAT,ge::DT_FLOAT,ge::DT_FLOAT,ge::DT_FLOAT16,false,false,65536UL},
{4,"matmul_all_reduce_test_tiling_float16_unaligned",COMPILE_INFO,"Ascend910B",20,196608,4096,{1, 65536},{65536, 128},{},{},{},{},{},{},{},{},{1, 128},ge::DT_FLOAT16,ge::DT_FLOAT16,ge::DT_FLOAT16,ge::DT_FLOAT16,ge::DT_FLOAT,ge::DT_FLOAT,ge::DT_FLOAT,ge::DT_FLOAT,ge::DT_FLOAT,ge::DT_FLOAT,ge::DT_FLOAT16,false,false,10000000000000001100UL},
{4,"matmul_all_reduce_test_tiling_float16_1_cube",COMPILE_INFO,"Ascend910B",20,196608,4096,{8192, 1536},{1536, 12288},{},{},{},{},{},{},{},{},{8192, 12288},ge::DT_FLOAT16,ge::DT_FLOAT16,ge::DT_FLOAT16,ge::DT_FLOAT16,ge::DT_FLOAT,ge::DT_FLOAT,ge::DT_FLOAT,ge::DT_FLOAT,ge::DT_FLOAT,ge::DT_FLOAT,ge::DT_FLOAT16,false,false,10000000000000001100UL},
{4,"matmul_all_reduce_test_tiling_float16_1",COMPILE_INFO,"Ascend910B",20,196608,4096,{8192, 1536},{1536, 12288},{},{8192, 12288},{},{},{},{},{},{},{8192, 12288},ge::DT_FLOAT16,ge::DT_FLOAT16,ge::DT_FLOAT16,ge::DT_FLOAT16,ge::DT_FLOAT,ge::DT_FLOAT,ge::DT_FLOAT,ge::DT_FLOAT,ge::DT_FLOAT,ge::DT_FLOAT,ge::DT_FLOAT16,false,false,65536UL},
{8,"matmul_all_reduce_test_tiling_int8_bf16",COMPILE_INFO,"Ascend910B",20,196608,4096,{256, 1536},{1536, 8192},{},{},{},{},{8192},{},{},{},{256, 8192},ge::DT_INT8,ge::DT_INT8,ge::DT_BF16,ge::DT_BF16,ge::DT_BF16,ge::DT_BF16,ge::DT_BF16,ge::DT_BF16,ge::DT_FLOAT,ge::DT_FLOAT,ge::DT_BF16,false,false,0UL},
{8,"matmul_all_reduce_test_tiling_int8_1",COMPILE_INFO,"Ascend910B",20,196608,4096,{256, 1536},{1536, 8192},{},{},{},{},{8192},{},{},{},{256, 8192},ge::DT_INT8,ge::DT_INT8,ge::DT_FLOAT16,ge::DT_FLOAT16,ge::DT_FLOAT16,ge::DT_FLOAT16,ge::DT_UINT64,ge::DT_FLOAT16,ge::DT_FLOAT,ge::DT_FLOAT,ge::DT_FLOAT16,false,false,0UL},
{9,"matmul_all_reduce_test_tiling_int8_2",COMPILE_INFO,"Ascend910B",20,196608,4096,{256, 1536},{1536, 8192},{},{},{},{},{1},{256},{},{},{256, 8192},ge::DT_INT8,ge::DT_INT8,ge::DT_FLOAT16,ge::DT_FLOAT16,ge::DT_FLOAT16,ge::DT_FLOAT16,ge::DT_FLOAT,ge::DT_FLOAT,ge::DT_FLOAT16,ge::DT_FLOAT,ge::DT_FLOAT16,false,false,16UL},
{10,"matmul_all_reduce_test_tiling_a8w8_910b_mCut_2",COMPILE_INFO,"Ascend910B",20,196608,4096,{4096, 1024},{1024, 8192},{},{},{},{},{8192},{},{8192},{8192},{4096, 8192},ge::DT_INT8,ge::DT_INT8,ge::DT_INT32,ge::DT_FLOAT16,ge::DT_FLOAT16,ge::DT_FLOAT16,ge::DT_UINT64,ge::DT_UINT64,ge::DT_FLOAT16,ge::DT_FLOAT16,ge::DT_FLOAT16,false,false,10UL},
{10,"matmul_all_reduce_test_tiling_a8w8_910b_mCut_1",COMPILE_INFO,"Ascend910B",20,196608,4096,{4096, 6272},{6272, 8192},{},{},{},{},{8192},{},{8192},{8192},{4096, 8192},ge::DT_INT8,ge::DT_INT8,ge::DT_INT32,ge::DT_FLOAT16,ge::DT_FLOAT16,ge::DT_FLOAT16,ge::DT_UINT64,ge::DT_UINT64,ge::DT_FLOAT16,ge::DT_FLOAT16,ge::DT_FLOAT16,false,false,10UL},
{10,"matmul_all_reduce_test_tiling_a8w8_scaleDimNum2_910b",COMPILE_INFO,"Ascend910B",20,196608,4096,{256, 1536},{1536, 8192},{},{},{},{},{1, 8192},{},{1, 8192},{1, 8192},{256, 8192},ge::DT_INT8,ge::DT_INT8,ge::DT_INT32,ge::DT_FLOAT16,ge::DT_FLOAT16,ge::DT_FLOAT16,ge::DT_UINT64,ge::DT_UINT64,ge::DT_FLOAT16,ge::DT_FLOAT16,ge::DT_FLOAT16,false,false,10UL},
{10,"matmul_all_reduce_test_tiling_a8w8_910b",COMPILE_INFO,"Ascend910B",20,196608,4096,{256, 1536},{1536, 8192},{},{},{},{},{8192},{},{8192},{8192},{256, 8192},ge::DT_INT8,ge::DT_INT8,ge::DT_INT32,ge::DT_FLOAT16,ge::DT_FLOAT16,ge::DT_FLOAT16,ge::DT_UINT64,ge::DT_UINT64,ge::DT_FLOAT16,ge::DT_FLOAT16,ge::DT_FLOAT16,false,false,10UL},
};

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

} // namespace MatmulAllReduceUT