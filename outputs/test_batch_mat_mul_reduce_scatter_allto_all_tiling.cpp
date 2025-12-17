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
#include <map>
#include <vector>
#include <string>
#include <float.h>
#include <array>
#include <cctype>
#include <gtest/gtest.h>
#include <opdev/platform.h>
#include <gmock/gmock.h>
#include "mc2_tiling_case_executor.h"

namespace BatchMatMulReduceScatterAlltoAllUT {

namespace {

template <typename T>
auto build_from(const T &value)
{
    return Ops::Transformer::AnyValue::CreateFrom<T>(value);
}

struct BatchMatMulReduceScatterAlltoAllTilingTestParam {
    uint64_t inputTotalNum;
    std::string case_name;
    uint64_t coreNum;
    uint64_t ubSize;

    std::initializer_list<int64_t> x_shape;
    std::initializer_list<int64_t> w_shape;
    std::initializer_list<int64_t> bias_shape;

    std::initializer_list<int64_t> y_shape;

    ge::DataType x_dtype;
    ge::DataType w_dtype;
    ge::DataType bias_dtype;
    ge::DataType y_dtype;

    std::string group_ep;
    std::string group_tp;
    int64_t ep_world_size;
    int64_t tp_world_size;
    int64_t y_shard_type;
    bool transpose_weight;

    bool hasExpectTilingKey;
    uint64_t expectTilingKey;
};

class BatchMatMulReduceScatterAlltoAllTilingParam
    : public ::testing::TestWithParam<BatchMatMulReduceScatterAlltoAllTilingTestParam> {
protected:
    static void SetUpTestCase()
    {
        std::cout << "BatchMatMulReduceScatterAlltoAllTiling SetUp" << std::endl;
    }

    static void TearDownTestCase()
    {
        std::cout << "BatchMatMulReduceScatterAlltoAllTiling TearDown" << std::endl;
    }
};

gert::StorageShape make_shape(const std::initializer_list<int64_t> &input_shape)
{
    if (input_shape.size() == 0) {
        return gert::StorageShape{};
    }
    return gert::StorageShape{input_shape, input_shape};
}

void TestOneParamCase(const BatchMatMulReduceScatterAlltoAllTilingTestParam &param)
{
    struct BatchMatMulReduceScatterAlltoAllCompileInfo {};
    BatchMatMulReduceScatterAlltoAllCompileInfo compileInfo;

    std::vector<std::pair<std::initializer_list<int64_t>, ge::DataType>> shapeDtypeList = {
        {param.x_shape, param.x_dtype},
        {param.w_shape, param.w_dtype},
        {param.bias_shape, param.bias_dtype},
    };

    std::vector<gert::TilingContextPara::TensorDescription> inputList;
    for (uint64_t i = 0; i < param.inputTotalNum; ++i) {
        inputList.push_back({make_shape(shapeDtypeList[i].first), shapeDtypeList[i].second, ge::FORMAT_ND});
    }

    std::vector<gert::TilingContextPara::TensorDescription> outputList = {
        {make_shape(param.y_shape), param.y_dtype, ge::FORMAT_ND}
    };

    gert::TilingContextPara tilingContextPara(
        "BatchMatMulReduceScatterAlltoAll",
        inputList,
        outputList,
        {
            {"group_ep", build_from<std::string>(param.group_ep)},
            {"group_tp", build_from<std::string>(param.group_tp)},
            {"ep_world_size", build_from<int64_t>(param.ep_world_size)},
            {"tp_world_size", build_from<int64_t>(param.tp_world_size)},
            {"y_shard_type", build_from<int64_t>(param.y_shard_type)},
            {"transpose_weight", build_from<bool>(param.transpose_weight)},
        },
        &compileInfo, "Ascend910_93", param.coreNum, param.ubSize);

    Mc2Hcom::MockValues hcomTopologyMockValues{{"rankNum", 8}};
    if (param.hasExpectTilingKey) {
        Mc2ExecuteTestCase(tilingContextPara, hcomTopologyMockValues, ge::GRAPH_SUCCESS, param.expectTilingKey);
    } else {
        Mc2ExecuteTestCase(tilingContextPara, hcomTopologyMockValues);
    }
}

BatchMatMulReduceScatterAlltoAllTilingTestParam cases_params[] = {
    {2, "batch_matmul_reduce_scatter_all_to_all_test_tiling_float16_1", 20, 196608, {2, 1024, 64}, {2, 64, 128}, {}, {16, 128, 64}, ge::DT_FLOAT16, ge::DT_FLOAT16, ge::DT_FLOAT16, ge::DT_FLOAT16, "ep_group", "tp_group", 8, 2, 1, false, true, 1000000000000001001UL},
    {2, "batch_matmul_reduce_scatter_all_to_all_test_tiling_float16_1_weight_trans", 20, 196608, {2, 1024, 64}, {2, 128, 64}, {}, {16, 128, 64}, ge::DT_FLOAT16, ge::DT_FLOAT16, ge::DT_FLOAT16, ge::DT_FLOAT16, "ep_group", "tp_group", 8, 2, 1, true, true, 1000000000000001011UL},
    {2, "batch_matmul_reduce_scatter_all_to_all_test_tiling_M_0", 20, 196608, {2, 1024, 0}, {2, 0, 128}, {}, {16, 128, 64}, ge::DT_FLOAT16, ge::DT_FLOAT16, ge::DT_FLOAT16, ge::DT_FLOAT16, "ep_group", "tp_group", 8, 2, 1, false, false, 0UL},
    {2, "batch_matmul_reduce_scatter_all_to_all_test_tiling_float16_shard", 20, 196608, {2, 1024, 64}, {2, 64, 128}, {}, {16, 128, 64}, ge::DT_FLOAT16, ge::DT_FLOAT16, ge::DT_FLOAT16, ge::DT_FLOAT16, "ep_group", "tp_group", 8, 2, 1, false, true, 1000000000000001001UL},
    {3, "batch_matmul_reduce_scatter_all_to_all_test_tiling_float16_shard_with_bias", 20, 196608, {2, 1024, 64}, {2, 64, 128}, {2, 1, 128}, {16, 128, 64}, ge::DT_FLOAT16, ge::DT_FLOAT16, ge::DT_FLOAT16, ge::DT_FLOAT16, "ep_group", "tp_group", 8, 2, 1, false, true, 1000000000000001101UL},
    {2, "batch_matmul_reduce_scatter_all_to_all_test_tiling_float16_shard_0", 20, 196608, {2, 1024, 64}, {2, 64, 128}, {}, {16, 128, 64}, ge::DT_FLOAT16, ge::DT_FLOAT16, ge::DT_FLOAT16, ge::DT_FLOAT16, "ep_group", "ep_group", 8, 2, 0, false, false, 0UL},
    {3, "batch_matmul_reduce_scatter_all_to_all_test_tiling_float16_shard_with_bias_test1", 20, 196608, {2, 1024, 64}, {2, 64, 128}, {2, 1, 128}, {16, 128, 64}, ge::DT_FLOAT16, ge::DT_FLOAT16, ge::DT_FLOAT16, ge::DT_FLOAT16, "ep_group", "tp_group", 3, 2, 1, false, false, 0UL},
    {3, "batch_matmul_reduce_scatter_all_to_all_test_tiling_float16_shard_with_bias_test2", 20, 196608, {2, 1024, 64}, {2, 64, 128}, {2, 1, 128}, {16, 128, 64}, ge::DT_FLOAT16, ge::DT_FLOAT16, ge::DT_FLOAT16, ge::DT_FLOAT16, "ep_group", "ep_group", 8, 2, 1, false, false, 0UL},
    {3, "batch_matmul_reduce_scatter_all_to_all_test_tiling_float16_shard_with_bias_test3", 20, 196608, {1, 1024, 64}, {2, 64, 128}, {2, 1, 128}, {16, 128, 64}, ge::DT_FLOAT16, ge::DT_FLOAT16, ge::DT_FLOAT16, ge::DT_FLOAT16, "ep_group", "tp_group", 8, 2, 1, false, false, 0UL},
    {3, "batch_matmul_reduce_scatter_all_to_all_test_tiling_float16_shard_with_bias_test4", 20, 196608, {2, 1024, 64}, {2, 64, 128}, {2, 1, 128, 1}, {16, 128, 64}, ge::DT_FLOAT16, ge::DT_FLOAT16, ge::DT_FLOAT16, ge::DT_FLOAT16, "ep_group", "tp_group", 8, 2, 1, false, false, 0UL},
    {3, "batch_matmul_reduce_scatter_all_to_all_test_tiling_float16_shard_with_bias_test5", 20, 196608, {}, {2, 64, 128}, {2, 1, 128}, {16, 128, 64}, ge::DT_FLOAT16, ge::DT_FLOAT16, ge::DT_FLOAT16, ge::DT_FLOAT16, "ep_group", "tp_group", 8, 2, 1, false, false, 0UL},
    {3, "batch_matmul_reduce_scatter_all_to_all_test_tiling_float16_shard_with_bias_test6", 20, 196608, {1024, 64}, {2, 64, 128}, {2, 1, 128}, {16, 128, 64}, ge::DT_FLOAT16, ge::DT_FLOAT16, ge::DT_FLOAT16, ge::DT_FLOAT16, "ep_group", "tp_group", 8, 2, 1, false, false, 0UL},
    {3, "batch_matmul_reduce_scatter_all_to_all_test_tiling_float16_shard_with_bias_test7", 20, 196608, {2, 1024, 0}, {2, 64, 128}, {2, 1, 128}, {16, 128, 64}, ge::DT_FLOAT16, ge::DT_FLOAT16, ge::DT_FLOAT16, ge::DT_FLOAT16, "ep_group", "tp_group", 8, 2, 1, false, false, 0UL},
    {3, "batch_matmul_reduce_scatter_all_to_all_test_tiling_float16_shard_with_bias_test8", 20, 196608, {2, 1024, 64}, {2, 64, 128}, {3, 1, 128}, {16, 128, 64}, ge::DT_FLOAT16, ge::DT_FLOAT16, ge::DT_FLOAT16, ge::DT_FLOAT16, "ep_group", "tp_group", 8, 2, 1, false, false, 0UL},
    {3, "batch_matmul_reduce_scatter_all_to_all_test_tiling_fp16_shard0_with_bias", 20, 196608, {2, 1024, 64}, {2, 64, 128}, {2, 1, 64}, {16, 128, 64}, ge::DT_FLOAT16, ge::DT_FLOAT16, ge::DT_FLOAT16, ge::DT_FLOAT16, "ep_group", "tp_group", 8, 2, 0, false, true, 1000000000000000100UL},
    {3, "batch_matmul_reduce_scatter_all_to_all_test_tiling_fp16_shard0_nonlocalE_tail_front", 20, 196608, {17, 3868, 637}, {17, 637, 2366}, {17, 1, 1183}, {68, 967, 1183}, ge::DT_FLOAT16, ge::DT_FLOAT16, ge::DT_FLOAT16, ge::DT_FLOAT16, "ep_group", "tp_group", 4, 2, 0, false, true, 1000000000000000100UL},
    {3, "batch_matmul_reduce_scatter_all_to_all_test_tiling_bf16_shard0_with_bias", 20, 196608, {2, 1024, 64}, {2, 64, 128}, {2, 1, 64}, {16, 128, 64}, ge::DT_BF16, ge::DT_BF16, ge::DT_FLOAT, ge::DT_BF16, "ep_group", "tp_group", 8, 2, 0, false, true, 1000000000000000100UL},
    {3, "batch_matmul_reduce_scatter_all_to_all_test_tiling_fp16_shard0_with_bias_invalid_Xshape", 20, 196608, {2, 1020, 64}, {2, 64, 128}, {2, 1, 64}, {16, 128, 64}, ge::DT_FLOAT16, ge::DT_FLOAT16, ge::DT_FLOAT16, ge::DT_FLOAT16, "ep_group", "tp_group", 8, 2, 0, false, false, 0UL},
    {3, "batch_matmul_reduce_scatter_all_to_all_test_tiling_fp16_shard1_with_bias_invalid_Xshape", 20, 196608, {2, 1020, 64}, {2, 64, 128}, {2, 1, 64}, {16, 128, 64}, ge::DT_FLOAT16, ge::DT_FLOAT16, ge::DT_FLOAT16, ge::DT_FLOAT16, "ep_group", "tp_group", 8, 2, 1, false, false, 0UL},
    {3, "batch_matmul_reduce_scatter_all_to_all_test_tiling_fp16_shard0_with_bias_invalid_H", 20, 196608, {2, 1024, 64}, {2, 64, 128}, {2, 1, 128}, {16, 128, 64}, ge::DT_FLOAT16, ge::DT_FLOAT16, ge::DT_FLOAT16, ge::DT_FLOAT16, "ep_group", "tp_group", 8, 2, 0, false, false, 0UL},
};

TEST_P(BatchMatMulReduceScatterAlltoAllTilingParam, general_case)
{
const auto &param = GetParam();
    TestOneParamCase(param);
}

INSTANTIATE_TEST_SUITE_P(
    general_cases_params,
    BatchMatMulReduceScatterAlltoAllTilingParam,
    ::testing::ValuesIn(cases_params),
    [](const ::testing::TestParamInfo<BatchMatMulReduceScatterAlltoAllTilingTestParam> &info) {
        std::string name = info.param.case_name;
        for (char &c : name) {
            if (!std::isalnum(static_cast<unsigned char>(c)) && c != '_') {
                c = '_';
            }
        }
        return name;
    });

} // anonymous namespace

} // namespace BatchMatMulReduceScatterAlltoAllUT
