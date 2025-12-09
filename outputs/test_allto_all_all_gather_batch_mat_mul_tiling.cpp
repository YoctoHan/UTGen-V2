/**
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
#include <cctype>

#include <gtest/gtest.h>
#include "mc2_tiling_case_executor.h"

namespace AlltoAllAllGatherBatchMatMulUT {

namespace {

template <typename T>
auto build_from(const T &value)
{
    return Ops::Transformer::AnyValue::CreateFrom<T>(value);
}

// 用例信息结构体
struct AlltoAllAllGatherBmmTilingTestParam {
    // 平台信息
    std::string case_name;
    std::string soc_version;
    uint64_t coreNum;
    uint64_t ubSize;
    uint64_t tilingDataSize;

    // 输入 / 输出 shape & dtype
    std::vector<std::vector<int64_t>> input_shapes;
    std::vector<ge::DataType> input_dtypes;
    std::vector<int64_t> output_shape;
    ge::DataType output_dtype;

    // Attr 信息
    std::string group_ep;
    std::string group_tp;
    int64_t ep_world_size;
    int64_t tp_world_size;
    int64_t x_shard_type;
    int64_t act_type;
    bool transpose_weight;
    bool output_y2_flag;
    bool output_y3_flag;

    // 期望结果
    bool has_expect_tiling_key;
    uint64_t expect_tiling_key;
};

class AlltoAllAllGatherBmmTilingParam : public ::testing::TestWithParam<AlltoAllAllGatherBmmTilingTestParam> {
protected:
    static void SetUpTestCase()
    {
        std::cout << "AlltoAllAllGatherBmmTiling SetUp" << std::endl;
    }

    static void TearDownTestCase()
    {
        std::cout << "AlltoAllAllGatherBmmTiling TearDown" << std::endl;
    }
};

gert::StorageShape make_shape(const std::vector<int64_t> &input_shape)
{
    gert::StorageShape storage_shape;
    if (input_shape.empty()) {
        return storage_shape;
    }
    auto &origin_shape = storage_shape.MutableOriginShape();
    auto &runtime_shape = storage_shape.MutableStorageShape();
    for (auto dim : input_shape) {
        origin_shape.AppendDim(dim);
        runtime_shape.AppendDim(dim);
    }
    return storage_shape;
}

void TestOneParamCase(const AlltoAllAllGatherBmmTilingTestParam &param)
{
    struct DistributeBarrierCompileInfo {};
    DistributeBarrierCompileInfo compileInfo;

    std::vector<gert::TilingContextPara::TensorDescription> inputList;
    inputList.reserve(param.input_shapes.size());
    for (size_t i = 0; i < param.input_shapes.size(); ++i) {
        inputList.push_back({make_shape(param.input_shapes[i]), param.input_dtypes[i], ge::FORMAT_ND});
    }

    std::vector<gert::TilingContextPara::TensorDescription> outputList = {
        {make_shape(param.output_shape), param.output_dtype, ge::FORMAT_ND},
    };

    gert::TilingContextPara tilingContextPara("AlltoAllAllGatherBatchMatMul",
        inputList,
        outputList,
        {
            {"group_ep", build_from<std::string>(param.group_ep)},
            {"group_tp", build_from<std::string>(param.group_tp)},
            {"ep_world_size", build_from<int64_t>(param.ep_world_size)},
            {"tp_world_size", build_from<int64_t>(param.tp_world_size)},
            {"x_shard_type", build_from<int64_t>(param.x_shard_type)},
            {"act_type", build_from<int64_t>(param.act_type)},
            {"transpose_weight", build_from<bool>(param.transpose_weight)},
            {"output_y2_flag", build_from<bool>(param.output_y2_flag)},
            {"output_y3_flag", build_from<bool>(param.output_y3_flag)},
        },
        &compileInfo,
        param.soc_version,
        param.coreNum,
        param.ubSize,
        param.tilingDataSize);

    Mc2Hcom::MockValues hcomTopologyMockValues{{"rankNum", 8}};
    if (param.has_expect_tiling_key) {
        Mc2ExecuteTestCase(tilingContextPara, hcomTopologyMockValues, ge::GRAPH_SUCCESS, param.expect_tiling_key);
    } else {
        Mc2ExecuteTestCase(tilingContextPara, hcomTopologyMockValues);
    }
}

// 用例列表
AlltoAllAllGatherBmmTilingTestParam cases_params[] = {
    {"all_to_all_all_gather_batch_matmul_test_tiling_float16_1", "Ascend910_93", 20, 196608, 4096, {{16, 128, 64}, {4, 64, 128}}, {ge::DT_FLOAT16, ge::DT_FLOAT16}, {4, 512, 64}, ge::DT_FLOAT16, "ep_group", "tp_group", 4, 2, 1, 0, false, false, false, true, 0xDE0B6B3A7640001},
    {"all_to_all_all_gather_batch_matmul_test_tiling_float16_xshard_0", "Ascend910_93", 20, 196608, 4096, {{16, 256, 32}, {4, 64, 128}}, {ge::DT_FLOAT16, ge::DT_FLOAT16}, {4, 512, 64}, ge::DT_FLOAT16, "ep_group", "tp_group", 4, 2, 0, 0, false, false, false, true, 0xDE0B6B3A7640000},
    {"all_to_all_all_gather_batch_matmul_test_tiling_float16_shard_0_invalid_H", "Ascend910_93", 20, 196608, 4096, {{16, 256, 65536}, {4, 64, 128}, {4, 1, 128}}, {ge::DT_FLOAT16, ge::DT_FLOAT16, ge::DT_FLOAT16}, {4, 512, 64}, ge::DT_FLOAT16, "ep_group", "tp_group", 4, 2, 0, 0, false, false, false, false, 0},
    {"all_to_all_all_gather_batch_matmul_test_tiling_float16_shard_0_unequal_H", "Ascend910_93", 20, 196608, 4096, {{16, 256, 64}, {4, 64, 128}, {4, 1, 128}}, {ge::DT_FLOAT16, ge::DT_FLOAT16, ge::DT_FLOAT16}, {4, 512, 64}, ge::DT_FLOAT16, "ep_group", "tp_group", 4, 2, 0, 0, false, false, false, false, 0},
    {"all_to_all_all_gather_batch_matmul_test_tiling_float16_1_weight_trans", "Ascend910_93", 20, 196608, 4096, {{16, 128, 64}, {4, 128, 64}}, {ge::DT_FLOAT16, ge::DT_FLOAT16}, {4, 512, 64}, ge::DT_FLOAT16, "ep_group", "tp_group", 4, 2, 1, 0, true, false, false, true, 0xDE0B6B3A764000B},
    {"all_to_all_all_gather_batch_matmul_test_tiling_float16_xShard_1_actType_1", "Ascend910_93", 20, 196608, 4096, {{16, 128, 64}, {4, 64, 128}}, {ge::DT_FLOAT16, ge::DT_FLOAT16}, {4, 512, 64}, ge::DT_FLOAT16, "ep_group", "tp_group", 4, 2, 1, 1, false, false, false, true, 0xDE0B6B3A7640001},
    {"all_to_all_all_gather_batch_matmul_test_tiling_float16_xShard_1_actType_4", "Ascend910_93", 20, 196608, 4096, {{16, 128, 64}, {4, 64, 128}}, {ge::DT_FLOAT16, ge::DT_FLOAT16}, {4, 512, 64}, ge::DT_FLOAT16, "ep_group", "tp_group", 4, 2, 1, 4, false, false, false, true, 0xDE0B6B3A7640001},
    {"all_to_all_all_gather_batch_matmul_test_tiling_float16_invalid_E", "Ascend910_93", 20, 196608, 4096, {{32, 128, 64}, {4, 64, 128}}, {ge::DT_FLOAT16, ge::DT_FLOAT16}, {4, 512, 64}, ge::DT_FLOAT16, "ep_group", "tp_group", 4, 2, 1, 1, false, false, false, false, 0},
    {"all_to_all_all_gather_batch_matmul_test_tiling_float16_shard", "Ascend910_93", 20, 196608, 4096, {{16, 128, 64}, {4, 64, 128}}, {ge::DT_FLOAT16, ge::DT_FLOAT16}, {4, 512, 64}, ge::DT_FLOAT16, "ep_group", "tp_group", 4, 2, 1, 0, false, false, false, true, 0xDE0B6B3A7640001},
    {"all_to_all_all_gather_batch_matmul_test_tiling_invalid_EOverep_intercept", "Ascend910_93", 20, 196608, 4096, {{160, 128, 64}, {40, 128, 64}}, {ge::DT_FLOAT16, ge::DT_FLOAT16}, {40, 512, 64}, ge::DT_FLOAT16, "ep_group", "tp_group", 4, 2, 1, 0, false, false, false, false, 0},
    {"all_to_all_all_gather_batch_matmul_test_tiling_float16_shard_with_bias", "Ascend910_93", 20, 196608, 4096, {{16, 128, 64}, {4, 64, 128}, {4, 1, 128}}, {ge::DT_FLOAT16, ge::DT_FLOAT16, ge::DT_FLOAT16}, {4, 512, 64}, ge::DT_FLOAT16, "ep_group", "tp_group", 4, 2, 1, 0, false, false, false, true, 0xDE0B6B3A7640065},
    {"all_to_all_all_gather_batch_matmul_test_tiling_float16_shard_with_bias_bf16", "Ascend910_93", 20, 196608, 4096, {{16, 128, 64}, {4, 64, 128}, {4, 1, 128}}, {ge::DT_FLOAT16, ge::DT_FLOAT16, ge::DT_FLOAT}, {4, 512, 64}, ge::DT_FLOAT16, "ep_group", "tp_group", 4, 2, 1, 0, false, false, false, true, 0xDE0B6B3A7640065},
    {"all_to_all_all_gather_batch_matmul_test_tiling_float16_shard_0", "Ascend910_93", 20, 196608, 4096, {{16, 256, 32}, {4, 64, 128}}, {ge::DT_FLOAT16, ge::DT_FLOAT16}, {4, 512, 64}, ge::DT_FLOAT16, "ep_group", "tp_group", 4, 2, 0, 0, false, true, false, false, 0},
    {"all_to_all_all_gather_batch_matmul_test_tiling_float16_shard_1", "Ascend910_93", 20, 196608, 4096, {{16, 128, 64}, {4, 64, 128}}, {ge::DT_FLOAT16, ge::DT_FLOAT16}, {4, 512, 64}, ge::DT_FLOAT16, "ep_group", "tp_group", 4, 2, 1, 0, false, false, true, false, 0},
    {"all_to_all_all_gather_batch_matmul_test_tiling_float16_shard_1_test1", "Ascend910_93", 20, 196608, 4096, {{16, 128, 64}, {4, 64, 128}}, {ge::DT_FLOAT16, ge::DT_FLOAT16}, {4, 512, 64}, ge::DT_FLOAT16, "", "tp_group", 4, 2, 1, 0, false, false, true, false, 0},
    {"all_to_all_all_gather_batch_matmul_test_tiling_float16_shard_with_bias_test1", "Ascend910_93", 20, 196608, 4096, {{16, 128, 64}, {4, 64, 128}, {4, 1, 128}}, {ge::DT_FLOAT16, ge::DT_FLOAT16, ge::DT_FLOAT16}, {4, 512, 64}, ge::DT_FLOAT16, "ep_group", "tp_group", 4, 2, 1, 0, false, true, true, false, 0},
    {"all_to_all_all_gather_batch_matmul_test_tiling_float16_shard_with_bias_test2", "Ascend910_93", 20, 196608, 4096, {{16, 128, 64}, {4, 64, 128}, {4, 1, 128}}, {ge::DT_FLOAT16, ge::DT_FLOAT16, ge::DT_FLOAT16}, {4, 512, 64}, ge::DT_FLOAT16, "ep_group", "tp_group", 4, 2, 1, 0, false, false, true, false, 0},
    {"all_to_all_all_gather_batch_matmul_test_tiling_float16_shard_with_bias_test3", "Ascend910_93", 20, 196608, 4096, {{16, 128, 64}, {4, 64, 128}, {4, 1, 128}}, {ge::DT_FLOAT16, ge::DT_FLOAT16, ge::DT_FLOAT16}, {4, 512, 64}, ge::DT_FLOAT16, "ep_group", "tp_group", 3, 2, 1, 0, false, false, false, false, 0},
    {"all_to_all_all_gather_batch_matmul_test_tiling_float16_shard_with_bias_test4", "Ascend910_93", 20, 196608, 4096, {{16, 128, 64}, {4, 64, 128}, {4, 1, 128}}, {ge::DT_FLOAT16, ge::DT_FLOAT16, ge::DT_FLOAT16}, {4, 512, 64}, ge::DT_FLOAT16, "ep_group", "tp_group", 4, 2, 1, 9, false, false, false, false, 0},
    {"all_to_all_all_gather_batch_matmul_test_tiling_float16_shard_with_bias_test5", "Ascend910_93", 20, 196608, 4096, {{16, 128, 64}, {4, 64, 128}, {4, 1, 128}}, {ge::DT_FLOAT16, ge::DT_FLOAT16, ge::DT_FLOAT16}, {4, 512, 64}, ge::DT_FLOAT16, "ep_group", "ep_group", 4, 2, 1, 0, false, false, false, false, 0},
    {"all_to_all_all_gather_batch_matmul_test_tiling_float16_shard_with_bias_test6", "Ascend910_93", 20, 196608, 4096, {{1, 128, 64}, {4, 64, 128}, {4, 1, 128}}, {ge::DT_FLOAT16, ge::DT_FLOAT16, ge::DT_FLOAT16}, {4, 512, 64}, ge::DT_FLOAT16, "ep_group", "tp_group", 4, 2, 1, 0, false, false, false, false, 0},
    {"all_to_all_all_gather_batch_matmul_test_tiling_float16_shard_with_bias_test7", "Ascend910_93", 20, 196608, 4096, {{16, 128, 65536}, {4, 64, 128}, {4, 1, 128}}, {ge::DT_FLOAT16, ge::DT_FLOAT16, ge::DT_FLOAT16}, {4, 512, 64}, ge::DT_FLOAT16, "ep_group", "tp_group", 4, 2, 1, 0, false, false, false, false, 0},
    {"all_to_all_all_gather_batch_matmul_test_tiling_float16_shard_with_bias_test8", "Ascend910_93", 20, 196608, 4096, {{}, {4, 64, 128}, {4, 1, 128}}, {ge::DT_FLOAT16, ge::DT_FLOAT16, ge::DT_FLOAT16}, {4, 512, 64}, ge::DT_FLOAT16, "ep_group", "tp_group", 4, 2, 1, 0, false, false, false, false, 0},
    {"all_to_all_all_gather_batch_matmul_test_tiling_float16_shard_with_bias_test9", "Ascend910_93", 20, 196608, 4096, {{16, 128, 64}, {4, 64, 128}, {4, 1, 128, 1}}, {ge::DT_FLOAT16, ge::DT_FLOAT16, ge::DT_FLOAT16}, {4, 512, 64}, ge::DT_FLOAT16, "ep_group", "tp_group", 4, 2, 1, 0, false, false, false, false, 0},
    {"all_to_all_all_gather_batch_matmul_test_tiling_float16_shard_with_bias_test10", "Ascend910_93", 20, 196608, 4096, {{16, 128, 64}, {4, 64, 128}, {5, 1, 128}}, {ge::DT_FLOAT16, ge::DT_FLOAT16, ge::DT_FLOAT16}, {4, 512, 64}, ge::DT_FLOAT16, "ep_group", "tp_group", 4, 2, 1, 0, false, false, false, false, 0},
    {"all_to_all_all_gather_batch_matmul_test_tiling_float16_shard_with_bias_test11", "Ascend910_93", 20, 196608, 4096, {{16, 128, 0}, {4, 64, 128}, {4, 1, 128}}, {ge::DT_FLOAT16, ge::DT_FLOAT16, ge::DT_FLOAT16}, {4, 512, 64}, ge::DT_FLOAT16, "ep_group", "tp_group", 4, 2, 1, 0, false, false, false, false, 0},
    {"all_to_all_all_gather_batch_matmul_test_tiling_float16_shard_with_bias_test12", "Ascend910_93", 20, 196608, 4096, {{128, 64}, {4, 64, 128}, {4, 1, 128}}, {ge::DT_FLOAT16, ge::DT_FLOAT16, ge::DT_FLOAT16}, {4, 512, 64}, ge::DT_FLOAT16, "ep_group", "tp_group", 4, 2, 1, 0, false, false, false, false, 0},
    {"all_to_all_all_gather_batch_matmul_test_tiling_float16_xshard_0_ep2", "Ascend910_93", 20, 196608, 4096, {{16, 256, 32}, {8, 128, 128}}, {ge::DT_FLOAT16, ge::DT_FLOAT16}, {8, 512, 128}, ge::DT_FLOAT16, "ep_group", "tp_group", 2, 4, 0, 0, false, false, false, true, 0xDE0B6B3A7640000},
    {"all_to_all_all_gather_batch_matmul_test_tiling_float16_xshard_0_cut_e", "Ascend910_93", 20, 196608, 4096, {{16, 2254, 2048}, {4, 4096, 1024}}, {ge::DT_FLOAT16, ge::DT_FLOAT16}, {4, 9016, 1024}, ge::DT_FLOAT16, "ep_group", "tp_group", 4, 2, 0, 0, false, false, false, true, 0xDE0B6B3A7640000},
    {"all_to_all_all_gather_batch_matmul_test_tiling_float16_xshard_0_cut_c", "Ascend910_93", 20, 196608, 4096, {{8, 2254, 2048}, {2, 4096, 1024}}, {ge::DT_FLOAT16, ge::DT_FLOAT16}, {2, 9016, 1024}, ge::DT_FLOAT16, "ep_group", "tp_group", 4, 2, 0, 0, false, false, false, true, 0xDE0B6B3A7640000},
    {"all_to_all_all_gather_batch_matmul_test_tiling_float16_xshard_0_tile_short", "Ascend910_93", 20, 196608, 4096, {{8, 2254, 6144}, {2, 12288, 6144}}, {ge::DT_FLOAT16, ge::DT_FLOAT16}, {2, 9016, 6144}, ge::DT_FLOAT16, "ep_group", "tp_group", 4, 2, 0, 0, false, false, false, true, 0xDE0B6B3A7640000},
    {"all_to_all_all_gather_batch_matmul_test_tiling_float16_xshard_0_multi_e", "Ascend910_93", 20, 196608, 4096, {{40, 2254, 6144}, {10, 12288, 1024}}, {ge::DT_FLOAT16, ge::DT_FLOAT16}, {10, 9016, 1024}, ge::DT_FLOAT16, "ep_group", "tp_group", 4, 2, 0, 0, false, false, false, true, 0xDE0B6B3A7640000},
    {"all_to_all_all_gather_batch_matmul_test_tiling_float16_xshard_0_local_tail_e", "Ascend910_93", 20, 196608, 4096, {{10, 2254, 1024}, {5, 8192, 8192}}, {ge::DT_FLOAT16, ge::DT_FLOAT16}, {5, 4508, 8192}, ge::DT_FLOAT16, "ep_group", "tp_group", 2, 8, 0, 0, false, false, false, true, 0xDE0B6B3A7640000},
};

TEST_P(AlltoAllAllGatherBmmTilingParam, general_case)
{
    if (!IsOpImplRegistryAvailable()) {
        GTEST_SKIP() << "Skip test: OpImplSpaceRegistryV2 is null on host.";
    }

    Mc2Hcom::MockValues hcomTopologyMockValues{{"rankNum", 8}};
    Mc2Hcom::MC2HcomTopologyMocker::GetInstance().SetValues(hcomTopologyMockValues);

    const auto &param = GetParam();
    TestOneParamCase(param);

    Mc2Hcom::MC2HcomTopologyMocker::GetInstance().Reset();
}

INSTANTIATE_TEST_SUITE_P(
    general_cases_params,
    AlltoAllAllGatherBmmTilingParam,
    ::testing::ValuesIn(cases_params),
    [](const ::testing::TestParamInfo<AlltoAllAllGatherBmmTilingTestParam> &info) {
        std::string name = info.param.case_name;
        for (char &c : name) {
            if (!std::isalnum(static_cast<unsigned char>(c)) && c != '_') {
                c = '_';
            }
        }
        return name;
    });

} // anonymous namespace

} // namespace AlltoAllAllGatherBatchMatMulUT
