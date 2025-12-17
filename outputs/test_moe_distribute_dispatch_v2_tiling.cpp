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
#include <thread>
#include <cctype>
#include <map>
#include <vector>
#include <string>

#include <gtest/gtest.h>

#include "exe_graph/runtime/storage_format.h"
#include "exe_graph/runtime/storage_shape.h"

#include "mc2_tiling_case_executor.h"

namespace MoeDistributeDispatchV2 {

namespace {

template <typename T>
auto build_from(const T &value)
{
    return Ops::Transformer::AnyValue::CreateFrom<T>(value);
}

struct MoeDistributeDispatchV2TilingTestParam {
    uint64_t inputTotalNum;
    uint64_t outputTotalNum;
    std::string case_name;
    std::string soc_version;
    uint64_t coreNum;
    uint64_t ubSize;

    std::initializer_list<int64_t> input0_shape;
    std::initializer_list<int64_t> input1_shape;
    std::initializer_list<int64_t> input2_shape;
    std::initializer_list<int64_t> input3_shape;
    std::initializer_list<int64_t> input4_shape;
    std::initializer_list<int64_t> input5_shape;

    std::initializer_list<int64_t> output0_shape;
    std::initializer_list<int64_t> output1_shape;
    std::initializer_list<int64_t> output2_shape;
    std::initializer_list<int64_t> output3_shape;
    std::initializer_list<int64_t> output4_shape;
    std::initializer_list<int64_t> output5_shape;
    std::initializer_list<int64_t> output6_shape;

    ge::DataType input0_dtype;
    ge::DataType input1_dtype;
    ge::DataType input2_dtype;
    ge::DataType input3_dtype;
    ge::DataType input4_dtype;
    ge::DataType input5_dtype;

    ge::DataType output0_dtype;
    ge::DataType output1_dtype;
    ge::DataType output2_dtype;
    ge::DataType output3_dtype;
    ge::DataType output4_dtype;
    ge::DataType output5_dtype;
    ge::DataType output6_dtype;

    std::string ep_group;
    std::string tp_group;
    int64_t ep_world_size;
    int64_t tp_world_size;
    int64_t ep_rank_id;
    int64_t tp_rank_id;
    int64_t expert_shard_type;
    int64_t shared_expert_num;
    int64_t shared_expert_rank_num;
    int64_t moe_expert_num;
    int64_t quant_mode;
    int64_t global_bs;
    int64_t expert_token_nums_type;
    std::string comm_alg;
    int64_t zero_expert_num;
    int64_t copy_expert_num;
    int64_t const_expert_num;

    bool has_expect_tiling_key;
    uint64_t expect_tiling_key;
};

gert::StorageShape make_shape(const std::initializer_list<int64_t> &shape)
{
    if (shape.size() == 0) {
        return gert::StorageShape{};
    }
    return gert::StorageShape{shape, shape};
}

class MoeDistributeDispatchV2TilingParam
    : public ::testing::TestWithParam<MoeDistributeDispatchV2TilingTestParam> {
protected:
    static void SetUpTestCase()
    {
        std::cout << "MoeDistributeDispatchV2Tiling SetUp" << std::endl;
    }

    static void TearDownTestCase()
    {
        std::cout << "MoeDistributeDispatchV2Tiling TearDown" << std::endl;
    }
};

void TestOneParamCase(const MoeDistributeDispatchV2TilingTestParam &param)
{
    struct MoeDistributeDispatchV2CompileInfo {};
    MoeDistributeDispatchV2CompileInfo compileInfo;

    std::vector<std::pair<std::initializer_list<int64_t>, ge::DataType>> inputShapeDtypeList = {
        {param.input0_shape, param.input0_dtype},
        {param.input1_shape, param.input1_dtype},
        {param.input2_shape, param.input2_dtype},
        {param.input3_shape, param.input3_dtype},
        {param.input4_shape, param.input4_dtype},
        {param.input5_shape, param.input5_dtype},
    };

    std::vector<gert::TilingContextPara::TensorDescription> inputList;
    for (uint64_t i = 0; i < param.inputTotalNum; ++i) {
        inputList.push_back({make_shape(inputShapeDtypeList[i].first),
                             inputShapeDtypeList[i].second,
                             ge::FORMAT_ND});
    }

    std::vector<std::pair<std::initializer_list<int64_t>, ge::DataType>> outputShapeDtypeList = {
        {param.output0_shape, param.output0_dtype},
        {param.output1_shape, param.output1_dtype},
        {param.output2_shape, param.output2_dtype},
        {param.output3_shape, param.output3_dtype},
        {param.output4_shape, param.output4_dtype},
        {param.output5_shape, param.output5_dtype},
        {param.output6_shape, param.output6_dtype},
    };

    std::vector<gert::TilingContextPara::TensorDescription> outputList;
    for (uint64_t i = 0; i < param.outputTotalNum; ++i) {
        outputList.push_back({make_shape(outputShapeDtypeList[i].first),
                              outputShapeDtypeList[i].second,
                              ge::FORMAT_ND});
    }

    gert::TilingContextPara tilingContextPara("MoeDistributeDispatchV2",
        inputList,
        outputList,
        {
            {"group_ep", build_from<std::string>(param.ep_group)},
            {"ep_world_size", build_from<int64_t>(param.ep_world_size)},
            {"ep_rank_id", build_from<int64_t>(param.ep_rank_id)},
            {"moe_expert_num", build_from<int64_t>(param.moe_expert_num)},
            {"group_tp", build_from<std::string>(param.tp_group)},
            {"tp_world_size", build_from<int64_t>(param.tp_world_size)},
            {"tp_rank_id", build_from<int64_t>(param.tp_rank_id)},
            {"expert_shard_type", build_from<int64_t>(param.expert_shard_type)},
            {"shared_expert_num", build_from<int64_t>(param.shared_expert_num)},
            {"shared_expert_rank_num", build_from<int64_t>(param.shared_expert_rank_num)},
            {"quant_mode", build_from<int64_t>(param.quant_mode)},
            {"global_bs", build_from<int64_t>(param.global_bs)},
            {"expert_token_nums_type", build_from<int64_t>(param.expert_token_nums_type)},
            {"comm_alg", build_from<std::string>(param.comm_alg)},
            {"zero_expert_num", build_from<int64_t>(param.zero_expert_num)},
            {"copy_expert_num", build_from<int64_t>(param.copy_expert_num)},
            {"const_expert_num", build_from<int64_t>(param.const_expert_num)},
        },
        &compileInfo, param.soc_version, param.coreNum, param.ubSize);

    Mc2Hcom::MockValues hcomTopologyMockValues{{"rankNum", 8}};
    if (param.has_expect_tiling_key) {
        Mc2ExecuteTestCase(tilingContextPara, hcomTopologyMockValues, ge::GRAPH_SUCCESS, param.expect_tiling_key);
    } else {
        Mc2ExecuteTestCase(tilingContextPara, hcomTopologyMockValues);
    }
}

MoeDistributeDispatchV2TilingTestParam cases_params[] = {
    {2, 6, "moe_distribute_dispatch_test_tiling_0", "Ascend910_93", 20, 196608, {32, 7168}, {32, 8}, {}, {}, {}, {}, {576, 7168}, {576}, {256}, {1}, {288}, {2}, {}, ge::DT_FLOAT16, ge::DT_INT32, ge::DT_FLOAT16, ge::DT_FLOAT16, ge::DT_FLOAT16, ge::DT_FLOAT16, ge::DT_FLOAT16, ge::DT_FLOAT, ge::DT_INT32, ge::DT_INT64, ge::DT_INT32, ge::DT_INT32, ge::DT_FLOAT, "ep_group", "tp_group", 288, 2, 0, 0, 0, 1, 32, 256, 0, 0, 1, "", 0, 0, 0, false, 0},
    {2, 6, "moe_distribute_dispatch_test_tiling_1", "Ascend910_93", 20, 196608, {16, 7160}, {16, 8}, {}, {}, {}, {}, {576, 7160}, {576}, {128}, {1}, {288}, {2}, {}, ge::DT_FLOAT16, ge::DT_INT32, ge::DT_FLOAT16, ge::DT_FLOAT16, ge::DT_FLOAT16, ge::DT_FLOAT16, ge::DT_FLOAT16, ge::DT_FLOAT, ge::DT_INT32, ge::DT_INT64, ge::DT_INT32, ge::DT_INT32, ge::DT_FLOAT, "ep_group", "tp_group", 288, 2, 0, 0, 0, 1, 32, 256, 0, 0, 1, "", 0, 0, 0, false, 0},
    {2, 6, "moe_distribute_dispatch_test_tiling_2", "Ascend910_93", 20, 196608, {16, 7160}, {16, 8}, {}, {}, {}, {}, {576, 7160}, {576}, {128}, {1}, {288}, {2}, {}, ge::DT_FLOAT16, ge::DT_INT32, ge::DT_FLOAT16, ge::DT_FLOAT16, ge::DT_FLOAT16, ge::DT_FLOAT16, ge::DT_FLOAT16, ge::DT_FLOAT, ge::DT_INT32, ge::DT_INT64, ge::DT_INT32, ge::DT_INT32, ge::DT_FLOAT, "ep_group", "tp_group", 288, 2, 0, 1024, 0, 1, 32, 256, 0, 0, 1, "", 0, 0, 0, false, 0},
    {2, 6, "moe_distribute_dispatch_test_tiling_3", "Ascend910_93", 20, 196608, {16, 7160}, {16, 8}, {}, {}, {}, {}, {576, 7160}, {576}, {128}, {1}, {288}, {2}, {}, ge::DT_FLOAT16, ge::DT_INT32, ge::DT_FLOAT16, ge::DT_FLOAT16, ge::DT_FLOAT16, ge::DT_FLOAT16, ge::DT_FLOAT16, ge::DT_FLOAT, ge::DT_INT32, ge::DT_INT64, ge::DT_INT32, ge::DT_INT32, ge::DT_FLOAT, "ep_group", "tp_group", 288, 2, 0, 0, 0, 1, 31, 256, 0, 0, 0, "", 0, 0, 0, false, 0},
    {2, 6, "moe_distribute_dispatch_test_tiling_4", "Ascend910_93", 20, 196608, {16, 7168}, {16, 8}, {}, {}, {}, {}, {576, 7168}, {576}, {128}, {1}, {288}, {2}, {}, ge::DT_FLOAT16, ge::DT_INT32, ge::DT_FLOAT16, ge::DT_FLOAT16, ge::DT_FLOAT16, ge::DT_FLOAT16, ge::DT_FLOAT16, ge::DT_FLOAT, ge::DT_INT32, ge::DT_INT64, ge::DT_INT32, ge::DT_INT32, ge::DT_FLOAT, "ep_group", "tp_group", 288, 2, 0, 0, 0, 1, 31, 257, 1, 0, 0, "", 0, 0, 0, false, 0},
    {2, 6, "moe_distribute_dispatch_test_tiling_5", "Ascend910_93", 20, 196608, {16, 7168}, {16, 8}, {}, {}, {}, {}, {576, 7168}, {576}, {128}, {1}, {288}, {2}, {}, ge::DT_FLOAT16, ge::DT_INT32, ge::DT_FLOAT16, ge::DT_FLOAT16, ge::DT_FLOAT16, ge::DT_FLOAT16, ge::DT_FLOAT16, ge::DT_FLOAT, ge::DT_INT32, ge::DT_INT64, ge::DT_INT32, ge::DT_INT32, ge::DT_FLOAT, "ep_group", "tp_group", 288, 2, 0, 0, 0, 1, 32, 256, 10, 0, 0, "", 0, 0, 0, false, 0},
    {2, 6, "moe_distribute_dispatch_test_tiling_6", "Ascend910_93", 20, 196608, {8, 7168}, {8, 7}, {}, {}, {}, {}, {64, 7168}, {64}, {512}, {1}, {8}, {1}, {}, ge::DT_FLOAT16, ge::DT_INT32, ge::DT_FLOAT16, ge::DT_FLOAT16, ge::DT_FLOAT16, ge::DT_FLOAT16, ge::DT_FLOAT16, ge::DT_FLOAT, ge::DT_INT32, ge::DT_INT64, ge::DT_INT32, ge::DT_INT32, ge::DT_FLOAT, "ep_group", "tp_group", 8, 1, 0, 0, 0, 1, 1, 7, 0, 0, 1, "", 0, 0, 0, true, 10000},
    {2, 6, "moe_distribute_dispatch_test_tiling_7", "Ascend910_93", 20, 196608, {32, 7168}, {32, 8}, {}, {}, {}, {}, {576, 7168}, {576}, {256}, {1}, {288}, {2}, {}, ge::DT_FLOAT16, ge::DT_INT32, ge::DT_FLOAT16, ge::DT_FLOAT16, ge::DT_FLOAT16, ge::DT_FLOAT16, ge::DT_FLOAT16, ge::DT_FLOAT, ge::DT_INT32, ge::DT_INT64, ge::DT_INT32, ge::DT_INT32, ge::DT_FLOAT, "ep_group", "tp_group", 288, 2, 0, -1, 0, 1, 32, 256, 2, 0, 0, "", 0, 0, 0, false, 0},
    {2, 6, "moe_distribute_dispatch_test_tiling_8", "Ascend910_93", 20, 196608, {32, 7168}, {32, 8}, {}, {}, {}, {}, {576, 7168}, {576}, {256}, {1}, {288}, {2}, {}, ge::DT_FLOAT16, ge::DT_INT32, ge::DT_FLOAT16, ge::DT_FLOAT16, ge::DT_FLOAT16, ge::DT_FLOAT16, ge::DT_FLOAT16, ge::DT_FLOAT, ge::DT_INT32, ge::DT_INT64, ge::DT_INT32, ge::DT_INT32, ge::DT_FLOAT, "ep_group", "", 288, 2, 1, 1024, 1, 1, 32, 256, 2, 1, 1, "", 0, 0, 0, false, 0},
    {2, 6, "moe_distribute_dispatch_test_tiling_9", "Ascend910_93", 20, 196608, {16, 7160}, {16, 8}, {}, {}, {}, {}, {576, 7160}, {576}, {128}, {1}, {288}, {2}, {}, ge::DT_FLOAT16, ge::DT_INT32, ge::DT_FLOAT16, ge::DT_FLOAT16, ge::DT_FLOAT16, ge::DT_FLOAT16, ge::DT_FLOAT16, ge::DT_FLOAT, ge::DT_INT32, ge::DT_INT64, ge::DT_INT32, ge::DT_INT32, ge::DT_FLOAT, "ep_group", "", 288, 2, 0, -1, 0, 1, 32, 256, 2, 0, 0, "", 0, 0, 0, false, 0},
    {2, 6, "moe_distribute_dispatch_test_tiling_10", "Ascend910_93", 20, 196608, {32, 7168}, {32, 8}, {}, {}, {}, {}, {576, 7168}, {576}, {256}, {1}, {288}, {2}, {}, ge::DT_FLOAT16, ge::DT_INT32, ge::DT_FLOAT16, ge::DT_FLOAT16, ge::DT_FLOAT16, ge::DT_FLOAT16, ge::DT_FLOAT16, ge::DT_FLOAT, ge::DT_INT32, ge::DT_INT64, ge::DT_INT32, ge::DT_INT32, ge::DT_FLOAT, "ep_group", "tp_group", 288, 2, 256, 0, 0, 1, 32, 256, 0, 0, 1, "", 0, 0, 0, false, 0},
    {2, 6, "moe_distribute_dispatch_test_tiling_ep_world_size_384", "Ascend910_93", 20, 196608, {32, 7168}, {32, 8}, {}, {}, {}, {}, {576, 7168}, {576}, {256}, {1}, {288}, {2}, {}, ge::DT_FLOAT16, ge::DT_INT32, ge::DT_FLOAT16, ge::DT_FLOAT16, ge::DT_FLOAT16, ge::DT_FLOAT16, ge::DT_FLOAT16, ge::DT_FLOAT, ge::DT_INT32, ge::DT_INT64, ge::DT_INT32, ge::DT_INT32, ge::DT_FLOAT, "ep_group", "tp_group", 384, 2, 0, 0, 0, 1, 32, 256, 0, 0, 1, "", 0, 0, 0, false, 0},
    {2, 6, "moe_distribute_dispatch_test_tiling_ep_world_size_72", "Ascend910_93", 20, 196608, {32, 7168}, {32, 8}, {}, {}, {}, {}, {576, 7168}, {576}, {256}, {1}, {288}, {2}, {}, ge::DT_FLOAT16, ge::DT_INT32, ge::DT_FLOAT16, ge::DT_FLOAT16, ge::DT_FLOAT16, ge::DT_FLOAT16, ge::DT_FLOAT16, ge::DT_FLOAT, ge::DT_INT32, ge::DT_INT64, ge::DT_INT32, ge::DT_INT32, ge::DT_FLOAT, "ep_group", "tp_group", 72, 2, 0, 0, 0, 1, 18, 216, 0, 0, 1, "", 0, 0, 0, false, 0},
    {5, 6, "moe_distribute_dispatch_test_tiling_x_active_mask_2dims", "Ascend910_93", 20, 196608, {32, 7168}, {32, 8}, {}, {32, 8}, {}, {}, {576, 7168}, {576}, {256}, {1}, {288}, {2}, {}, ge::DT_FLOAT16, ge::DT_INT32, ge::DT_FLOAT, ge::DT_BOOL, ge::DT_FLOAT, ge::DT_FLOAT16, ge::DT_FLOAT16, ge::DT_FLOAT, ge::DT_INT32, ge::DT_INT64, ge::DT_INT32, ge::DT_INT32, ge::DT_FLOAT, "ep_group", "tp_group", 72, 2, 0, 0, 0, 1, 18, 216, 0, 0, 1, "", 0, 0, 0, false, 0},
    {6, 6, "moe_distribute_dispatch_test_tiling_elastic_info", "Ascend910_93", 20, 196608, {32, 7168}, {32, 8}, {}, {32, 8}, {}, {148}, {576, 7168}, {576}, {256}, {1}, {288}, {2}, {}, ge::DT_FLOAT16, ge::DT_INT32, ge::DT_FLOAT, ge::DT_BOOL, ge::DT_FLOAT, ge::DT_INT32, ge::DT_FLOAT16, ge::DT_FLOAT, ge::DT_INT32, ge::DT_INT64, ge::DT_INT32, ge::DT_INT32, ge::DT_FLOAT, "ep_group", "tp_group", 72, 1, 0, 0, 0, 1, 18, 216, 0, 0, 1, "", 0, 0, 0, false, 0},
    {2, 6, "moe_distribute_dispatch_test_zeroComputeExpertNum", "Ascend910_93", 20, 196608, {8, 7168}, {8, 7}, {}, {}, {}, {}, {64, 7168}, {64}, {512}, {1}, {8}, {1}, {}, ge::DT_FLOAT16, ge::DT_INT32, ge::DT_FLOAT16, ge::DT_FLOAT16, ge::DT_FLOAT16, ge::DT_FLOAT16, ge::DT_FLOAT16, ge::DT_FLOAT, ge::DT_INT32, ge::DT_INT64, ge::DT_INT32, ge::DT_INT32, ge::DT_FLOAT, "ep_group", "tp_group", 8, 1, 0, 0, 0, 1, 1, 7, 0, 0, 1, "", 1, 2, 3, true, 10000},
    {2, 6, "moe_distribute_dispatch_test_zeroComputeExpertNum_invalid", "Ascend910_93", 20, 196608, {8, 7168}, {8, 7}, {}, {}, {}, {}, {64, 7168}, {64}, {512}, {1}, {8}, {1}, {}, ge::DT_FLOAT16, ge::DT_INT32, ge::DT_FLOAT16, ge::DT_FLOAT16, ge::DT_FLOAT16, ge::DT_FLOAT16, ge::DT_FLOAT16, ge::DT_FLOAT, ge::DT_INT32, ge::DT_INT64, ge::DT_INT32, ge::DT_INT32, ge::DT_FLOAT, "ep_group", "tp_group", 8, 1, 0, 0, 0, 1, 1, 7, 0, 0, 1, "", 0xFFFFFFFF, 2, 3, false, 0},
    {2, 6, "moe_distribute_dispatch_test_tiling_a2_commalg_empty", "Ascend910B", 48, 196608, {8, 7168}, {8, 8}, {}, {}, {}, {}, {2048, 7168}, {2048}, {64}, {8}, {256}, {1}, {}, ge::DT_FLOAT16, ge::DT_INT32, ge::DT_FLOAT16, ge::DT_FLOAT16, ge::DT_FLOAT16, ge::DT_FLOAT16, ge::DT_FLOAT16, ge::DT_FLOAT, ge::DT_INT32, ge::DT_INT64, ge::DT_INT32, ge::DT_INT32, ge::DT_FLOAT, "ep_group", "", 32, 1, 0, 0, 0, 1, 0, 256, 0, 0, 0, "", 0, 0, 0, true, 0x773597E8},
    {2, 6, "moe_distribute_dispatch_test_tiling_a2_commalg_fullmesh", "Ascend910B", 48, 196608, {8, 7168}, {8, 8}, {}, {}, {}, {}, {2048, 7168}, {2048}, {64}, {8}, {256}, {1}, {}, ge::DT_FLOAT16, ge::DT_INT32, ge::DT_FLOAT16, ge::DT_FLOAT16, ge::DT_FLOAT16, ge::DT_FLOAT16, ge::DT_FLOAT16, ge::DT_FLOAT, ge::DT_INT32, ge::DT_INT64, ge::DT_INT32, ge::DT_INT32, ge::DT_FLOAT, "ep_group", "", 32, 1, 0, 0, 0, 1, 0, 256, 0, 0, 0, "fullmesh", 0, 0, 0, true, 0x773597E8},
    {5, 7, "moe_distribute_dispatch_test_tiling_a2_commalg_hierarchy", "Ascend910B", 48, 196608, {8, 7168}, {8, 8}, {}, {}, {8, 8}, {}, {2048, 7168}, {2048}, {64}, {8}, {256}, {1}, {2048}, ge::DT_FLOAT16, ge::DT_INT32, ge::DT_FLOAT, ge::DT_BOOL, ge::DT_FLOAT, ge::DT_FLOAT16, ge::DT_FLOAT16, ge::DT_FLOAT, ge::DT_INT32, ge::DT_INT64, ge::DT_INT32, ge::DT_INT32, ge::DT_FLOAT, "ep_group", "", 32, 0, 0, 0, 0, 1, 0, 256, 0, 0, 0, "hierarchy", 0, 0, 0, true, 0x7D2B78E8},
    {2, 6, "moe_distribute_dispatch_test_tiling_a2_commalg_error", "Ascend910B", 48, 196608, {8, 7168}, {8, 8}, {}, {}, {}, {}, {2048, 7168}, {2048}, {64}, {8}, {256}, {1}, {}, ge::DT_FLOAT16, ge::DT_INT32, ge::DT_FLOAT16, ge::DT_FLOAT16, ge::DT_FLOAT16, ge::DT_FLOAT16, ge::DT_FLOAT16, ge::DT_FLOAT, ge::DT_INT32, ge::DT_INT64, ge::DT_INT32, ge::DT_INT32, ge::DT_FLOAT, "ep_group", "", 32, 0, 0, 0, 0, 1, 0, 256, 0, 0, 0, "error", 0, 0, 0, false, 0},
    {5, 7, "moe_distribute_dispatch_test_tiling_a2_commalg_empty_with_env", "Ascend910B", 48, 196608, {8, 7168}, {8, 8}, {}, {}, {8, 8}, {}, {2048, 7168}, {2048}, {64}, {8}, {256}, {1}, {2048}, ge::DT_FLOAT16, ge::DT_INT32, ge::DT_INT32, ge::DT_INT32, ge::DT_FLOAT, ge::DT_FLOAT16, ge::DT_FLOAT16, ge::DT_FLOAT, ge::DT_INT32, ge::DT_INT64, ge::DT_INT32, ge::DT_INT32, ge::DT_FLOAT, "ep_group", "", 32, 0, 0, 0, 0, 1, 0, 256, 0, 0, 0, "", 0, 0, 0, true, 0x773597E8},
    {2, 6, "moe_distribute_dispatch_test_tiling_a2_commalg_fullmesh_with_env", "Ascend910B", 48, 196608, {8, 7168}, {8, 8}, {}, {}, {}, {}, {2048, 7168}, {2048}, {64}, {8}, {256}, {1}, {}, ge::DT_FLOAT16, ge::DT_INT32, ge::DT_FLOAT16, ge::DT_FLOAT16, ge::DT_FLOAT16, ge::DT_FLOAT16, ge::DT_FLOAT16, ge::DT_FLOAT, ge::DT_INT32, ge::DT_INT64, ge::DT_INT32, ge::DT_INT32, ge::DT_FLOAT, "ep_group", "", 32, 1, 0, 0, 0, 1, 0, 256, 0, 0, 0, "fullmesh", 0, 0, 0, true, 0x773597E8},
    {2, 6, "moe_distribute_dispatch_test_tiling_a2_commalg_fullmesh_zeroComputeExpert_not_zero", "Ascend910B", 48, 196608, {8, 7168}, {8, 8}, {}, {}, {}, {}, {2048, 7168}, {2048}, {64}, {8}, {256}, {1}, {}, ge::DT_FLOAT16, ge::DT_INT32, ge::DT_FLOAT16, ge::DT_FLOAT16, ge::DT_FLOAT16, ge::DT_FLOAT16, ge::DT_FLOAT16, ge::DT_FLOAT, ge::DT_INT32, ge::DT_INT64, ge::DT_INT32, ge::DT_INT32, ge::DT_FLOAT, "ep_group", "", 32, 0, 0, 0, 0, 1, 0, 256, 0, 0, 0, "fullmesh", 1, 0, 0, true, 0x773597E8},
};

TEST_P(MoeDistributeDispatchV2TilingParam, general_case)
{
    const auto &param = GetParam();
    TestOneParamCase(param);
}

INSTANTIATE_TEST_SUITE_P(
    general_cases_params_v2,
    MoeDistributeDispatchV2TilingParam,
    ::testing::ValuesIn(cases_params),
    [](const ::testing::TestParamInfo<MoeDistributeDispatchV2TilingTestParam> &info) {
        std::string name = info.param.case_name;
        for (char &c : name) {
            if (!std::isalnum(static_cast<unsigned char>(c)) && c != '_') {
                c = '_';
            }
        }
        return name;
    });

} // anonymous namespace

} // namespace MoeDistributeDispatchV2

