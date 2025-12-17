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
#include <gtest/gtest.h>
#include "mc2_tiling_case_executor.h"

namespace MoeDistributeDispatchUT {

namespace {

template <typename T>
auto build_from(const T &value)
{
    return Ops::Transformer::AnyValue::CreateFrom<T>(value);
}

struct MoeDistributeDispatchTilingTestParam {
    uint64_t inputTotalNum;
    std::string case_name;
    std::string soc_version;
    uint64_t coreNum;
    uint64_t ubSize;

    std::initializer_list<int64_t> input0_shape;
    std::initializer_list<int64_t> input1_shape;
    std::initializer_list<int64_t> input2_shape;

    std::initializer_list<int64_t> output0_shape;
    std::initializer_list<int64_t> output1_shape;
    std::initializer_list<int64_t> output2_shape;
    std::initializer_list<int64_t> output3_shape;
    std::initializer_list<int64_t> output4_shape;
    std::initializer_list<int64_t> output5_shape;

    ge::DataType input0_dtype;
    ge::DataType input1_dtype;
    ge::DataType input2_dtype;

    ge::DataType output0_dtype;
    ge::DataType output1_dtype;
    ge::DataType output2_dtype;
    ge::DataType output3_dtype;
    ge::DataType output4_dtype;
    ge::DataType output5_dtype;

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

class MoeDistributeDispatchTilingParam : public ::testing::TestWithParam<MoeDistributeDispatchTilingTestParam> {
protected:
    static void SetUpTestCase()
    {
        std::cout << "MoeDistributeDispatchTiling SetUp" << std::endl;
    }

    static void TearDownTestCase()
    {
        std::cout << "MoeDistributeDispatchTiling TearDown" << std::endl;
    }
};

void TestOneParamCase(const MoeDistributeDispatchTilingTestParam &param)
{
    struct MoeDistributeDispatchCompileInfo {};
    MoeDistributeDispatchCompileInfo compileInfo;

    std::vector<std::pair<std::initializer_list<int64_t>, ge::DataType>> shapeDtypeList = {
        {param.input0_shape, param.input0_dtype},
        {param.input1_shape, param.input1_dtype},
        {param.input2_shape, param.input2_dtype},
    };

    std::vector<gert::TilingContextPara::TensorDescription> inputList;
    for (uint64_t i = 0; i < param.inputTotalNum; ++i) {
        inputList.push_back({make_shape(shapeDtypeList[i].first), shapeDtypeList[i].second, ge::FORMAT_ND});
    }

    // 输出列表
    std::vector<gert::TilingContextPara::TensorDescription> outputList = {
        {{param.output0_shape, param.output0_shape}, param.output0_dtype, ge::FORMAT_ND},
        {{param.output1_shape, param.output1_shape}, param.output1_dtype, ge::FORMAT_ND},
        {{param.output2_shape, param.output2_shape}, param.output2_dtype, ge::FORMAT_ND},
        {{param.output3_shape, param.output3_shape}, param.output3_dtype, ge::FORMAT_ND},
        {{param.output4_shape, param.output4_shape}, param.output4_dtype, ge::FORMAT_ND},
        {{param.output5_shape, param.output5_shape}, param.output5_dtype, ge::FORMAT_ND},
    };

    gert::TilingContextPara tilingContextPara("MoeDistributeDispatch", inputList, outputList,
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
        },
        &compileInfo, param.soc_version, param.coreNum, param.ubSize);

    Mc2Hcom::MockValues hcomTopologyMockValues{{"rankNum", 8}};
    if (param.has_expect_tiling_key) {
        Mc2ExecuteTestCase(tilingContextPara, hcomTopologyMockValues, ge::GRAPH_SUCCESS, param.expect_tiling_key);
    } else {
        Mc2ExecuteTestCase(tilingContextPara, hcomTopologyMockValues);
    }
}

MoeDistributeDispatchTilingTestParam cases_params[] = {
    {2, "moe_distribute_dispatch_test_tiling_0", "Ascend910_93", 20, 196608, {32, 7168}, {32, 8}, {}, {576, 7168}, {576}, {256}, {1}, {288}, {2}, ge::DT_FLOAT16, ge::DT_INT32, ge::DT_FLOAT16, ge::DT_FLOAT16, ge::DT_FLOAT16, ge::DT_INT32, ge::DT_INT64, ge::DT_INT32, ge::DT_INT32, "ep_group", "tp_group", 288, 2, 8, 0, 0, 1, 32, 256, 0, 0, 1, false, 0},
    {2, "moe_distribute_dispatch_test_tiling_1", "Ascend910_93", 20, 196608, {16, 7160}, {16, 8}, {}, {576, 7160}, {576}, {128}, {1}, {288}, {2}, ge::DT_FLOAT16, ge::DT_INT32, ge::DT_FLOAT16, ge::DT_FLOAT16, ge::DT_FLOAT, ge::DT_INT32, ge::DT_INT64, ge::DT_INT32, ge::DT_INT32, "ep_group", "tp_group", 288, 2, 0, 0, 0, 1, 32, 256, 0, 0, 1, false, 0},
    {2, "moe_distribute_dispatch_test_tiling_2", "Ascend910_93", 20, 196608, {16, 7160}, {16, 8}, {}, {576, 7160}, {576}, {128}, {1}, {288}, {2}, ge::DT_FLOAT16, ge::DT_INT32, ge::DT_FLOAT16, ge::DT_FLOAT16, ge::DT_FLOAT, ge::DT_INT32, ge::DT_INT64, ge::DT_INT32, ge::DT_INT32, "ep_group", "tp_group", 288, 2, 0, 1024, 0, 1, 32, 256, 0, 0, 1, false, 0},
    {2, "moe_distribute_dispatch_test_tiling_3", "Ascend910_93", 20, 196608, {16, 7160}, {16, 8}, {}, {576, 7160}, {576}, {128}, {1}, {288}, {2}, ge::DT_FLOAT16, ge::DT_INT32, ge::DT_FLOAT16, ge::DT_FLOAT16, ge::DT_FLOAT, ge::DT_INT32, ge::DT_INT64, ge::DT_INT32, ge::DT_INT32, "ep_group", "tp_group", 288, 2, 0, 0, 0, 1, 31, 256, 0, 0, 0, false, 0},
    {3, "moe_distribute_dispatch_test_tiling_4", "Ascend910_93", 20, 196608, {16, 7168}, {16, 8}, {33, 7168}, {576, 7168}, {576}, {128}, {1}, {288}, {2}, ge::DT_FLOAT16, ge::DT_INT32, ge::DT_FLOAT, ge::DT_INT8, ge::DT_FLOAT, ge::DT_INT32, ge::DT_INT64, ge::DT_INT32, ge::DT_INT32, "ep_group", "tp_group", 288, 2, 0, 0, 0, 1, 31, 257, 1, 0, 0, false, 0},
    {2, "moe_distribute_dispatch_test_tiling_5", "Ascend910_93", 20, 196608, {16, 7168}, {16, 8}, {}, {576, 7168}, {576}, {128}, {1}, {288}, {2}, ge::DT_FLOAT16, ge::DT_INT32, ge::DT_FLOAT16, ge::DT_FLOAT16, ge::DT_FLOAT, ge::DT_INT32, ge::DT_INT64, ge::DT_INT32, ge::DT_INT32, "ep_group", "tp_group", 288, 2, 0, 0, 0, 1, 32, 256, 10, 0, 0, false, 0},
    {2, "moe_distribute_dispatch_test_tiling_6", "Ascend910_93", 20, 196608, {8, 7168}, {8, 7}, {}, {64, 7168}, {64}, {56}, {1}, {8}, {1}, ge::DT_FLOAT16, ge::DT_INT32, ge::DT_FLOAT16, ge::DT_FLOAT16, ge::DT_FLOAT, ge::DT_INT32, ge::DT_INT64, ge::DT_INT32, ge::DT_INT32, "ep_group", "tp_group", 8, 1, 0, 0, 0, 1, 1, 7, 0, 0, 1, true, 1000},
    {2, "moe_distribute_dispatch_test_tiling_7", "Ascend910_93", 20, 196608, {32, 7168}, {32, 8}, {}, {576, 7168}, {576}, {256}, {1}, {288}, {2}, ge::DT_FLOAT16, ge::DT_INT32, ge::DT_FLOAT16, ge::DT_FLOAT16, ge::DT_FLOAT, ge::DT_INT32, ge::DT_INT64, ge::DT_INT32, ge::DT_INT32, "ep_group", "tp_group", 288, 2, 0, -1, 0, 1, 32, 256, 2, 0, 0, false, 0},
    {2, "moe_distribute_dispatch_test_tiling_8", "Ascend910_93", 20, 196608, {32, 7168}, {32, 8}, {}, {576, 7168}, {576}, {256}, {1}, {288}, {2}, ge::DT_FLOAT16, ge::DT_INT32, ge::DT_FLOAT16, ge::DT_FLOAT16, ge::DT_FLOAT, ge::DT_INT32, ge::DT_INT64, ge::DT_INT32, ge::DT_INT32, "ep_group", "", 288, 2, 1, 1024, 1, 1, 32, 256, 2, 1, 1, false, 0},
    {2, "moe_distribute_dispatch_test_tiling_9", "Ascend910_93", 20, 196608, {16, 7168}, {16, 8}, {}, {576, 7168}, {576}, {128}, {1}, {288}, {1}, ge::DT_FLOAT16, ge::DT_INT32, ge::DT_FLOAT16, ge::DT_INT8, ge::DT_FLOAT, ge::DT_INT32, ge::DT_INT64, ge::DT_INT32, ge::DT_INT32, "ep_group", "", 288, 2, 0, -1, 0, 1, 32, 256, 2, 0, 0, false, 0},
    {2, "moe_distribute_dispatch_test_tiling_10", "Ascend910_93", 20, 196608, {32, 7168}, {32, 8}, {}, {576, 7168}, {576}, {256}, {1}, {288}, {2}, ge::DT_FLOAT16, ge::DT_INT32, ge::DT_FLOAT16, ge::DT_FLOAT16, ge::DT_FLOAT, ge::DT_INT32, ge::DT_INT64, ge::DT_INT32, ge::DT_INT32, "ep_group", "tp_group", 288, 2, 256, 0, 0, 1, 32, 256, 0, 0, 1, false, 0},
    {2, "moe_distribute_dispatch_test_tiling_A2_quant0_layered", "Ascend910B", 48, 196608, {8, 7168}, {8, 8}, {}, {2048, 7168}, {2048}, {64}, {8}, {256}, {1}, ge::DT_FLOAT16, ge::DT_INT32, ge::DT_FLOAT16, ge::DT_FLOAT16, ge::DT_FLOAT, ge::DT_INT32, ge::DT_INT64, ge::DT_INT32, ge::DT_INT32, "ep_group", "", 32, 0, 0, 0, 0, 1, 0, 256, 0, 0, 0, true, 0x773597E8},
    {2, "moe_distribute_dispatch_test_tiling_A2_quant0", "Ascend910B", 48, 196608, {8, 7168}, {8, 8}, {}, {2048, 7168}, {2048}, {64}, {8}, {256}, {1}, ge::DT_FLOAT16, ge::DT_INT32, ge::DT_FLOAT16, ge::DT_FLOAT16, ge::DT_FLOAT, ge::DT_INT32, ge::DT_INT64, ge::DT_INT32, ge::DT_INT32, "ep_group", "", 32, 0, 0, 0, 0, 1, 0, 256, 0, 0, 0, true, 0x773597E8},
    {2, "moe_distribute_dispatch_test_tiling_A2_global_bs", "Ascend910B", 48, 196608, {8, 7168}, {8, 8}, {}, {2048, 7168}, {2048}, {64}, {8}, {256}, {1}, ge::DT_FLOAT16, ge::DT_INT32, ge::DT_FLOAT16, ge::DT_FLOAT16, ge::DT_FLOAT, ge::DT_INT32, ge::DT_INT64, ge::DT_INT32, ge::DT_INT32, "ep_group", "", 32, 0, 0, 0, 0, 1, 0, 256, 2, 512, 0, true, 0x773597EA},
    {2, "moe_distribute_dispatch_test_tiling_A2_ShapeAndEp_rank_id", "Ascend910B", 48, 196608, {8, 7160}, {8, 8}, {}, {2048, 7168}, {2048}, {64}, {8}, {256}, {1}, ge::DT_FLOAT16, ge::DT_INT32, ge::DT_FLOAT16, ge::DT_INT8, ge::DT_FLOAT, ge::DT_INT32, ge::DT_INT64, ge::DT_INT32, ge::DT_INT32, "ep_group", "", 32, 0, 33, 0, 0, 1, 0, 256, 2, 0, 0, false, 0},
    {2, "moe_distribute_dispatch_test_tiling_A2_moe_expert_num", "Ascend910B", 48, 196608, {8, 7168}, {8, 8}, {}, {2048, 7168}, {2048}, {64}, {8}, {256}, {1}, ge::DT_FLOAT16, ge::DT_INT32, ge::DT_FLOAT16, ge::DT_INT8, ge::DT_FLOAT, ge::DT_INT32, ge::DT_INT64, ge::DT_INT32, ge::DT_INT32, "ep_group", "", 32, 0, 0, 0, 0, 1, 0, 257, 2, 0, 0, false, 0},
    {2, "moe_distribute_dispatch_test_tiling_ep_world_size_384", "Ascend910_93", 20, 196608, {32, 7168}, {32, 8}, {}, {576, 7168}, {576}, {256}, {1}, {288}, {2}, ge::DT_FLOAT16, ge::DT_INT32, ge::DT_FLOAT16, ge::DT_FLOAT16, ge::DT_FLOAT, ge::DT_INT32, ge::DT_INT64, ge::DT_INT32, ge::DT_INT32, "ep_group", "tp_group", 384, 2, 0, 0, 0, 1, 32, 256, 0, 0, 1, false, 0},
    {2, "moe_distribute_dispatch_test_tiling_ep_world_size_72", "Ascend910_93", 20, 196608, {32, 7168}, {32, 8}, {}, {576, 7168}, {576}, {256}, {1}, {288}, {2}, ge::DT_FLOAT16, ge::DT_INT32, ge::DT_FLOAT16, ge::DT_FLOAT16, ge::DT_FLOAT, ge::DT_INT32, ge::DT_INT64, ge::DT_INT32, ge::DT_INT32, "ep_group", "tp_group", 72, 2, 0, 0, 0, 1, 18, 216, 0, 0, 1, false, 0},
};

TEST_P(MoeDistributeDispatchTilingParam, general_case)
{
    const auto &param = GetParam();
    TestOneParamCase(param);
}

INSTANTIATE_TEST_SUITE_P(
    general_cases_params,
    MoeDistributeDispatchTilingParam,
    ::testing::ValuesIn(cases_params),
    [](const ::testing::TestParamInfo<MoeDistributeDispatchTilingTestParam> &info) {
        std::string name = info.param.case_name;
        for (char &c : name) {
            if (!std::isalnum(static_cast<unsigned char>(c)) && c != '_') {
                c = '_';
            }
        }
        return name;
    });

} // anonymous namespace

} // namespace MoeDistributeDispatchUT

