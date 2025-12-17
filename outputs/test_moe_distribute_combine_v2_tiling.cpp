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

namespace MoeDistributeCombineV2UT {

namespace {

template <typename T>
auto build_from(const T &value)
{
    return Ops::Transformer::AnyValue::CreateFrom<T>(value);
}

struct MoeDistributeCombineV2TilingTestParam {
    std::string case_name;
    std::string soc_version;
    uint64_t coreNum;
    uint64_t ubSize;

    std::vector<gert::TilingContextPara::TensorDescription> inputs;
    std::vector<gert::TilingContextPara::TensorDescription> outputs;

    std::vector<gert::TilingContextPara::OpAttr> attrs;

    bool hasExpectTilingKey;
    uint64_t expectTilingKey;
};

class MoeDistributeCombineV2TilingParam
    : public ::testing::TestWithParam<MoeDistributeCombineV2TilingTestParam> {
protected:
    static void SetUpTestCase()
    {
        std::cout << "MoeDistributeCombineV2Tiling SetUp" << std::endl;
    }

    static void TearDownTestCase()
    {
        std::cout << "MoeDistributeCombineV2Tiling TearDown" << std::endl;
    }
};

void TestOneParamCase(const MoeDistributeCombineV2TilingTestParam &param)
{
    struct MoeDistributeCombineV2TilingCompileInfo {};
    MoeDistributeCombineV2TilingCompileInfo compileInfo;

    gert::TilingContextPara tilingContextPara("MoeDistributeCombineV2",
        param.inputs,
        param.outputs,
        param.attrs,
        &compileInfo,
        param.soc_version,
        param.coreNum,
        param.ubSize);

    Mc2Hcom::MockValues hcomTopologyMockValues{{"rankNum", 8}};

    if (param.hasExpectTilingKey) {
        Mc2ExecuteTestCase(tilingContextPara, hcomTopologyMockValues,
            ge::GRAPH_SUCCESS, param.expectTilingKey);
    } else {
        Mc2ExecuteTestCase(tilingContextPara, hcomTopologyMockValues);
    }
}

MoeDistributeCombineV2TilingTestParam cases_params[] = {
    {
        "moe_distribute_combine_test_tiling_shared_expert_x_0",
        "Ascend910_93",
        20,
        196608,
        {
            {{{64, 7168}, {64, 7168}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{8, 7}, {8, 7}}, ge::DT_INT32, ge::FORMAT_ND},
            {{{192}, {192}}, ge::DT_INT32, ge::FORMAT_ND},
            {{{8}, {8}}, ge::DT_INT32, ge::FORMAT_ND},
            {{{8, 7}, {8, 7}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{1}, {1}}, ge::DT_INT32, ge::FORMAT_ND},
            {{}, ge::DT_INT32, ge::FORMAT_ND},
            {{}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{}, ge::DT_INT32, ge::FORMAT_ND},
            {{}, ge::DT_INT32, ge::FORMAT_ND},
            {{}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{8, 7168}, {8, 7168}}, ge::DT_FLOAT16, ge::FORMAT_ND},
        },
        {
            {{{8, 7168}, {8, 7168}}, ge::DT_FLOAT16, ge::FORMAT_ND},
        },
        {
            {"group_ep", build_from<std::string>("ep_group")},
            {"ep_world_size", build_from<int64_t>(8)},
            {"ep_rank_id", build_from<int64_t>(0)},
            {"moe_expert_num", build_from<int64_t>(7)},
            {"group_tp", build_from<std::string>("tp_group")},
            {"tp_world_size", build_from<int64_t>(1)},
            {"tp_rank_id", build_from<int64_t>(0)},
            {"expert_shard_type", build_from<int64_t>(0)},
            {"shared_expert_num", build_from<int64_t>(1)},
            {"shared_expert_rank_num", build_from<int64_t>(1)},
            {"global_bs", build_from<int64_t>(0)},
            {"out_dtype", build_from<int64_t>(0)},
            {"comm_quant_mode", build_from<int64_t>(0)},
            {"group_list_type", build_from<int64_t>(0)},
            {"comm_alg", build_from<std::string>("")},
            {"zero_expert_num", build_from<int64_t>(0)},
            {"copy_expert_num", build_from<int64_t>(0)},
            {"const_expert_num", build_from<int64_t>(0)}
        },
        false,
        0UL
    },
    {
        "moe_distribute_combine_test_tiling_shared_expert_x_1",
        "Ascend910_93",
        20,
        196608,
        {
            {{{64, 7168}, {64, 7168}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{8, 7}, {8, 7}}, ge::DT_INT32, ge::FORMAT_ND},
            {{{16384}, {16384}}, ge::DT_INT32, ge::FORMAT_ND},
            {{{8}, {8}}, ge::DT_INT32, ge::FORMAT_ND},
            {{{8, 7}, {8, 7}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{1}, {1}}, ge::DT_INT32, ge::FORMAT_ND},
            {{}, ge::DT_INT32, ge::FORMAT_ND},
            {{}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{}, ge::DT_INT32, ge::FORMAT_ND},
            {{}, ge::DT_INT32, ge::FORMAT_ND},
            {{}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{8, 7168}, {8, 7168}}, ge::DT_FLOAT16, ge::FORMAT_ND},
        },
        {
            {{{8, 7168}, {8, 7168}}, ge::DT_FLOAT16, ge::FORMAT_ND},
        },
        {
            {"group_ep", build_from<std::string>("ep_group")},
            {"ep_world_size", build_from<int64_t>(8)},
            {"ep_rank_id", build_from<int64_t>(0)},
            {"moe_expert_num", build_from<int64_t>(8)},
            {"group_tp", build_from<std::string>("tp_group")},
            {"tp_world_size", build_from<int64_t>(1)},
            {"tp_rank_id", build_from<int64_t>(0)},
            {"expert_shard_type", build_from<int64_t>(0)},
            {"shared_expert_num", build_from<int64_t>(0)},
            {"shared_expert_rank_num", build_from<int64_t>(0)},
            {"global_bs", build_from<int64_t>(0)},
            {"out_dtype", build_from<int64_t>(0)},
            {"comm_quant_mode", build_from<int64_t>(0)},
            {"group_list_type", build_from<int64_t>(0)},
            {"comm_alg", build_from<std::string>("")},
            {"zero_expert_num", build_from<int64_t>(0)},
            {"copy_expert_num", build_from<int64_t>(0)},
            {"const_expert_num", build_from<int64_t>(0)}
        },
        true,
        10000UL
    },
    {
        "moe_distribute_combine_test_tiling_shared_expert_x_three_dims",
        "Ascend910_93",
        20,
        196608,
        {
            {{{64, 7168}, {64, 7168}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{8, 7}, {8, 7}}, ge::DT_INT32, ge::FORMAT_ND},
            {{{8192}, {8192}}, ge::DT_INT32, ge::FORMAT_ND},
            {{{8}, {8}}, ge::DT_INT32, ge::FORMAT_ND},
            {{{8, 7}, {8, 7}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{1}, {1}}, ge::DT_INT32, ge::FORMAT_ND},
            {{}, ge::DT_INT32, ge::FORMAT_ND},
            {{}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{}, ge::DT_INT32, ge::FORMAT_ND},
            {{}, ge::DT_INT32, ge::FORMAT_ND},
            {{}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{2, 4, 7168}, {2, 4, 7168}}, ge::DT_FLOAT16, ge::FORMAT_ND},
        },
        {
            {{{8, 7168}, {8, 7168}}, ge::DT_FLOAT16, ge::FORMAT_ND},
        },
        {
            {"group_ep", build_from<std::string>("ep_group")},
            {"ep_world_size", build_from<int64_t>(8)},
            {"ep_rank_id", build_from<int64_t>(0)},
            {"moe_expert_num", build_from<int64_t>(8)},
            {"group_tp", build_from<std::string>("tp_group")},
            {"tp_world_size", build_from<int64_t>(1)},
            {"tp_rank_id", build_from<int64_t>(0)},
            {"expert_shard_type", build_from<int64_t>(0)},
            {"shared_expert_num", build_from<int64_t>(0)},
            {"shared_expert_rank_num", build_from<int64_t>(0)},
            {"global_bs", build_from<int64_t>(0)},
            {"out_dtype", build_from<int64_t>(0)},
            {"comm_quant_mode", build_from<int64_t>(0)},
            {"group_list_type", build_from<int64_t>(0)},
            {"comm_alg", build_from<std::string>("")},
            {"zero_expert_num", build_from<int64_t>(0)},
            {"copy_expert_num", build_from<int64_t>(0)},
            {"const_expert_num", build_from<int64_t>(0)}
        },
        true,
        10000UL
    },
    {
        "moe_distribute_combine_test_tiling_0",
        "Ascend910_93",
        20,
        196608,
        {
            {{{64, 7168}, {64, 7168}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{8, 7}, {8, 7}}, ge::DT_INT32, ge::FORMAT_ND},
            {{{16384}, {16384}}, ge::DT_INT32, ge::FORMAT_ND},
            {{{8}, {8}}, ge::DT_INT32, ge::FORMAT_ND},
            {{{8, 7}, {8, 7}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{1}, {1}}, ge::DT_INT32, ge::FORMAT_ND},
        },
        {
            {{{8, 7168}, {8, 7168}}, ge::DT_FLOAT16, ge::FORMAT_ND},
        },
        {
            {"group_ep", build_from<std::string>("ep_group")},
            {"ep_world_size", build_from<int64_t>(8)},
            {"ep_rank_id", build_from<int64_t>(0)},
            {"moe_expert_num", build_from<int64_t>(7)},
            {"group_tp", build_from<std::string>("tp_group")},
            {"tp_world_size", build_from<int64_t>(1)},
            {"tp_rank_id", build_from<int64_t>(0)},
            {"expert_shard_type", build_from<int64_t>(0)},
            {"shared_expert_num", build_from<int64_t>(1)},
            {"shared_expert_rank_num", build_from<int64_t>(1)},
            {"global_bs", build_from<int64_t>(0)},
            {"out_dtype", build_from<int64_t>(0)},
            {"comm_quant_mode", build_from<int64_t>(0)},
            {"group_list_type", build_from<int64_t>(0)},
            {"comm_alg", build_from<std::string>("")},
            {"zero_expert_num", build_from<int64_t>(0)},
            {"copy_expert_num", build_from<int64_t>(0)},
            {"const_expert_num", build_from<int64_t>(0)}
        },
        true,
        10000UL
    },
    {
        "moe_distribute_combine_test_tiling_1",
        "Ascend910_93",
        20,
        196608,
        {
            {{{576, 7160}, {576, 7160}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{16, 8}, {16, 8}}, ge::DT_INT32, ge::FORMAT_ND},
            {{{256}, {256}}, ge::DT_INT32, ge::FORMAT_ND},
            {{{288}, {288}}, ge::DT_INT32, ge::FORMAT_ND},
            {{{2}, {2}}, ge::DT_INT32, ge::FORMAT_ND},
            {{{32, 8}, {32, 8}}, ge::DT_FLOAT, ge::FORMAT_ND},
        },
        {
            {{{32, 7160}, {32, 7160}}, ge::DT_FLOAT16, ge::FORMAT_ND},
        },
        {
            {"group_ep", build_from<std::string>("ep_group")},
            {"ep_world_size", build_from<int64_t>(288)},
            {"ep_rank_id", build_from<int64_t>(0)},
            {"moe_expert_num", build_from<int64_t>(256)},
            {"group_tp", build_from<std::string>("tp_group")},
            {"tp_world_size", build_from<int64_t>(2)},
            {"tp_rank_id", build_from<int64_t>(0)},
            {"expert_shard_type", build_from<int64_t>(0)},
            {"shared_expert_num", build_from<int64_t>(1)},
            {"shared_expert_rank_num", build_from<int64_t>(32)},
            {"global_bs", build_from<int64_t>(0)},
            {"out_dtype", build_from<int64_t>(0)},
            {"comm_quant_mode", build_from<int64_t>(0)},
            {"group_list_type", build_from<int64_t>(0)},
            {"comm_alg", build_from<std::string>("")},
            {"zero_expert_num", build_from<int64_t>(0)},
            {"copy_expert_num", build_from<int64_t>(0)},
            {"const_expert_num", build_from<int64_t>(0)}
        },
        false,
        0UL
    },
    {
        "moe_distribute_combine_test_tiling_2",
        "Ascend910_93",
        20,
        196608,
        {
            {{{576, 7160}, {576, 7160}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{16, 8}, {16, 8}}, ge::DT_INT32, ge::FORMAT_ND},
            {{{256}, {256}}, ge::DT_INT32, ge::FORMAT_ND},
            {{{288}, {288}}, ge::DT_INT32, ge::FORMAT_ND},
            {{{2}, {2}}, ge::DT_INT32, ge::FORMAT_ND},
            {{{32, 8}, {32, 8}}, ge::DT_FLOAT, ge::FORMAT_ND},
        },
        {
            {{{32, 7160}, {32, 7160}}, ge::DT_FLOAT16, ge::FORMAT_ND},
        },
        {
            {"group_ep", build_from<std::string>("ep_group")},
            {"ep_world_size", build_from<int64_t>(288)},
            {"ep_rank_id", build_from<int64_t>(0)},
            {"moe_expert_num", build_from<int64_t>(256)},
            {"group_tp", build_from<std::string>("tp_group")},
            {"tp_world_size", build_from<int64_t>(2)},
            {"tp_rank_id", build_from<int64_t>(1024)},
            {"expert_shard_type", build_from<int64_t>(0)},
            {"shared_expert_num", build_from<int64_t>(1)},
            {"shared_expert_rank_num", build_from<int64_t>(32)},
            {"global_bs", build_from<int64_t>(0)},
            {"out_dtype", build_from<int64_t>(0)},
            {"comm_quant_mode", build_from<int64_t>(0)},
            {"group_list_type", build_from<int64_t>(0)},
            {"comm_alg", build_from<std::string>("")},
            {"zero_expert_num", build_from<int64_t>(0)},
            {"copy_expert_num", build_from<int64_t>(0)},
            {"const_expert_num", build_from<int64_t>(0)}
        },
        false,
        0UL
    },
    {
        "moe_distribute_combine_test_tiling_3",
        "Ascend910_93",
        20,
        196608,
        {
            {{{576, 7160}, {576, 7160}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{16, 8}, {16, 8}}, ge::DT_INT32, ge::FORMAT_ND},
            {{{256}, {256}}, ge::DT_INT32, ge::FORMAT_ND},
            {{{288}, {288}}, ge::DT_INT32, ge::FORMAT_ND},
            {{{2}, {2}}, ge::DT_INT32, ge::FORMAT_ND},
            {{{32, 8}, {32, 8}}, ge::DT_FLOAT, ge::FORMAT_ND},
        },
        {
            {{{32, 7160}, {32, 7160}}, ge::DT_FLOAT16, ge::FORMAT_ND},
        },
        {
            {"group_ep", build_from<std::string>("ep_group")},
            {"ep_world_size", build_from<int64_t>(288)},
            {"ep_rank_id", build_from<int64_t>(0)},
            {"moe_expert_num", build_from<int64_t>(256)},
            {"group_tp", build_from<std::string>("tp_group")},
            {"tp_world_size", build_from<int64_t>(2)},
            {"tp_rank_id", build_from<int64_t>(0)},
            {"expert_shard_type", build_from<int64_t>(0)},
            {"shared_expert_num", build_from<int64_t>(1)},
            {"shared_expert_rank_num", build_from<int64_t>(31)},
            {"global_bs", build_from<int64_t>(0)},
            {"out_dtype", build_from<int64_t>(0)},
            {"comm_quant_mode", build_from<int64_t>(0)},
            {"group_list_type", build_from<int64_t>(0)},
            {"comm_alg", build_from<std::string>("")},
            {"zero_expert_num", build_from<int64_t>(0)},
            {"copy_expert_num", build_from<int64_t>(0)},
            {"const_expert_num", build_from<int64_t>(0)}
        },
        false,
        0UL
    },
    {
        "moe_distribute_combine_test_tiling_ep_world_size_384",
        "Ascend910_93",
        20,
        196608,
        {
            {{{576, 7168}, {576, 7168}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{32, 8}, {32, 8}}, ge::DT_INT32, ge::FORMAT_ND},
            {{{256}, {256}}, ge::DT_INT32, ge::FORMAT_ND},
            {{{288}, {288}}, ge::DT_INT32, ge::FORMAT_ND},
            {{{32, 8}, {32, 8}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{2}, {2}}, ge::DT_INT32, ge::FORMAT_ND},
        },
        {
            {{{32, 7168}, {32, 7168}}, ge::DT_FLOAT16, ge::FORMAT_ND},
        },
        {
            {"group_ep", build_from<std::string>("ep_group")},
            {"ep_world_size", build_from<int64_t>(384)},
            {"ep_rank_id", build_from<int64_t>(0)},
            {"moe_expert_num", build_from<int64_t>(256)},
            {"group_tp", build_from<std::string>("tp_group")},
            {"tp_world_size", build_from<int64_t>(2)},
            {"tp_rank_id", build_from<int64_t>(0)},
            {"expert_shard_type", build_from<int64_t>(0)},
            {"shared_expert_num", build_from<int64_t>(1)},
            {"shared_expert_rank_num", build_from<int64_t>(32)},
            {"global_bs", build_from<int64_t>(0)},
            {"out_dtype", build_from<int64_t>(0)},
            {"comm_quant_mode", build_from<int64_t>(0)},
            {"group_list_type", build_from<int64_t>(0)},
            {"comm_alg", build_from<std::string>("")},
            {"zero_expert_num", build_from<int64_t>(0)},
            {"copy_expert_num", build_from<int64_t>(0)},
            {"const_expert_num", build_from<int64_t>(0)}
        },
        false,
        0UL
    },
    {
        "moe_distribute_combine_test_tiling_ep_world_size_72",
        "Ascend910_93",
        20,
        196608,
        {
            {{{576, 7168}, {576, 7168}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{32, 8}, {32, 8}}, ge::DT_INT32, ge::FORMAT_ND},
            {{{256}, {256}}, ge::DT_INT32, ge::FORMAT_ND},
            {{{288}, {288}}, ge::DT_INT32, ge::FORMAT_ND},
            {{{32, 8}, {32, 8}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{2}, {2}}, ge::DT_INT32, ge::FORMAT_ND},
        },
        {
            {{{32, 7168}, {32, 7168}}, ge::DT_FLOAT16, ge::FORMAT_ND},
        },
        {
            {"group_ep", build_from<std::string>("ep_group")},
            {"ep_world_size", build_from<int64_t>(72)},
            {"ep_rank_id", build_from<int64_t>(0)},
            {"moe_expert_num", build_from<int64_t>(216)},
            {"group_tp", build_from<std::string>("tp_group")},
            {"tp_world_size", build_from<int64_t>(2)},
            {"tp_rank_id", build_from<int64_t>(0)},
            {"expert_shard_type", build_from<int64_t>(0)},
            {"shared_expert_num", build_from<int64_t>(1)},
            {"shared_expert_rank_num", build_from<int64_t>(32)},
            {"global_bs", build_from<int64_t>(0)},
            {"out_dtype", build_from<int64_t>(0)},
            {"comm_quant_mode", build_from<int64_t>(0)},
            {"group_list_type", build_from<int64_t>(0)},
            {"comm_alg", build_from<std::string>("")},
            {"zero_expert_num", build_from<int64_t>(0)},
            {"copy_expert_num", build_from<int64_t>(0)},
            {"const_expert_num", build_from<int64_t>(0)}
        },
        false,
        0UL
    },
    {
        "moe_distribute_combine_test_tiling_x_activate_mask_2dims",
        "Ascend910_93",
        20,
        196608,
        {
            {{{576, 7168}, {576, 7168}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{32, 8}, {32, 8}}, ge::DT_INT32, ge::FORMAT_ND},
            {{{256}, {256}}, ge::DT_INT32, ge::FORMAT_ND},
            {{{288}, {288}}, ge::DT_INT32, ge::FORMAT_ND},
            {{{32, 8}, {32, 8}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{2}, {2}}, ge::DT_INT32, ge::FORMAT_ND},
            {{{32, 8}, {32, 8}}, ge::DT_BOOL, ge::FORMAT_ND},
            {{}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{}, ge::DT_INT64, ge::FORMAT_ND},
            {{}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{}, ge::DT_INT32, ge::FORMAT_ND},
        },
        {
            {{{32, 7168}, {32, 7168}}, ge::DT_FLOAT16, ge::FORMAT_ND},
        },
        {
            {"group_ep", build_from<std::string>("ep_group")},
            {"ep_world_size", build_from<int64_t>(72)},
            {"ep_rank_id", build_from<int64_t>(0)},
            {"moe_expert_num", build_from<int64_t>(216)},
            {"group_tp", build_from<std::string>("tp_group")},
            {"tp_world_size", build_from<int64_t>(2)},
            {"tp_rank_id", build_from<int64_t>(0)},
            {"expert_shard_type", build_from<int64_t>(0)},
            {"shared_expert_num", build_from<int64_t>(1)},
            {"shared_expert_rank_num", build_from<int64_t>(18)},
            {"global_bs", build_from<int64_t>(0)},
            {"out_dtype", build_from<int64_t>(0)},
            {"comm_quant_mode", build_from<int64_t>(0)},
            {"group_list_type", build_from<int64_t>(0)},
            {"comm_alg", build_from<std::string>("")},
            {"zero_expert_num", build_from<int64_t>(0)},
            {"copy_expert_num", build_from<int64_t>(0)},
            {"const_expert_num", build_from<int64_t>(0)}
        },
        false,
        0UL
    },
    {
        "moe_distribute_combine_test_tiling_elastic_info",
        "Ascend910_93",
        20,
        196608,
        {
            {{{576, 7168}, {576, 7168}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{32, 8}, {32, 8}}, ge::DT_INT32, ge::FORMAT_ND},
            {{{256}, {256}}, ge::DT_INT32, ge::FORMAT_ND},
            {{{288}, {288}}, ge::DT_INT32, ge::FORMAT_ND},
            {{{32, 8}, {32, 8}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{2}, {2}}, ge::DT_INT32, ge::FORMAT_ND},
            {{{32, 8}, {32, 8}}, ge::DT_BOOL, ge::FORMAT_ND},
            {{}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{}, ge::DT_INT64, ge::FORMAT_ND},
            {{}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{}, ge::DT_INT32, ge::FORMAT_ND},
            {{{288}, {288}}, ge::DT_INT32, ge::FORMAT_ND},
        },
        {
            {{{32, 7168}, {32, 7168}}, ge::DT_FLOAT16, ge::FORMAT_ND},
        },
        {
            {"group_ep", build_from<std::string>("ep_group")},
            {"ep_world_size", build_from<int64_t>(72)},
            {"ep_rank_id", build_from<int64_t>(0)},
            {"moe_expert_num", build_from<int64_t>(216)},
            {"group_tp", build_from<std::string>("tp_group")},
            {"tp_world_size", build_from<int64_t>(2)},
            {"tp_rank_id", build_from<int64_t>(0)},
            {"expert_shard_type", build_from<int64_t>(0)},
            {"shared_expert_num", build_from<int64_t>(1)},
            {"shared_expert_rank_num", build_from<int64_t>(18)},
            {"global_bs", build_from<int64_t>(0)},
            {"out_dtype", build_from<int64_t>(0)},
            {"comm_quant_mode", build_from<int64_t>(0)},
            {"group_list_type", build_from<int64_t>(0)},
            {"comm_alg", build_from<std::string>("")},
            {"zero_expert_num", build_from<int64_t>(0)},
            {"copy_expert_num", build_from<int64_t>(0)},
            {"const_expert_num", build_from<int64_t>(0)}
        },
        false,
        0UL
    },
    {
        "moe_distribute_combine_test_tiling_moepp",
        "Ascend910_93",
        20,
        196608,
        {
            {{{576, 7168}, {576, 7168}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{32, 8}, {32, 8}}, ge::DT_INT32, ge::FORMAT_ND},
            {{{256}, {256}}, ge::DT_INT32, ge::FORMAT_ND},
            {{{288}, {288}}, ge::DT_INT32, ge::FORMAT_ND},
            {{{32, 8}, {32, 8}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{2}, {2}}, ge::DT_INT32, ge::FORMAT_ND},
            {{{32, 8}, {32, 8}}, ge::DT_BOOL, ge::FORMAT_ND},
            {{}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{}, ge::DT_INT64, ge::FORMAT_ND},
            {{}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{}, ge::DT_INT32, ge::FORMAT_ND},
            {{{148}, {148}}, ge::DT_INT32, ge::FORMAT_ND},
            {{{32, 7168}, {32, 7168}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{6}, {6}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{6}, {6}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{6, 7168}, {6, 7168}}, ge::DT_FLOAT16, ge::FORMAT_ND},
        },
        {
            {{{32, 7168}, {32, 7168}}, ge::DT_FLOAT16, ge::FORMAT_ND},
        },
        {
            {"group_ep", build_from<std::string>("ep_group")},
            {"ep_world_size", build_from<int64_t>(72)},
            {"ep_rank_id", build_from<int64_t>(0)},
            {"moe_expert_num", build_from<int64_t>(216)},
            {"group_tp", build_from<std::string>("tp_group")},
            {"tp_world_size", build_from<int64_t>(2)},
            {"tp_rank_id", build_from<int64_t>(0)},
            {"expert_shard_type", build_from<int64_t>(0)},
            {"shared_expert_num", build_from<int64_t>(1)},
            {"shared_expert_rank_num", build_from<int64_t>(18)},
            {"global_bs", build_from<int64_t>(0)},
            {"out_dtype", build_from<int64_t>(0)},
            {"comm_quant_mode", build_from<int64_t>(0)},
            {"group_list_type", build_from<int64_t>(0)},
            {"comm_alg", build_from<std::string>("")},
            {"zero_expert_num", build_from<int64_t>(6)},
            {"copy_expert_num", build_from<int64_t>(6)},
            {"const_expert_num", build_from<int64_t>(6)}
        },
        false,
        0UL
    },
    {
        "moe_distribute_combine_test_tiling_copyExpert_without_OriX",
        "Ascend910_93",
        20,
        196608,
        {
            {{{576, 7168}, {576, 7168}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{32, 8}, {32, 8}}, ge::DT_INT32, ge::FORMAT_ND},
            {{{256}, {256}}, ge::DT_INT32, ge::FORMAT_ND},
            {{{288}, {288}}, ge::DT_INT32, ge::FORMAT_ND},
            {{{32, 8}, {32, 8}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{2}, {2}}, ge::DT_INT32, ge::FORMAT_ND},
            {{{32, 8}, {32, 8}}, ge::DT_BOOL, ge::FORMAT_ND},
            {{}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{}, ge::DT_INT64, ge::FORMAT_ND},
            {{}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{}, ge::DT_INT32, ge::FORMAT_ND},
            {{{148}, {148}}, ge::DT_INT32, ge::FORMAT_ND},
            {{}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{6}, {6}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{6}, {6}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{6, 7168}, {6, 7168}}, ge::DT_FLOAT16, ge::FORMAT_ND},
        },
        {
            {{{32, 7168}, {32, 7168}}, ge::DT_FLOAT16, ge::FORMAT_ND},
        },
        {
            {"group_ep", build_from<std::string>("ep_group")},
            {"ep_world_size", build_from<int64_t>(72)},
            {"ep_rank_id", build_from<int64_t>(0)},
            {"moe_expert_num", build_from<int64_t>(216)},
            {"group_tp", build_from<std::string>("tp_group")},
            {"tp_world_size", build_from<int64_t>(2)},
            {"tp_rank_id", build_from<int64_t>(0)},
            {"expert_shard_type", build_from<int64_t>(0)},
            {"shared_expert_num", build_from<int64_t>(1)},
            {"shared_expert_rank_num", build_from<int64_t>(18)},
            {"global_bs", build_from<int64_t>(0)},
            {"out_dtype", build_from<int64_t>(0)},
            {"comm_quant_mode", build_from<int64_t>(0)},
            {"group_list_type", build_from<int64_t>(0)},
            {"comm_alg", build_from<std::string>("")},
            {"zero_expert_num", build_from<int64_t>(6)},
            {"copy_expert_num", build_from<int64_t>(6)},
            {"const_expert_num", build_from<int64_t>(6)}
        },
        false,
        0UL
    },
    {
        "moe_distribute_combine_test_tiling_constExpert_without_OriX",
        "Ascend910_93",
        20,
        196608,
        {
            {{{576, 7168}, {576, 7168}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{32, 8}, {32, 8}}, ge::DT_INT32, ge::FORMAT_ND},
            {{{256}, {256}}, ge::DT_INT32, ge::FORMAT_ND},
            {{{288}, {288}}, ge::DT_INT32, ge::FORMAT_ND},
            {{{32, 8}, {32, 8}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{2}, {2}}, ge::DT_INT32, ge::FORMAT_ND},
            {{{32, 8}, {32, 8}}, ge::DT_BOOL, ge::FORMAT_ND},
            {{}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{}, ge::DT_INT64, ge::FORMAT_ND},
            {{}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{}, ge::DT_INT32, ge::FORMAT_ND},
            {{{148}, {148}}, ge::DT_INT32, ge::FORMAT_ND},
            {{}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{6}, {6}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{6}, {6}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{6, 7168}, {6, 7168}}, ge::DT_FLOAT16, ge::FORMAT_ND},
        },
        {
            {{{32, 7168}, {32, 7168}}, ge::DT_FLOAT16, ge::FORMAT_ND},
        },
        {
            {"group_ep", build_from<std::string>("ep_group")},
            {"ep_world_size", build_from<int64_t>(72)},
            {"ep_rank_id", build_from<int64_t>(0)},
            {"moe_expert_num", build_from<int64_t>(216)},
            {"group_tp", build_from<std::string>("tp_group")},
            {"tp_world_size", build_from<int64_t>(2)},
            {"tp_rank_id", build_from<int64_t>(0)},
            {"expert_shard_type", build_from<int64_t>(0)},
            {"shared_expert_num", build_from<int64_t>(1)},
            {"shared_expert_rank_num", build_from<int64_t>(18)},
            {"global_bs", build_from<int64_t>(0)},
            {"out_dtype", build_from<int64_t>(0)},
            {"comm_quant_mode", build_from<int64_t>(0)},
            {"group_list_type", build_from<int64_t>(0)},
            {"comm_alg", build_from<std::string>("")},
            {"zero_expert_num", build_from<int64_t>(0)},
            {"copy_expert_num", build_from<int64_t>(0)},
            {"const_expert_num", build_from<int64_t>(0)}
        },
        false,
        0UL
    },
    {
        "moe_distribute_combine_test_tiling_a2_commalg_empty",
        "Ascend910B",
        48,
        196608,
        {
            {{{2048, 7168}, {2048, 7168}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{8, 8}, {8, 8}}, ge::DT_INT32, ge::FORMAT_ND},
            {{{64}, {64}}, ge::DT_INT32, ge::FORMAT_ND},
            {{{256}, {256}}, ge::DT_INT32, ge::FORMAT_ND},
            {{{1}, {1}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{8, 8}, {8, 8}}, ge::DT_INT32, ge::FORMAT_ND},
        },
        {
            {{{8, 7168}, {8, 7168}}, ge::DT_FLOAT16, ge::FORMAT_ND},
        },
        {
            {"group_ep", build_from<std::string>("ep_group")},
            {"ep_world_size", build_from<int64_t>(32)},
            {"ep_rank_id", build_from<int64_t>(0)},
            {"moe_expert_num", build_from<int64_t>(256)},
            {"group_tp", build_from<std::string>("")},
            {"tp_world_size", build_from<int64_t>(0)},
            {"tp_rank_id", build_from<int64_t>(0)},
            {"expert_shard_type", build_from<int64_t>(0)},
            {"shared_expert_num", build_from<int64_t>(1)},
            {"shared_expert_rank_num", build_from<int64_t>(32)},
            {"global_bs", build_from<int64_t>(0)},
            {"out_dtype", build_from<int64_t>(0)},
            {"comm_quant_mode", build_from<int64_t>(0)},
            {"group_list_type", build_from<int64_t>(0)},
            {"comm_alg", build_from<std::string>("")},
            {"zero_expert_num", build_from<int64_t>(0)},
            {"copy_expert_num", build_from<int64_t>(0)},
            {"const_expert_num", build_from<int64_t>(0)}
        },
        true,
        2000UL
    },
    {
        "moe_distribute_combine_test_tiling_a2_commalg_empty_with_env",
        "Ascend910B",
        48,
        196608,
        {
            {{{2048, 7168}, {2048, 7168}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{8, 8}, {8, 8}}, ge::DT_INT32, ge::FORMAT_ND},
            {{{64}, {64}}, ge::DT_INT32, ge::FORMAT_ND},
            {{{256}, {256}}, ge::DT_INT32, ge::FORMAT_ND},
            {{{1}, {1}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{8, 8}, {8, 8}}, ge::DT_INT32, ge::FORMAT_ND},
        },
        {
            {{{8, 7168}, {8, 7168}}, ge::DT_FLOAT16, ge::FORMAT_ND},
        },
        {
            {"group_ep", build_from<std::string>("ep_group")},
            {"ep_world_size", build_from<int64_t>(32)},
            {"ep_rank_id", build_from<int64_t>(0)},
            {"moe_expert_num", build_from<int64_t>(256)},
            {"group_tp", build_from<std::string>("")},
            {"tp_world_size", build_from<int64_t>(0)},
            {"tp_rank_id", build_from<int64_t>(0)},
            {"expert_shard_type", build_from<int64_t>(0)},
            {"shared_expert_num", build_from<int64_t>(1)},
            {"shared_expert_rank_num", build_from<int64_t>(32)},
            {"global_bs", build_from<int64_t>(0)},
            {"out_dtype", build_from<int64_t>(0)},
            {"comm_quant_mode", build_from<int64_t>(0)},
            {"group_list_type", build_from<int64_t>(0)},
            {"comm_alg", build_from<std::string>("")},
            {"zero_expert_num", build_from<int64_t>(0)},
            {"copy_expert_num", build_from<int64_t>(0)},
            {"const_expert_num", build_from<int64_t>(0)}
        },
        true,
        2000UL
    },
    {
        "moe_distribute_combine_test_tiling_a2_commalg_fullmesh",
        "Ascend910B",
        48,
        196608,
        {
            {{{2048, 7168}, {2048, 7168}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{8, 8}, {8, 8}}, ge::DT_INT32, ge::FORMAT_ND},
            {{{64}, {64}}, ge::DT_INT32, ge::FORMAT_ND},
            {{{256}, {256}}, ge::DT_INT32, ge::FORMAT_ND},
            {{{1}, {1}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{8, 8}, {8, 8}}, ge::DT_INT32, ge::FORMAT_ND},
        },
        {
            {{{8, 7168}, {8, 7168}}, ge::DT_FLOAT16, ge::FORMAT_ND},
        },
        {
            {"group_ep", build_from<std::string>("ep_group")},
            {"ep_world_size", build_from<int64_t>(32)},
            {"ep_rank_id", build_from<int64_t>(0)},
            {"moe_expert_num", build_from<int64_t>(256)},
            {"group_tp", build_from<std::string>("")},
            {"tp_world_size", build_from<int64_t>(0)},
            {"tp_rank_id", build_from<int64_t>(0)},
            {"expert_shard_type", build_from<int64_t>(0)},
            {"shared_expert_num", build_from<int64_t>(1)},
            {"shared_expert_rank_num", build_from<int64_t>(32)},
            {"global_bs", build_from<int64_t>(0)},
            {"out_dtype", build_from<int64_t>(0)},
            {"comm_quant_mode", build_from<int64_t>(0)},
            {"group_list_type", build_from<int64_t>(0)},
            {"comm_alg", build_from<std::string>("fullmesh")},
            {"zero_expert_num", build_from<int64_t>(0)},
            {"copy_expert_num", build_from<int64_t>(0)},
            {"const_expert_num", build_from<int64_t>(0)}
        },
        true,
        2000UL
    },
    {
        "moe_distribute_combine_test_tiling_a2_commalg_fullmesh_with_env",
        "Ascend910B",
        48,
        196608,
        {
            {{{2048, 7168}, {2048, 7168}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{8, 8}, {8, 8}}, ge::DT_INT32, ge::FORMAT_ND},
            {{{64}, {64}}, ge::DT_INT32, ge::FORMAT_ND},
            {{{256}, {256}}, ge::DT_INT32, ge::FORMAT_ND},
            {{{1}, {1}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{8, 8}, {8, 8}}, ge::DT_INT32, ge::FORMAT_ND},
        },
        {
            {{{8, 7168}, {8, 7168}}, ge::DT_FLOAT16, ge::FORMAT_ND},
        },
        {
            {"group_ep", build_from<std::string>("ep_group")},
            {"ep_world_size", build_from<int64_t>(32)},
            {"ep_rank_id", build_from<int64_t>(0)},
            {"moe_expert_num", build_from<int64_t>(256)},
            {"group_tp", build_from<std::string>("")},
            {"tp_world_size", build_from<int64_t>(0)},
            {"tp_rank_id", build_from<int64_t>(0)},
            {"expert_shard_type", build_from<int64_t>(0)},
            {"shared_expert_num", build_from<int64_t>(1)},
            {"shared_expert_rank_num", build_from<int64_t>(32)},
            {"global_bs", build_from<int64_t>(0)},
            {"out_dtype", build_from<int64_t>(0)},
            {"comm_quant_mode", build_from<int64_t>(0)},
            {"group_list_type", build_from<int64_t>(0)},
            {"comm_alg", build_from<std::string>("fullmesh")},
            {"zero_expert_num", build_from<int64_t>(0)},
            {"copy_expert_num", build_from<int64_t>(0)},
            {"const_expert_num", build_from<int64_t>(0)}
        },
        true,
        2000UL
    },
    {
        "moe_distribute_combine_test_tiling_a2_commalg_hierarchy",
        "Ascend910B",
        48,
        196608,
        {
            {{{2048, 7168}, {2048, 7168}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{8, 8}, {8, 8}}, ge::DT_INT32, ge::FORMAT_ND},
            {{{64}, {64}}, ge::DT_INT32, ge::FORMAT_ND},
            {{{256}, {256}}, ge::DT_INT32, ge::FORMAT_ND},
            {{{1}, {1}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{8, 8}, {8, 8}}, ge::DT_INT32, ge::FORMAT_ND},
        },
        {
            {{{8, 7168}, {8, 7168}}, ge::DT_FLOAT16, ge::FORMAT_ND},
        },
        {
            {"group_ep", build_from<std::string>("ep_group")},
            {"ep_world_size", build_from<int64_t>(32)},
            {"ep_rank_id", build_from<int64_t>(0)},
            {"moe_expert_num", build_from<int64_t>(256)},
            {"group_tp", build_from<std::string>("")},
            {"tp_world_size", build_from<int64_t>(0)},
            {"tp_rank_id", build_from<int64_t>(0)},
            {"expert_shard_type", build_from<int64_t>(0)},
            {"shared_expert_num", build_from<int64_t>(1)},
            {"shared_expert_rank_num", build_from<int64_t>(32)},
            {"global_bs", build_from<int64_t>(0)},
            {"out_dtype", build_from<int64_t>(0)},
            {"comm_quant_mode", build_from<int64_t>(0)},
            {"group_list_type", build_from<int64_t>(0)},
            {"comm_alg", build_from<std::string>("hierarchy")},
            {"zero_expert_num", build_from<int64_t>(0)},
            {"copy_expert_num", build_from<int64_t>(0)},
            {"const_expert_num", build_from<int64_t>(0)}
        },
        true,
        3000UL
    },
    {
        "moe_distribute_combine_test_tiling_a2_commalg_error",
        "Ascend910B",
        48,
        196608,
        {
            {{{2048, 7168}, {2048, 7168}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{8, 8}, {8, 8}}, ge::DT_INT32, ge::FORMAT_ND},
            {{{64}, {64}}, ge::DT_INT32, ge::FORMAT_ND},
            {{{256}, {256}}, ge::DT_INT32, ge::FORMAT_ND},
            {{{1}, {1}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{8, 8}, {8, 8}}, ge::DT_INT32, ge::FORMAT_ND},
        },
        {
            {{{8, 7168}, {8, 7168}}, ge::DT_FLOAT16, ge::FORMAT_ND},
        },
        {
            {"group_ep", build_from<std::string>("ep_group")},
            {"ep_world_size", build_from<int64_t>(32)},
            {"ep_rank_id", build_from<int64_t>(0)},
            {"moe_expert_num", build_from<int64_t>(256)},
            {"group_tp", build_from<std::string>("")},
            {"tp_world_size", build_from<int64_t>(0)},
            {"tp_rank_id", build_from<int64_t>(0)},
            {"expert_shard_type", build_from<int64_t>(0)},
            {"shared_expert_num", build_from<int64_t>(1)},
            {"shared_expert_rank_num", build_from<int64_t>(32)},
            {"global_bs", build_from<int64_t>(0)},
            {"out_dtype", build_from<int64_t>(0)},
            {"comm_quant_mode", build_from<int64_t>(0)},
            {"group_list_type", build_from<int64_t>(0)},
            {"comm_alg", build_from<std::string>("error")},
            {"zero_expert_num", build_from<int64_t>(0)},
            {"copy_expert_num", build_from<int64_t>(0)},
            {"const_expert_num", build_from<int64_t>(0)}
        },
        false,
        0UL
    },
    {
        "moe_distribute_combine_test_tiling_a2_commalg_empty_with_env_commint8",
        "Ascend910B",
        48,
        196608,
        {
            {{{2048, 7168}, {2048, 7168}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{8, 8}, {8, 8}}, ge::DT_INT32, ge::FORMAT_ND},
            {{{64}, {64}}, ge::DT_INT32, ge::FORMAT_ND},
            {{{256}, {256}}, ge::DT_INT32, ge::FORMAT_ND},
            {{{1}, {1}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{8, 8}, {8, 8}}, ge::DT_INT32, ge::FORMAT_ND},
        },
        {
            {{{8, 7168}, {8, 7168}}, ge::DT_FLOAT16, ge::FORMAT_ND},
        },
        {
            {"group_ep", build_from<std::string>("ep_group")},
            {"ep_world_size", build_from<int64_t>(32)},
            {"ep_rank_id", build_from<int64_t>(0)},
            {"moe_expert_num", build_from<int64_t>(256)},
            {"group_tp", build_from<std::string>("")},
            {"tp_world_size", build_from<int64_t>(0)},
            {"tp_rank_id", build_from<int64_t>(0)},
            {"expert_shard_type", build_from<int64_t>(0)},
            {"shared_expert_num", build_from<int64_t>(1)},
            {"shared_expert_rank_num", build_from<int64_t>(32)},
            {"global_bs", build_from<int64_t>(0)},
            {"out_dtype", build_from<int64_t>(0)},
            {"comm_quant_mode", build_from<int64_t>(0)},
            {"group_list_type", build_from<int64_t>(0)},
            {"comm_alg", build_from<std::string>("")},
            {"zero_expert_num", build_from<int64_t>(0)},
            {"copy_expert_num", build_from<int64_t>(0)},
            {"const_expert_num", build_from<int64_t>(0)}
        },
        true,
        2000UL
    },
    {
        "moe_distribute_combine_test_tiling_a2_commalg_hierarchy_commint8",
        "Ascend910B",
        48,
        196608,
        {
            {{{2048, 7168}, {2048, 7168}}, ge::DT_FLOAT16, ge::FORMAT_ND},
            {{{8, 8}, {8, 8}}, ge::DT_INT32, ge::FORMAT_ND},
            {{{64}, {64}}, ge::DT_INT32, ge::FORMAT_ND},
            {{{256}, {256}}, ge::DT_INT32, ge::FORMAT_ND},
            {{{1}, {1}}, ge::DT_FLOAT, ge::FORMAT_ND},
            {{{8, 8}, {8, 8}}, ge::DT_INT32, ge::FORMAT_ND},
        },
        {
            {{{8, 7168}, {8, 7168}}, ge::DT_FLOAT16, ge::FORMAT_ND},
        },
        {
            {"group_ep", build_from<std::string>("ep_group")},
            {"ep_world_size", build_from<int64_t>(32)},
            {"ep_rank_id", build_from<int64_t>(0)},
            {"moe_expert_num", build_from<int64_t>(256)},
            {"group_tp", build_from<std::string>("")},
            {"tp_world_size", build_from<int64_t>(0)},
            {"tp_rank_id", build_from<int64_t>(0)},
            {"expert_shard_type", build_from<int64_t>(0)},
            {"shared_expert_num", build_from<int64_t>(1)},
            {"shared_expert_rank_num", build_from<int64_t>(32)},
            {"global_bs", build_from<int64_t>(0)},
            {"out_dtype", build_from<int64_t>(0)},
            {"comm_quant_mode", build_from<int64_t>(2)},
            {"group_list_type", build_from<int64_t>(0)},
            {"comm_alg", build_from<std::string>("hierarchy")},
            {"zero_expert_num", build_from<int64_t>(0)},
            {"copy_expert_num", build_from<int64_t>(0)},
            {"const_expert_num", build_from<int64_t>(0)}
        },
        true,
        3100UL
    },
};

TEST_P(MoeDistributeCombineV2TilingParam, general_case)
{
const auto &param = GetParam();
    TestOneParamCase(param);
}

INSTANTIATE_TEST_SUITE_P(
    general_cases_params,
    MoeDistributeCombineV2TilingParam,
    ::testing::ValuesIn(cases_params),
    [](const ::testing::TestParamInfo<MoeDistributeCombineV2TilingTestParam> &info) {
        std::string name = info.param.case_name;
        for (char &c : name) {
            if (!std::isalnum(static_cast<unsigned char>(c)) && c != '_') {
                c = '_';
            }
        }
        return name;
    });

}  // anonymous namespace

}  // namespace MoeDistributeCombineV2UT


