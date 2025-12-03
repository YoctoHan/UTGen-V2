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

const std::string COMPILE_INFO = "8 8 20 196352 0 0 ";

DistributeBarrierTilingTestParam cases_params[] = {
    {"distribute_barrier_basic_small", 4, 4, ge::DT_FLOAT16, "default_group", 8,
        "Ascend910_93", 20, 196608, 10000UL, COMPILE_INFO, {16777216}, 42},
    {"distribute_barrier_basic_medium", 256, 256, ge::DT_FLOAT16, "default_group", 8,
        "Ascend910_93", 20, 196608, 10000UL, COMPILE_INFO, {16777216}, 42},
    {"distribute_barrier_basic_large", 1024, 1024, ge::DT_FLOAT16, "default_group", 8,
        "Ascend910_93", 20, 196608, 10000UL, COMPILE_INFO, {16777216}, 42},
    {"distribute_barrier_world_size_2", 128, 128, ge::DT_FLOAT16, "default_group", 2,
        "Ascend910_93", 20, 196608, 10000UL, "8 2 20 196352 0 0 ", {16777216}, 42},
    {"distribute_barrier_world_size_4", 128, 128, ge::DT_FLOAT16, "default_group", 4,
        "Ascend910_93", 20, 196608, 10000UL, "8 4 20 196352 0 0 ", {16777216}, 42},
    {"distribute_barrier_world_size_16", 128, 128, ge::DT_FLOAT16, "default_group", 16,
        "Ascend910_93", 20, 196608, 10000UL, "8 16 20 196352 0 0 ", {16777216}, 42},
    {"distribute_barrier_world_size_32", 128, 128, ge::DT_FLOAT16, "default_group", 32,
        "Ascend910_93", 20, 196608, 10000UL, "8 32 20 196352 0 0 ", {16777216}, 42},
    {"distribute_barrier_custom_group", 64, 64, ge::DT_FLOAT16, "custom_group_name", 8,
        "Ascend910_93", 20, 196608, 10000UL, COMPILE_INFO, {16777216}, 42},
    {"distribute_barrier_bfloat16", 128, 128, ge::DT_BF16, "default_group", 8,
        "Ascend910_93", 20, 196608, 10000UL, COMPILE_INFO, {16777216}, 42},
    {"distribute_barrier_single_element", 1, 1, ge::DT_FLOAT16, "default_group", 8,
        "Ascend910_93", 20, 196608, 10000UL, COMPILE_INFO, {16777216}, 42},
};

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