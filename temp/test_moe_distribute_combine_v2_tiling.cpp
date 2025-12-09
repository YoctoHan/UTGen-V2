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
    // 用例基本信息
    std::string case_name;
    std::string soc_version;
    uint64_t coreNum;
    uint64_t ubSize;

    // 输入 / 输出描述
    std::vector<gert::TilingContextPara::TensorDescription> inputs;
    std::vector<gert::TilingContextPara::TensorDescription> outputs;

    // Attr 信息
    std::vector<gert::TilingContextPara::OpAttr> attrs;

    // 是否检查 tiling key
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

TEST_P(MoeDistributeCombineV2TilingParam, general_case)
{
    if (!IsOpImplRegistryAvailable()) {
        GTEST_SKIP() << "Skip test: OpImplSpaceRegistryV2 is null on host.";
    }
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


