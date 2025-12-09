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

namespace MatmulReduceScatterV2UT {

namespace {

template <typename T>
auto build_from(const T &value)
{
    return Ops::Transformer::AnyValue::CreateFrom<T>(value);
}

// 用例参数结构体，对齐 MatmulReduceScatter / AllGatherMatmulV2 Tiling 风格
struct MatmulReduceScatterV2TilingTestParam {
    // 平台信息
    uint64_t inputTotalNum;
    std::string case_name;
    std::string soc_version;
    uint64_t coreNum;
    uint64_t ubSize;
    uint64_t tilingDataSize;
    std::string compile_info;

    // 输入 shape
    std::initializer_list<int64_t> x1_shape;
    std::initializer_list<int64_t> x2_shape;

    // 输出 shape
    std::initializer_list<int64_t> y_shape;

    // 输入 / 输出 dtype
    ge::DataType x1_dtype;
    ge::DataType x2_dtype;
    ge::DataType y_dtype;

    bool is_trans_a;
    bool is_trans_b;

    // 结果
    uint64_t expectTilingKey;
};

class MatmulReduceScatterV2TilingParam
    : public ::testing::TestWithParam<MatmulReduceScatterV2TilingTestParam> {
protected:
    static void SetUpTestCase()
    {
        std::cout << "MatmulReduceScatterV2Tiling SetUp" << std::endl;
    }

    static void TearDownTestCase()
    {
        std::cout << "MatmulReduceScatterV2Tiling TearDown" << std::endl;
    }
};

gert::StorageShape make_shape(const std::initializer_list<int64_t> &input_shape)
{
    if (input_shape.size() == 0) {
        return gert::StorageShape{};
    }
    return gert::StorageShape{input_shape, input_shape};
}

void TestOneParamCase(const MatmulReduceScatterV2TilingTestParam &param)
{
    struct MatmulReduceScatterV2CompileInfo {};
    MatmulReduceScatterV2CompileInfo compileInfo;

    std::vector<std::pair<std::initializer_list<int64_t>, ge::DataType>> shapeDtypeList = {
        {param.x1_shape, param.x1_dtype},
        {param.x2_shape, param.x2_dtype},
    };

    std::vector<gert::TilingContextPara::TensorDescription> inputList;
    for (uint64_t i = 0; i < param.inputTotalNum; ++i) {
        inputList.push_back({make_shape(shapeDtypeList[i].first), shapeDtypeList[i].second, ge::FORMAT_ND});
    }

    std::vector<gert::TilingContextPara::TensorDescription> outputList = {
        {make_shape(param.y_shape), param.y_dtype, ge::FORMAT_ND},
    };

    gert::TilingContextPara tilingContextPara("MatmulReduceScatterV2",
        inputList,
        outputList,
        {
            {"group", build_from<std::string>("group")},
            {"reduce_op", build_from<std::string>("sum")},
            {"is_trans_a", build_from<bool>(param.is_trans_a)},
            {"is_trans_b", build_from<bool>(param.is_trans_b)},
            {"comm_turn", build_from<int64_t>(0)},
            {"rank_size", build_from<int64_t>(0)},
            {"block_size", build_from<int64_t>(0)},
            {"group_size", build_from<int64_t>(0)},
            {"is_amax_out", build_from<bool>(false)},
            {"y_dtype", build_from<int64_t>(static_cast<int64_t>(ge::DT_FLOAT16))},
            {"comm_mode", build_from<std::string>("aicpu")},
        },
        &compileInfo, param.soc_version, param.coreNum, param.ubSize, param.tilingDataSize, param.compile_info);

    Mc2Hcom::MockValues hcomTopologyMockValues{{"rankNum", 8}};
    Mc2ExecuteTestCase(tilingContextPara, hcomTopologyMockValues, ge::GRAPH_SUCCESS, param.expectTilingKey);
}

TEST_P(MatmulReduceScatterV2TilingParam, general_case)
{
    if (!IsOpImplRegistryAvailable()) {
        GTEST_SKIP() << "Skip test: OpImplSpaceRegistryV2 is null on host.";
    }
    const auto &param = GetParam();
    TestOneParamCase(param);
}

INSTANTIATE_TEST_SUITE_P(
    general_cases_params_v2,
    MatmulReduceScatterV2TilingParam,
    ::testing::ValuesIn(cases_params),
    [](const ::testing::TestParamInfo<MatmulReduceScatterV2TilingTestParam> &info) {
        std::string name = info.param.case_name;
        for (char &c : name) {
            if (!std::isalnum(static_cast<unsigned char>(c)) && c != '_') {
                c = '_';
            }
        }
        return name;
    });

} // anonymous namespace

} // namespace MatmulReduceScatterV2UT

