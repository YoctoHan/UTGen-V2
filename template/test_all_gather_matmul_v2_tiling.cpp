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
#include <cctype>
#include <gtest/gtest.h>
#include "../../../../../tests/ut/framework_normal/common/mc2_tiling_case_executor.h"

namespace AllGatherMatmulV2UT {

namespace {

template <typename T>
auto build_from(const T &value)
{
    return Ops::Transformer::AnyValue::CreateFrom<T>(value);
}

struct AllGatherMatmulTilingTestParam {
    uint64_t inputTotalNum;
    std::string case_name;
    std::string compile_info;
    std::string soc_version;
    uint64_t coreNum;
    uint64_t ubSize;
    uint64_t tilingDataSize;

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
    std::initializer_list<int64_t> output_shape;

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
    ge::DataType output_dtype;

    bool is_trans_a;
    bool is_trans_b;

    bool expectSuccess;
    uint64_t expectTilingKey;
};

class AllGatherMatmulV2TilingParam : public ::testing::TestWithParam<AllGatherMatmulTilingTestParam> {
protected:
    static void SetUpTestCase()
    {
        std::cout << "AllGatherMatmulV2Tiling SetUp" << std::endl;
    }

    static void TearDownTestCase()
    {
        std::cout << "AllGatherMatmulV2Tiling TearDown" << std::endl;
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

    std::vector<gert::TilingContextPara::TensorDescription> inputList;
    for (uint64_t i = 0; i < param.inputTotalNum; ++i) {
        inputList.push_back({make_shape(shapeDtypeList[i].first), shapeDtypeList[i].second, ge::FORMAT_ND});
    }

    std::vector<gert::TilingContextPara::TensorDescription> outputList = {
        {{param.output_shape, param.output_shape}, param.output_dtype, ge::FORMAT_ND},
        {make_shape(param.x1_shape), param.x1_dtype, ge::FORMAT_ND},
    };

    gert::TilingContextPara tilingContextPara("AllGatherMatmulV2", inputList, outputList,
        {
            {"group", build_from<std::string>("group")},
            {"is_trans_a", build_from<bool>(param.is_trans_a)},
            {"is_trans_b", build_from<bool>(param.is_trans_b)},
            {"gather_index", build_from<int64_t>(0)},
            {"comm_turn", build_from<int64_t>(0)},
            {"rank_size", build_from<int64_t>(0)},
            {"block_size", build_from<int64_t>(0)},
            {"group_size", build_from<int64_t>(0)},
            {"is_gather_out", build_from<int64_t>(0)},
            {"is_amax_out", build_from<int64_t>(0)},
            {"y_dtype", build_from<int64_t>(0)},
        },
        &compileInfo, param.soc_version, param.coreNum, param.ubSize, param.tilingDataSize, param.compile_info);

    Mc2Hcom::MockValues hcomTopologyMockValues{{"rankNum", 8}};
    if (!param.expectSuccess) {
        Mc2ExecuteTestCase(tilingContextPara, hcomTopologyMockValues, ge::GRAPH_FAILED, 0);
    } else {
        Mc2ExecuteTestCase(tilingContextPara, hcomTopologyMockValues, ge::GRAPH_SUCCESS, param.expectTilingKey);
    }
}

TEST_P(AllGatherMatmulV2TilingParam, general_case)
{
    if (!IsOpImplRegistryAvailable()) {
        GTEST_SKIP() << "Skip test: OpImplSpaceRegistryV2 is null on host.";
    }
    const auto &param = GetParam();
    TestOneParamCase(param);
}

INSTANTIATE_TEST_SUITE_P(
    general_cases_params_v2,
    AllGatherMatmulV2TilingParam,
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

} // namespace AllGatherMatmulV2UT


