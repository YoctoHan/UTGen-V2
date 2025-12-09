/**
 * This program is free software, you can redistribute it and/or modify.
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file test_grouped_mat_mul_all_reduce_tiling.cpp
 * \brief
 */

#include <iostream>
#include <cctype>
#include <gtest/gtest.h>

#include "mc2_tiling_case_executor.h"

namespace GroupedMatMulAllReduceUT {

namespace {

template <typename T>
auto build_from(const T &value)
{
    return Ops::Transformer::AnyValue::CreateFrom<T>(value);
}

struct GroupedMatMulAllReduceTilingTestParam {
    std::string case_name;
    std::string soc_version;
    uint64_t coreNum;
    uint64_t ubSize;
    uint64_t tilingDataSize;

    std::initializer_list<int64_t> x1_shape;
    std::initializer_list<int64_t> x2_shape;
    std::initializer_list<int64_t> output_shape;

    ge::DataType x1_dtype;
    ge::DataType x2_dtype;
    ge::DataType output_dtype;

    int64_t rankNum;
    uint64_t expectTilingKey;
};

class GroupedMatMulAllReduceTilingParam
    : public ::testing::TestWithParam<GroupedMatMulAllReduceTilingTestParam> {
protected:
    static void SetUpTestCase()
    {
        std::cout << "GroupedMatMulAllReduceTiling SetUp" << std::endl;
    }

    static void TearDownTestCase()
    {
        std::cout << "GroupedMatMulAllReduceTiling TearDown" << std::endl;
    }
};

struct GroupedMatMulAllReduceCompileInfo {
    int32_t totalCoreNum = 0;
    uint64_t ubSize = 0;
};

gert::StorageShape make_shape(const std::initializer_list<int64_t> &input_shape)
{
    if (input_shape.size() == 0) {
        return gert::StorageShape{};
    }
    return gert::StorageShape{input_shape, input_shape};
}

void TestOneParamCase(const GroupedMatMulAllReduceTilingTestParam &param)
{
    GroupedMatMulAllReduceCompileInfo compileInfo {
        static_cast<int32_t>(param.coreNum),
        param.ubSize
    };

    std::vector<gert::TilingContextPara::TensorDescription> inputList = {
        {make_shape(param.x1_shape), param.x1_dtype, ge::FORMAT_ND},
        {make_shape(param.x2_shape), param.x2_dtype, ge::FORMAT_ND},
    };

    std::vector<gert::TilingContextPara::TensorDescription> outputList = {
        {make_shape(param.output_shape), param.output_dtype, ge::FORMAT_ND},
    };

    std::string group("group");
    std::string reduceOp("sum");

    gert::TilingContextPara tilingContextPara("GroupedMatMulAllReduce",
        inputList,
        outputList,
        {
            {"splitItem", build_from<int64_t>(0)},
            {"group", build_from<std::string>(group)},
            {"reduceOp", build_from<std::string>(reduceOp)},
            {"commTurn", build_from<int64_t>(0)},
        },
        &compileInfo,
        param.soc_version,
        param.coreNum,
        param.ubSize,
        param.tilingDataSize);

    Mc2Hcom::MockValues hcomTopologyMockValues{{"rankNum", param.rankNum}};
    Mc2ExecuteTestCase(tilingContextPara, hcomTopologyMockValues, ge::GRAPH_SUCCESS, param.expectTilingKey);
}


TEST_P(GroupedMatMulAllReduceTilingParam, general_case)
{
    if (!IsOpImplRegistryAvailable()) {
        GTEST_SKIP() << "Skip test: OpImplSpaceRegistryV2 is null on host.";
    }
    const auto &param = GetParam();
    TestOneParamCase(param);
}

INSTANTIATE_TEST_SUITE_P(
    general_cases_params,
    GroupedMatMulAllReduceTilingParam,
    ::testing::ValuesIn(cases_params),
    [](const ::testing::TestParamInfo<GroupedMatMulAllReduceTilingTestParam> &info) {
        std::string name = info.param.case_name;
        for (char &c : name) {
            if (!std::isalnum(static_cast<unsigned char>(c)) && c != '_') {
                c = '_';
            }
        }
        return name;
    });

} // anonymous namespace

} // namespace GroupedMatMulAllReduceUT
