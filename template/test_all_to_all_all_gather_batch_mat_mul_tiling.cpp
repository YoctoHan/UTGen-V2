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

struct AlltoAllAllGatherBmmTilingTestParam {
    std::string case_name;
    std::string soc_version;
    uint64_t coreNum;
    uint64_t ubSize;
    uint64_t tilingDataSize;

    std::vector<std::vector<int64_t>> input_shapes;
    std::vector<ge::DataType> input_dtypes;
    std::vector<int64_t> output_shape;
    ge::DataType output_dtype;

    std::string group_ep;
    std::string group_tp;
    int64_t ep_world_size;
    int64_t tp_world_size;
    int64_t x_shard_type;
    int64_t act_type;
    bool transpose_weight;
    bool output_y2_flag;
    bool output_y3_flag;

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

TEST_P(AlltoAllAllGatherBmmTilingParam, general_case)
{
    const auto &param = GetParam();
    TestOneParamCase(param);
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

