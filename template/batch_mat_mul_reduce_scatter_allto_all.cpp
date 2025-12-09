/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd. All rights reserved.
 */
#include <iostream>
#include <vector>
#include <string>
#include <gtest/gtest.h>
#include "mc2_tiling_case_executor.h"

namespace BatchMatMulReduceScatterAlltoAllUT {
namespace {
template <typename T>
auto build_from(const T &value) { return Ops::Transformer::AnyValue::CreateFrom<T>(value); }

gert::StorageShape make_shape(const std::initializer_list<int64_t> &input_shape)
{
    if (input_shape.size() == 0) { return gert::StorageShape{}; }
    return gert::StorageShape{input_shape, input_shape};
}

struct BatchMatMulReduceScatterAlltoAllTilingTestParam {
    uint64_t inputTotalNum;
    std::string case_name;
    std::string compile_info;
    std::string soc_version;
    uint64_t coreNum;
    uint64_t ubSize;
    uint64_t tilingDataSize;

    std::initializer_list<int64_t> x_shape;
    ge::DataType x_dtype;
    std::initializer_list<int64_t> w_shape;
    ge::DataType w_dtype;
    std::initializer_list<int64_t> bias_shape;
    ge::DataType bias_dtype;
    std::initializer_list<int64_t> y_shape;
    ge::DataType y_dtype;

    std::string group_ep;
    std::string group_tp;
    int64_t ep_world_size;
    int64_t tp_world_size;
    int64_t y_shard_type;
    bool transpose_weight;
    bool hasExpectTilingKey;
    uint64_t expectTilingKey;
};

class BatchMatMulReduceScatterAlltoAllTilingParam : public ::testing::TestWithParam<BatchMatMulReduceScatterAlltoAllTilingTestParam> {
protected:
    static void SetUpTestCase() { std::cout << "SetUp" << std::endl; }
    static void TearDownTestCase() { std::cout << "TearDown" << std::endl; }
};

void TestOneParamCase(const BatchMatMulReduceScatterAlltoAllTilingTestParam &param)
{
    struct BatchMatMulReduceScatterAlltoAllCompileInfo {} compileInfo;

    std::vector<std::pair<std::initializer_list<int64_t>, ge::DataType>> shapeDtypeList = {
        {param.x_shape, param.x_dtype},
        {param.w_shape, param.w_dtype},
        {param.bias_shape, param.bias_dtype},
    };

    std::vector<gert::TilingContextPara::TensorDescription> inputList;
    for (uint64_t i = 0; i < param.inputTotalNum; ++i) {
        inputList.push_back({make_shape(shapeDtypeList[i].first), shapeDtypeList[i].second, ge::FORMAT_ND});
    }

    std::vector<gert::TilingContextPara::TensorDescription> outputList;
    outputList.push_back({make_shape(param.y_shape), param.y_dtype, ge::FORMAT_ND});

    gert::TilingContextPara tilingContextPara("BatchMatMulReduceScatterAlltoAll", inputList, outputList,
        {
            {"group_ep", build_from<std::string>(param.group_ep)},
            {"group_tp", build_from<std::string>(param.group_tp)},
            {"ep_world_size", build_from<int64_t>(param.ep_world_size)},
            {"tp_world_size", build_from<int64_t>(param.tp_world_size)},
            {"y_shard_type", build_from<int64_t>(param.y_shard_type)},
            {"transpose_weight", build_from<bool>(param.transpose_weight)},
        },
        &compileInfo, param.soc_version, param.coreNum, param.ubSize);

    Mc2Hcom::MockValues hcomTopologyMockValues{{"rankNum", 8}};
    if (param.hasExpectTilingKey) {
        Mc2ExecuteTestCase(tilingContextPara, hcomTopologyMockValues, ge::GRAPH_SUCCESS, param.expectTilingKey);
    } else {
        Mc2ExecuteTestCase(tilingContextPara, hcomTopologyMockValues);
    }
}

TEST_P(BatchMatMulReduceScatterAlltoAllTilingParam, general_case)
{
    const auto &param = GetParam();
    TestOneParamCase(param);
}

INSTANTIATE_TEST_SUITE_P(
    general_cases_params,
    BatchMatMulReduceScatterAlltoAllTilingParam,
    ::testing::ValuesIn(cases_params),
    [](const ::testing::TestParamInfo<BatchMatMulReduceScatterAlltoAllTilingTestParam> &info) {
        std::string name = info.param.case_name;
        for (char &c : name) { if (!std::isalnum(static_cast<unsigned char>(c)) && c != '_') c = '_'; }
        return name;
    });
} // anonymous namespace
} // namespace BatchMatMulReduceScatterAlltoAllUT
