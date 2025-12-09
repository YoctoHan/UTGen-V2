/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd. All rights reserved.
 */
#include <iostream>
#include <vector>
#include <string>
#include <gtest/gtest.h>
#include "mc2_tiling_case_executor.h"

namespace GroupedMatMulAllReduceUT {
namespace {
template <typename T>
auto build_from(const T &value) { return Ops::Transformer::AnyValue::CreateFrom<T>(value); }

gert::StorageShape make_shape(const std::initializer_list<int64_t> &input_shape)
{
    if (input_shape.size() == 0) { return gert::StorageShape{}; }
    return gert::StorageShape{input_shape, input_shape};
}

struct GroupedMatMulAllReduceTilingTestParam {
    std::string case_name;
    std::string compile_info;
    std::string soc_version;
    uint64_t coreNum;
    uint64_t ubSize;
    uint64_t tilingDataSize;

    std::initializer_list<int64_t> x_shape;
    ge::DataType x_dtype;
    std::initializer_list<int64_t> weight_shape;
    ge::DataType weight_dtype;
    std::initializer_list<int64_t> bias_shape;
    ge::DataType bias_dtype;
    std::initializer_list<int64_t> group_list_shape;
    ge::DataType group_list_dtype;
    std::initializer_list<int64_t> y_shape;
    ge::DataType y_dtype;

    int64_t splitItem;
    std::string group;
    std::string reduceOp;
    int64_t commTurn;
    uint64_t expectTilingKey;
};

class GroupedMatMulAllReduceTilingParam : public ::testing::TestWithParam<GroupedMatMulAllReduceTilingTestParam> {
protected:
    static void SetUpTestCase() { std::cout << "SetUp" << std::endl; }
    static void TearDownTestCase() { std::cout << "TearDown" << std::endl; }
};

void TestOneParamCase(const GroupedMatMulAllReduceTilingTestParam &param)
{
    struct GroupedMatMulAllReduceCompileInfo {} compileInfo;

    std::vector<gert::TilingContextPara::TensorDescription> inputList;
    inputList.push_back({make_shape(param.x_shape), param.x_dtype, ge::FORMAT_ND});
    inputList.push_back({make_shape(param.weight_shape), param.weight_dtype, ge::FORMAT_ND});
    inputList.push_back({make_shape(param.bias_shape), param.bias_dtype, ge::FORMAT_ND});
    inputList.push_back({make_shape(param.group_list_shape), param.group_list_dtype, ge::FORMAT_ND});

    std::vector<gert::TilingContextPara::TensorDescription> outputList;
    outputList.push_back({make_shape(param.y_shape), param.y_dtype, ge::FORMAT_ND});

    gert::TilingContextPara tilingContextPara("GroupedMatMulAllReduce", inputList, outputList,
        {
            {"splitItem", build_from<int64_t>(param.splitItem)},
            {"group", build_from<std::string>(param.group)},
            {"reduceOp", build_from<std::string>(param.reduceOp)},
            {"commTurn", build_from<int64_t>(param.commTurn)},
        },
        &compileInfo, param.soc_version, param.compile_info, param.tilingDataSize);

    Mc2Hcom::MockValues hcomTopologyMockValues{{"rankNum", 8}};
    Mc2ExecuteTestCase(tilingContextPara, hcomTopologyMockValues, ge::GRAPH_SUCCESS, param.expectTilingKey);
}

GroupedMatMulAllReduceTilingTestParam cases_params[] = {
    {"grouped_mat_mul_all_reduce_test_tiling_float16_1", "", "Ascend910B", 20, 196608, 40960, {8192, 1536}, ge::DT_FLOAT16, {1536, 12288}, ge::DT_FLOAT16, {}, ge::DT_FLOAT, {}, ge::DT_FLOAT, {}, ge::DT_FLOAT, 0, "", "", 0, 0UL},
    {"grouped_mat_mul_all_reduce_test_mcut_float16_910B_1", "", "Ascend910B", 20, 196608, 40960, {12290, 15360}, ge::DT_FLOAT16, {15360, 12288}, ge::DT_FLOAT16, {}, ge::DT_FLOAT, {}, ge::DT_FLOAT, {}, ge::DT_FLOAT, 0, "", "", 0, 0UL},
    {"grouped_mat_mul_all_reduce_test_mcut_float16_910B_2", "", "Ascend910B", 20, 196608, 40960, {20, 2}, ge::DT_FLOAT16, {2, 2}, ge::DT_FLOAT16, {}, ge::DT_FLOAT, {}, ge::DT_FLOAT, {}, ge::DT_FLOAT, 0, "", "", 0, 0UL},
    {"grouped_mat_mul_all_reduce_test_mcut_float16_910B_win2win", "", "Ascend910B", 20, 196608, 40960, {12290, 15360}, ge::DT_FLOAT16, {15360, 12288}, ge::DT_FLOAT16, {}, ge::DT_FLOAT, {}, ge::DT_FLOAT, {}, ge::DT_FLOAT, 0, "", "", 0, 0UL},
    {"grouped_mat_mul_all_reduce_test_tiling_float16_2", "", "Ascend910B", 20, 196608, 40960, {8192, 1536}, ge::DT_FLOAT16, {1536, 12288}, ge::DT_FLOAT16, {}, ge::DT_FLOAT, {}, ge::DT_FLOAT, {}, ge::DT_FLOAT, 0, "", "", 0, 0UL},
    {"grouped_mat_mul_all_reduce_test_tiling_float16_3", "", "Ascend910B", 20, 196608, 40960, {128, 1536}, ge::DT_FLOAT16, {1536, 8192}, ge::DT_FLOAT16, {}, ge::DT_FLOAT, {}, ge::DT_FLOAT, {}, ge::DT_FLOAT, 0, "", "", 0, 0UL},
    {"grouped_mat_mul_all_reduce_test_tiling_float16_4", "", "Ascend910B", 20, 196608, 40960, {1024, 1536}, ge::DT_FLOAT16, {1536, 8192}, ge::DT_FLOAT16, {}, ge::DT_FLOAT, {}, ge::DT_FLOAT, {}, ge::DT_FLOAT, 0, "", "", 0, 0UL},
    {"grouped_mat_mul_all_reduce_test_tiling_float16_5", "", "Ascend910B", 20, 196608, 40960, {256, 1536}, ge::DT_FLOAT16, {1536, 8192}, ge::DT_FLOAT16, {}, ge::DT_FLOAT, {}, ge::DT_FLOAT, {}, ge::DT_FLOAT, 0, "", "", 0, 0UL},
    {"grouped_mat_mul_all_reduce_test_tiling_float16_support_3_dim", "", "Ascend910B", 20, 196608, 40960, {1, 8192, 1536}, ge::DT_FLOAT16, {1536, 12288}, ge::DT_FLOAT16, {}, ge::DT_FLOAT, {}, ge::DT_FLOAT, {}, ge::DT_FLOAT, 0, "", "", 0, 0UL},
    {"grouped_mat_mul_all_reduce_test_tiling_bfloat16", "", "Ascend910B", 20, 196608, 40960, {8192, 1536}, ge::DT_BF16, {1536, 12288}, ge::DT_BF16, {}, ge::DT_FLOAT, {}, ge::DT_FLOAT, {}, ge::DT_FLOAT, 0, "", "", 0, 0UL},
};

TEST_P(GroupedMatMulAllReduceTilingParam, general_case)
{
    const auto &param = GetParam();
    TestOneParamCase(param);
}

INSTANTIATE_TEST_SUITE_P(
    general_cases_params,
    GroupedMatMulAllReduceTilingParam,
    ::testing::ValuesIn(cases_params),
    [](const ::testing::TestParamInfo<GroupedMatMulAllReduceTilingTestParam> &info) {
        std::string name = info.param.case_name;
        for (char &c : name) { if (!std::isalnum(static_cast<unsigned char>(c)) && c != '_') c = '_'; }
        return name;
    });
} // anonymous namespace
} // namespace GroupedMatMulAllReduceUT