/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd. All rights reserved.
 */
#include <iostream>
#include <vector>
#include <string>
#include <gtest/gtest.h>
#include "mc2_tiling_case_executor.h"

namespace AlltoAllvGroupedMatMulUT {
namespace {
template <typename T>
auto build_from(const T &value) { return Ops::Transformer::AnyValue::CreateFrom<T>(value); }

gert::StorageShape make_shape(const std::initializer_list<int64_t> &input_shape)
{
    if (input_shape.size() == 0) { return gert::StorageShape{}; }
    return gert::StorageShape{input_shape, input_shape};
}

std::unique_ptr<gert::TilingContextPara::TensorDescription> CreateTensorShape(
    std::initializer_list<int64_t> shape,
    ge::DataType dtype,
    ge::Format format
) {
    return std::unique_ptr<gert::TilingContextPara::TensorDescription>(
        new gert::TilingContextPara::TensorDescription(
            make_shape(shape),
            dtype,
            format
        )
    );
}

struct AlltoAllvGroupedMatMulTilingTestParam {
    std::string case_name;
    uint64_t BSK;
    uint64_t BS;
    uint64_t K;
    uint64_t H1;
    uint64_t H2;
    uint64_t A;
    uint64_t N1;
    uint64_t N2;
    uint64_t ep_world_size;
    uint64_t e;
    uint64_t commOut;
    uint64_t aivCoreNum;
    uint64_t aicCoreNum;
    uint64_t coreNum;
    uint64_t totalUbSize;
    uint64_t gmm_weight_dim1;
    uint64_t gmm_y_dim1;
    uint64_t mm_weight_dim0;
    bool trans_gmm_weight;
    bool trans_mm_weight;
    bool permute_out_flag;
    bool is_Need_MM;
    std::string group;
    std::vector<int64_t> send_counts;
    std::vector<int64_t> recv_counts;
    uint64_t expect_tiling_key;
};

class AlltoAllvGroupedMatMulTilingParam : public ::testing::TestWithParam<AlltoAllvGroupedMatMulTilingTestParam> {
protected:
    static void SetUpTestCase() { std::cout << "SetUp" << std::endl; }
    static void TearDownTestCase() { std::cout << "TearDown" << std::endl; }
};

std::vector<gert::TilingContextPara::TensorDescription> CreateInputTensors(
    const AlltoAllvGroupedMatMulTilingTestParam& param,
    const std::unique_ptr<gert::TilingContextPara::TensorDescription>& mm_x_shape,
    const std::unique_ptr<gert::TilingContextPara::TensorDescription>& mm_weight_shape
) {
    return {
        {make_shape({static_cast<int64_t>(param.BSK), static_cast<int64_t>(param.H1)}), ge::DT_FLOAT16, ge::FORMAT_ND},
        {make_shape({static_cast<int64_t>(param.e), static_cast<int64_t>(param.gmm_weight_dim1), static_cast<int64_t>(param.N1)}), ge::DT_FLOAT16, ge::FORMAT_ND},
        {{}, ge::DT_FLOAT16, ge::FORMAT_ND}, // placeholder
        {{}, ge::DT_FLOAT16, ge::FORMAT_ND}, // placeholder
        *mm_x_shape,
        *mm_weight_shape, 
    };
}

std::vector<gert::TilingContextPara::TensorDescription> CreateOutputTensors(
    const AlltoAllvGroupedMatMulTilingTestParam& param,
    const std::unique_ptr<gert::TilingContextPara::TensorDescription>& mm_y_shape
) {
    return {
        {make_shape({static_cast<int64_t>(param.A), static_cast<int64_t>(param.gmm_y_dim1)}), ge::DT_FLOAT16, ge::FORMAT_ND},
        *mm_y_shape,
        {make_shape({static_cast<int64_t>(param.A), static_cast<int64_t>(param.H1)}), ge::DT_FLOAT16, ge::FORMAT_ND},
    };
}

void TestOneParamCase(const AlltoAllvGroupedMatMulTilingTestParam &param)
{
    struct AlltoAllvGroupedMatMulCompileInfo {} compileInfo;

    auto mm_x_shape = CreateTensorShape({static_cast<int64_t>(param.BS), static_cast<int64_t>(param.H2)}, ge::DT_FLOAT16, ge::FORMAT_ND);
    auto mm_weight_shape = CreateTensorShape({static_cast<int64_t>(param.mm_weight_dim0), static_cast<int64_t>(param.N2)}, ge::DT_FLOAT16, ge::FORMAT_ND);
    auto mm_y_shape = CreateTensorShape({static_cast<int64_t>(param.BS), static_cast<int64_t>(param.N2)}, ge::DT_FLOAT16, ge::FORMAT_ND);

    gert::TilingContextPara tilingContextPara("AlltoAllvGroupedMatMul", 
        CreateInputTensors(param, mm_x_shape, mm_weight_shape),
        CreateOutputTensors(param, mm_y_shape),
        {
            {"group", build_from<std::string>(param.group)},
            {"ep_world_size", build_from<int64_t>(param.ep_world_size)},
            {"send_counts", build_from<std::vector<int64_t>>(param.send_counts)},
            {"recv_counts", build_from<std::vector<int64_t>>(param.recv_counts)},
            {"trans_gmm_weight", build_from<bool>(param.trans_gmm_weight)},
            {"trans_mm_weight", build_from<bool>(param.trans_mm_weight)},
            {"permute_out_flag", build_from<bool>(param.permute_out_flag)},
        },
        &compileInfo, "Ascend910_93", param.coreNum, param.totalUbSize, 8192);

    Mc2Hcom::MockValues hcomTopologyMockValues{{"rankNum", 8}};
    Mc2ExecuteTestCase(tilingContextPara, hcomTopologyMockValues, ge::GRAPH_SUCCESS, param.expect_tiling_key);
}

TEST_P(AlltoAllvGroupedMatMulTilingParam, general_case)
{
    const auto &param = GetParam();
    TestOneParamCase(param);
}

// Suppress error about uninstantiated parameterized test
GTEST_ALLOW_UNINSTANTIATED_PARAMETERIZED_TEST(AlltoAllvGroupedMatMulTilingParam);

/*
INSTANTIATE_TEST_SUITE_P(
    general_cases_params,
    AlltoAllvGroupedMatMulTilingParam,
    ::testing::ValuesIn(cases_params),
    [](const ::testing::TestParamInfo<AlltoAllvGroupedMatMulTilingTestParam> &info) {
        std::string name = info.param.case_name;
        for (char &c : name) { if (!std::isalnum(static_cast<unsigned char>(c)) && c != '_') c = '_'; }
        return name;
    });
*/
} // anonymous namespace
} // namespace AlltoAllvGroupedMatMulUT

