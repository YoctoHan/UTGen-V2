/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd. All rights reserved.
 */
#include <iostream>
#include <vector>
#include <string>
#include <gtest/gtest.h>
#include "mc2_tiling_case_executor.h"

namespace MoeDistributeCombineAddRmsNormUT {
namespace {
template <typename T>
auto build_from(const T &value) { return Ops::Transformer::AnyValue::CreateFrom<T>(value); }

gert::StorageShape make_shape(const std::initializer_list<int64_t> &input_shape)
{
    if (input_shape.size() == 0) { return gert::StorageShape{}; }
    return gert::StorageShape{input_shape, input_shape};
}

struct MoeDistributeCombineAddRmsNormTilingTestParam {
    std::string case_name;
    std::string compile_info;
    std::string soc_version;
    uint64_t coreNum;
    uint64_t ubSize;
    uint64_t tilingDataSize;

    std::initializer_list<int64_t> expand_x_shape;
    ge::DataType expand_x_dtype;
    std::initializer_list<int64_t> expert_ids_shape;
    ge::DataType expert_ids_dtype;
    std::initializer_list<int64_t> assist_info_for_combine_shape;
    ge::DataType assist_info_for_combine_dtype;
    std::initializer_list<int64_t> ep_send_counts_shape;
    ge::DataType ep_send_counts_dtype;
    std::initializer_list<int64_t> expert_scales_shape;
    ge::DataType expert_scales_dtype;
    std::initializer_list<int64_t> residual_x_shape;
    ge::DataType residual_x_dtype;
    std::initializer_list<int64_t> gamma_shape;
    ge::DataType gamma_dtype;
    std::initializer_list<int64_t> tp_send_counts_shape;
    ge::DataType tp_send_counts_dtype;
    std::initializer_list<int64_t> x_active_mask_shape;
    ge::DataType x_active_mask_dtype;
    std::initializer_list<int64_t> activation_scale_shape;
    ge::DataType activation_scale_dtype;
    std::initializer_list<int64_t> weight_scale_shape;
    ge::DataType weight_scale_dtype;
    std::initializer_list<int64_t> group_list_shape;
    ge::DataType group_list_dtype;
    std::initializer_list<int64_t> expand_scales_shape;
    ge::DataType expand_scales_dtype;
    std::initializer_list<int64_t> shared_expert_x_shape;
    ge::DataType shared_expert_x_dtype;
    std::initializer_list<int64_t> elastic_info_shape;
    ge::DataType elastic_info_dtype;
    std::initializer_list<int64_t> ori_x_shape;
    ge::DataType ori_x_dtype;
    std::initializer_list<int64_t> const_expert_alpha_1_shape;
    ge::DataType const_expert_alpha_1_dtype;
    std::initializer_list<int64_t> const_expert_alpha_2_shape;
    ge::DataType const_expert_alpha_2_dtype;
    std::initializer_list<int64_t> const_expert_v_shape;
    ge::DataType const_expert_v_dtype;
    std::initializer_list<int64_t> y_shape;
    ge::DataType y_dtype;
    std::initializer_list<int64_t> rstdOut_shape;
    ge::DataType rstdOut_dtype;
    std::initializer_list<int64_t> x_shape;
    ge::DataType x_dtype;

    std::string group_ep;
    int64_t ep_world_size;
    int64_t ep_rank_id;
    int64_t moe_expert_num;
    std::string group_tp;
    int64_t tp_world_size;
    int64_t tp_rank_id;
    int64_t expert_shard_type;
    int64_t shared_expert_num;
    int64_t shared_expert_rank_num;
    int64_t global_bs;
    int64_t out_dtype;
    int64_t comm_quant_mode;
    int64_t group_list_type;
    std::string comm_alg;
    float norm_eps;
    int64_t zero_expert_num;
    int64_t copy_expert_num;
    int64_t const_expert_num;
    uint64_t expectTilingKey;
};

class MoeDistributeCombineAddRmsNormTilingParam : public ::testing::TestWithParam<MoeDistributeCombineAddRmsNormTilingTestParam> {
protected:
    static void SetUpTestCase() { std::cout << "SetUp" << std::endl; }
    static void TearDownTestCase() { std::cout << "TearDown" << std::endl; }
};

void TestOneParamCase(const MoeDistributeCombineAddRmsNormTilingTestParam &param)
{
    struct MoeDistributeCombineAddRmsNormCompileInfo {} compileInfo;

    std::vector<gert::TilingContextPara::TensorDescription> inputList;
    inputList.push_back({make_shape(param.expand_x_shape), param.expand_x_dtype, ge::FORMAT_ND});
    inputList.push_back({make_shape(param.expert_ids_shape), param.expert_ids_dtype, ge::FORMAT_ND});
    inputList.push_back({make_shape(param.assist_info_for_combine_shape), param.assist_info_for_combine_dtype, ge::FORMAT_ND});
    inputList.push_back({make_shape(param.ep_send_counts_shape), param.ep_send_counts_dtype, ge::FORMAT_ND});
    inputList.push_back({make_shape(param.expert_scales_shape), param.expert_scales_dtype, ge::FORMAT_ND});
    inputList.push_back({make_shape(param.residual_x_shape), param.residual_x_dtype, ge::FORMAT_ND});
    inputList.push_back({make_shape(param.gamma_shape), param.gamma_dtype, ge::FORMAT_ND});
    inputList.push_back({make_shape(param.tp_send_counts_shape), param.tp_send_counts_dtype, ge::FORMAT_ND});
    inputList.push_back({make_shape(param.x_active_mask_shape), param.x_active_mask_dtype, ge::FORMAT_ND});
    inputList.push_back({make_shape(param.activation_scale_shape), param.activation_scale_dtype, ge::FORMAT_ND});
    inputList.push_back({make_shape(param.weight_scale_shape), param.weight_scale_dtype, ge::FORMAT_ND});
    inputList.push_back({make_shape(param.group_list_shape), param.group_list_dtype, ge::FORMAT_ND});
    inputList.push_back({make_shape(param.expand_scales_shape), param.expand_scales_dtype, ge::FORMAT_ND});
    inputList.push_back({make_shape(param.shared_expert_x_shape), param.shared_expert_x_dtype, ge::FORMAT_ND});
    inputList.push_back({make_shape(param.elastic_info_shape), param.elastic_info_dtype, ge::FORMAT_ND});
    inputList.push_back({make_shape(param.ori_x_shape), param.ori_x_dtype, ge::FORMAT_ND});
    inputList.push_back({make_shape(param.const_expert_alpha_1_shape), param.const_expert_alpha_1_dtype, ge::FORMAT_ND});
    inputList.push_back({make_shape(param.const_expert_alpha_2_shape), param.const_expert_alpha_2_dtype, ge::FORMAT_ND});
    inputList.push_back({make_shape(param.const_expert_v_shape), param.const_expert_v_dtype, ge::FORMAT_ND});

    std::vector<gert::TilingContextPara::TensorDescription> outputList;
    outputList.push_back({make_shape(param.y_shape), param.y_dtype, ge::FORMAT_ND});
    outputList.push_back({make_shape(param.rstdOut_shape), param.rstdOut_dtype, ge::FORMAT_ND});
    outputList.push_back({make_shape(param.x_shape), param.x_dtype, ge::FORMAT_ND});

    gert::TilingContextPara tilingContextPara("MoeDistributeCombineAddRmsNorm", inputList, outputList,
        {
            {"group_ep", build_from<std::string>(param.group_ep)},
            {"ep_world_size", build_from<int64_t>(param.ep_world_size)},
            {"ep_rank_id", build_from<int64_t>(param.ep_rank_id)},
            {"moe_expert_num", build_from<int64_t>(param.moe_expert_num)},
            {"group_tp", build_from<std::string>(param.group_tp)},
            {"tp_world_size", build_from<int64_t>(param.tp_world_size)},
            {"tp_rank_id", build_from<int64_t>(param.tp_rank_id)},
            {"expert_shard_type", build_from<int64_t>(param.expert_shard_type)},
            {"shared_expert_num", build_from<int64_t>(param.shared_expert_num)},
            {"shared_expert_rank_num", build_from<int64_t>(param.shared_expert_rank_num)},
            {"global_bs", build_from<int64_t>(param.global_bs)},
            {"out_dtype", build_from<int64_t>(param.out_dtype)},
            {"comm_quant_mode", build_from<int64_t>(param.comm_quant_mode)},
            {"group_list_type", build_from<int64_t>(param.group_list_type)},
            {"comm_alg", build_from<std::string>(param.comm_alg)},
            {"norm_eps", build_from<float>(param.norm_eps)},
            {"zero_expert_num", build_from<int64_t>(param.zero_expert_num)},
            {"copy_expert_num", build_from<int64_t>(param.copy_expert_num)},
            {"const_expert_num", build_from<int64_t>(param.const_expert_num)},
        },
        &compileInfo, param.soc_version, param.coreNum, param.ubSize, param.tilingDataSize);

    Mc2Hcom::MockValues hcomTopologyMockValues{{"rankNum", 8}};
    Mc2ExecuteTestCase(tilingContextPara, hcomTopologyMockValues, ge::GRAPH_SUCCESS, param.expectTilingKey);
}

// 移除默认参数，等待输入文件
MoeDistributeCombineAddRmsNormTilingTestParam cases_params[] = {
};

TEST_P(MoeDistributeCombineAddRmsNormTilingParam, general_case)
{
    const auto &param = GetParam();
    TestOneParamCase(param);
}

// Suppress error about uninstantiated parameterized test
GTEST_ALLOW_UNINSTANTIATED_PARAMETERIZED_TEST(MoeDistributeCombineAddRmsNormTilingParam);

/*
INSTANTIATE_TEST_SUITE_P(
    general_cases_params,
    MoeDistributeCombineAddRmsNormTilingParam,
    ::testing::ValuesIn(cases_params),
    [](const ::testing::TestParamInfo<MoeDistributeCombineAddRmsNormTilingTestParam> &info) {
        std::string name = info.param.case_name;
        for (char &c : name) { if (!std::isalnum(static_cast<unsigned char>(c)) && c != '_') c = '_'; }
        return name;
    });
*/
} // anonymous namespace
} // namespace MoeDistributeCombineAddRmsNormUT
