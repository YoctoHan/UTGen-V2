/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd. All rights reserved.
 */
#include <iostream>
#include <vector>
#include <string>
#include <gtest/gtest.h>
#include "mc2_tiling_case_executor.h"

namespace MoeDistributeDispatchV2UT {
namespace {
template <typename T>
auto build_from(const T &value) { return Ops::Transformer::AnyValue::CreateFrom<T>(value); }

gert::StorageShape make_shape(const std::initializer_list<int64_t> &input_shape)
{
    if (input_shape.size() == 0) { return gert::StorageShape{}; }
    return gert::StorageShape{input_shape, input_shape};
}

struct MoeDistributeDispatchV2TilingTestParam {
    std::string case_name;
    std::string compile_info;
    std::string soc_version;
    uint64_t coreNum;
    uint64_t ubSize;
    uint64_t tilingDataSize;

    std::initializer_list<int64_t> x_shape;
    ge::DataType x_dtype;
    std::initializer_list<int64_t> expert_ids_shape;
    ge::DataType expert_ids_dtype;
    std::initializer_list<int64_t> scales_shape;
    ge::DataType scales_dtype;
    std::initializer_list<int64_t> x_active_mask_shape;
    ge::DataType x_active_mask_dtype;
    std::initializer_list<int64_t> expert_scales_shape;
    ge::DataType expert_scales_dtype;
    std::initializer_list<int64_t> elastic_info_shape;
    ge::DataType elastic_info_dtype;
    std::initializer_list<int64_t> performance_info_shape;
    ge::DataType performance_info_dtype;
    std::initializer_list<int64_t> expand_x_shape;
    ge::DataType expand_x_dtype;
    std::initializer_list<int64_t> dynamic_scales_shape;
    ge::DataType dynamic_scales_dtype;
    std::initializer_list<int64_t> assist_info_for_combine_shape;
    ge::DataType assist_info_for_combine_dtype;
    std::initializer_list<int64_t> expert_token_nums_shape;
    ge::DataType expert_token_nums_dtype;
    std::initializer_list<int64_t> ep_recv_count_shape;
    ge::DataType ep_recv_count_dtype;
    std::initializer_list<int64_t> tp_recv_count_shape;
    ge::DataType tp_recv_count_dtype;
    std::initializer_list<int64_t> expand_scales_shape;
    ge::DataType expand_scales_dtype;

    std::string ep_group;
    int64_t ep_world_size;
    int64_t ep_rank_id;
    int64_t moe_expert_num;
    std::string tp_group;
    int64_t tp_world_size;
    int64_t tp_rank_id;
    int64_t expert_shard_type;
    int64_t shared_expert_num;
    int64_t shared_expert_rank_num;
    int64_t quant_mode;
    int64_t global_bs;
    int64_t expert_token_nums_type;
    std::string comm_alg;
    int64_t zero_expert_num;
    int64_t copy_expert_num;
    int64_t const_expert_num;
    bool has_expect_tiling_key;
    uint64_t expect_tiling_key;
};

class MoeDistributeDispatchV2TilingParam : public ::testing::TestWithParam<MoeDistributeDispatchV2TilingTestParam> {
protected:
    static void SetUpTestCase() { std::cout << "SetUp" << std::endl; }
    static void TearDownTestCase() { std::cout << "TearDown" << std::endl; }
};

void TestOneParamCase(const MoeDistributeDispatchV2TilingTestParam &param)
{
    struct MoeDistributeDispatchV2CompileInfo {} compileInfo;

    std::vector<gert::TilingContextPara::TensorDescription> inputList;
    inputList.push_back({make_shape(param.x_shape), param.x_dtype, ge::FORMAT_ND});
    inputList.push_back({make_shape(param.expert_ids_shape), param.expert_ids_dtype, ge::FORMAT_ND});
    inputList.push_back({make_shape(param.scales_shape), param.scales_dtype, ge::FORMAT_ND});
    inputList.push_back({make_shape(param.x_active_mask_shape), param.x_active_mask_dtype, ge::FORMAT_ND});
    inputList.push_back({make_shape(param.expert_scales_shape), param.expert_scales_dtype, ge::FORMAT_ND});
    inputList.push_back({make_shape(param.elastic_info_shape), param.elastic_info_dtype, ge::FORMAT_ND});
    inputList.push_back({make_shape(param.performance_info_shape), param.performance_info_dtype, ge::FORMAT_ND});

    std::vector<gert::TilingContextPara::TensorDescription> outputList;
    outputList.push_back({make_shape(param.expand_x_shape), param.expand_x_dtype, ge::FORMAT_ND});
    outputList.push_back({make_shape(param.dynamic_scales_shape), param.dynamic_scales_dtype, ge::FORMAT_ND});
    outputList.push_back({make_shape(param.assist_info_for_combine_shape), param.assist_info_for_combine_dtype, ge::FORMAT_ND});
    outputList.push_back({make_shape(param.expert_token_nums_shape), param.expert_token_nums_dtype, ge::FORMAT_ND});
    outputList.push_back({make_shape(param.ep_recv_count_shape), param.ep_recv_count_dtype, ge::FORMAT_ND});
    outputList.push_back({make_shape(param.tp_recv_count_shape), param.tp_recv_count_dtype, ge::FORMAT_ND});
    outputList.push_back({make_shape(param.expand_scales_shape), param.expand_scales_dtype, ge::FORMAT_ND});

    gert::TilingContextPara tilingContextPara("MoeDistributeDispatchV2", inputList, outputList,
        {
            {"group_ep", build_from<std::string>(param.ep_group)},
            {"ep_world_size", build_from<int64_t>(param.ep_world_size)},
            {"ep_rank_id", build_from<int64_t>(param.ep_rank_id)},
            {"moe_expert_num", build_from<int64_t>(param.moe_expert_num)},
            {"group_tp", build_from<std::string>(param.tp_group)},
            {"tp_world_size", build_from<int64_t>(param.tp_world_size)},
            {"tp_rank_id", build_from<int64_t>(param.tp_rank_id)},
            {"expert_shard_type", build_from<int64_t>(param.expert_shard_type)},
            {"shared_expert_num", build_from<int64_t>(param.shared_expert_num)},
            {"shared_expert_rank_num", build_from<int64_t>(param.shared_expert_rank_num)},
            {"quant_mode", build_from<int64_t>(param.quant_mode)},
            {"global_bs", build_from<int64_t>(param.global_bs)},
            {"expert_token_nums_type", build_from<int64_t>(param.expert_token_nums_type)},
            {"comm_alg", build_from<std::string>(param.comm_alg)},
            {"zero_expert_num", build_from<int64_t>(param.zero_expert_num)},
            {"copy_expert_num", build_from<int64_t>(param.copy_expert_num)},
            {"const_expert_num", build_from<int64_t>(param.const_expert_num)},
        },
        &compileInfo, param.soc_version, param.coreNum, param.ubSize);

    Mc2Hcom::MockValues hcomTopologyMockValues{{"rankNum", 8}};
    if (param.has_expect_tiling_key) {
        Mc2ExecuteTestCase(tilingContextPara, hcomTopologyMockValues, ge::GRAPH_SUCCESS, param.expect_tiling_key);
    } else {
        Mc2ExecuteTestCase(tilingContextPara, hcomTopologyMockValues);
    }
}

MoeDistributeDispatchV2TilingTestParam cases_params[] = {
    {"moe_distribute_dispatch_test_tiling_0", "", "Ascend910_93", 20, 196608, 0, {}, ge::DT_FLOAT, {}, ge::DT_FLOAT, {}, ge::DT_FLOAT, {}, ge::DT_FLOAT, {}, ge::DT_FLOAT, {}, ge::DT_FLOAT, {}, ge::DT_FLOAT, {}, ge::DT_FLOAT, {}, ge::DT_FLOAT, {}, ge::DT_FLOAT, {}, ge::DT_FLOAT, {}, ge::DT_FLOAT, {}, ge::DT_FLOAT, {}, ge::DT_FLOAT, "ep_group", 288, 0, 256, "tp_group", 2, 0, 0, 1, 32, 0, 0, 1, "", 0, 0, 0, false, 0},
    {"moe_distribute_dispatch_test_tiling_1", "", "Ascend910_93", 20, 196608, 0, {}, ge::DT_FLOAT, {}, ge::DT_FLOAT, {}, ge::DT_FLOAT, {}, ge::DT_FLOAT, {}, ge::DT_FLOAT, {}, ge::DT_FLOAT, {}, ge::DT_FLOAT, {}, ge::DT_FLOAT, {}, ge::DT_FLOAT, {}, ge::DT_FLOAT, {}, ge::DT_FLOAT, {}, ge::DT_FLOAT, {}, ge::DT_FLOAT, {}, ge::DT_FLOAT, "ep_group", 288, 0, 256, "tp_group", 2, 0, 0, 1, 32, 0, 0, 1, "", 0, 0, 0, false, 0},
    {"moe_distribute_dispatch_test_tiling_2", "", "Ascend910_93", 20, 196608, 0, {}, ge::DT_FLOAT, {}, ge::DT_FLOAT, {}, ge::DT_FLOAT, {}, ge::DT_FLOAT, {}, ge::DT_FLOAT, {}, ge::DT_FLOAT, {}, ge::DT_FLOAT, {}, ge::DT_FLOAT, {}, ge::DT_FLOAT, {}, ge::DT_FLOAT, {}, ge::DT_FLOAT, {}, ge::DT_FLOAT, {}, ge::DT_FLOAT, {}, ge::DT_FLOAT, "ep_group", 288, 0, 256, "tp_group", 2, 1024, 0, 1, 32, 0, 0, 1, "", 0, 0, 0, false, 0},
    {"moe_distribute_dispatch_test_tiling_3", "", "Ascend910_93", 20, 196608, 0, {}, ge::DT_FLOAT, {}, ge::DT_FLOAT, {}, ge::DT_FLOAT, {}, ge::DT_FLOAT, {}, ge::DT_FLOAT, {}, ge::DT_FLOAT, {}, ge::DT_FLOAT, {}, ge::DT_FLOAT, {}, ge::DT_FLOAT, {}, ge::DT_FLOAT, {}, ge::DT_FLOAT, {}, ge::DT_FLOAT, {}, ge::DT_FLOAT, {}, ge::DT_FLOAT, "ep_group", 288, 0, 256, "tp_group", 2, 0, 0, 1, 31, 0, 0, 0, "", 0, 0, 0, false, 0},
    {"moe_distribute_dispatch_test_tiling_4", "", "Ascend910_93", 20, 196608, 0, {}, ge::DT_FLOAT, {}, ge::DT_FLOAT, {}, ge::DT_FLOAT, {}, ge::DT_FLOAT, {}, ge::DT_FLOAT, {}, ge::DT_FLOAT, {}, ge::DT_FLOAT, {}, ge::DT_FLOAT, {}, ge::DT_FLOAT, {}, ge::DT_FLOAT, {}, ge::DT_FLOAT, {}, ge::DT_FLOAT, {}, ge::DT_FLOAT, {}, ge::DT_FLOAT, "ep_group", 288, 0, 257, "tp_group", 2, 0, 0, 1, 31, 1, 0, 0, "", 0, 0, 0, false, 0},
    {"moe_distribute_dispatch_test_tiling_5", "", "Ascend910_93", 20, 196608, 0, {}, ge::DT_FLOAT, {}, ge::DT_FLOAT, {}, ge::DT_FLOAT, {}, ge::DT_FLOAT, {}, ge::DT_FLOAT, {}, ge::DT_FLOAT, {}, ge::DT_FLOAT, {}, ge::DT_FLOAT, {}, ge::DT_FLOAT, {}, ge::DT_FLOAT, {}, ge::DT_FLOAT, {}, ge::DT_FLOAT, {}, ge::DT_FLOAT, {}, ge::DT_FLOAT, "ep_group", 288, 0, 256, "tp_group", 2, 0, 0, 1, 32, 10, 0, 0, "", 0, 0, 0, false, 0},
    {"moe_distribute_dispatch_test_tiling_7", "", "Ascend910_93", 20, 196608, 0, {}, ge::DT_FLOAT, {}, ge::DT_FLOAT, {}, ge::DT_FLOAT, {}, ge::DT_FLOAT, {}, ge::DT_FLOAT, {}, ge::DT_FLOAT, {}, ge::DT_FLOAT, {}, ge::DT_FLOAT, {}, ge::DT_FLOAT, {}, ge::DT_FLOAT, {}, ge::DT_FLOAT, {}, ge::DT_FLOAT, {}, ge::DT_FLOAT, {}, ge::DT_FLOAT, "ep_group", 288, 0, 256, "tp_group", 2, -1, 0, 1, 32, 2, 0, 0, "", 0, 0, 0, false, 0},
    {"moe_distribute_dispatch_test_tiling_8", "", "Ascend910_93", 20, 196608, 0, {}, ge::DT_FLOAT, {}, ge::DT_FLOAT, {}, ge::DT_FLOAT, {}, ge::DT_FLOAT, {}, ge::DT_FLOAT, {}, ge::DT_FLOAT, {}, ge::DT_FLOAT, {}, ge::DT_FLOAT, {}, ge::DT_FLOAT, {}, ge::DT_FLOAT, {}, ge::DT_FLOAT, {}, ge::DT_FLOAT, {}, ge::DT_FLOAT, {}, ge::DT_FLOAT, "ep_group", 288, 1, 256, "", 2, 1024, 1, 1, 32, 2, 1, 1, "", 0, 0, 0, false, 0},
    {"moe_distribute_dispatch_test_tiling_9", "", "Ascend910_93", 20, 196608, 0, {}, ge::DT_FLOAT, {}, ge::DT_FLOAT, {}, ge::DT_FLOAT, {}, ge::DT_FLOAT, {}, ge::DT_FLOAT, {}, ge::DT_FLOAT, {}, ge::DT_FLOAT, {}, ge::DT_FLOAT, {}, ge::DT_FLOAT, {}, ge::DT_FLOAT, {}, ge::DT_FLOAT, {}, ge::DT_FLOAT, {}, ge::DT_FLOAT, {}, ge::DT_FLOAT, "ep_group", 288, 0, 256, "", 2, -1, 0, 1, 32, 2, 0, 0, "", 0, 0, 0, false, 0},
    {"moe_distribute_dispatch_test_tiling_10", "", "Ascend910_93", 20, 196608, 0, {}, ge::DT_FLOAT, {}, ge::DT_FLOAT, {}, ge::DT_FLOAT, {}, ge::DT_FLOAT, {}, ge::DT_FLOAT, {}, ge::DT_FLOAT, {}, ge::DT_FLOAT, {}, ge::DT_FLOAT, {}, ge::DT_FLOAT, {}, ge::DT_FLOAT, {}, ge::DT_FLOAT, {}, ge::DT_FLOAT, {}, ge::DT_FLOAT, {}, ge::DT_FLOAT, "ep_group", 288, 256, 256, "tp_group", 2, 0, 0, 1, 32, 0, 0, 1, "", 0, 0, 0, false, 0},
    {"moe_distribute_dispatch_test_tiling_ep_world_size_384", "", "Ascend910_93", 20, 196608, 0, {}, ge::DT_FLOAT, {}, ge::DT_FLOAT, {}, ge::DT_FLOAT, {}, ge::DT_FLOAT, {}, ge::DT_FLOAT, {}, ge::DT_FLOAT, {}, ge::DT_FLOAT, {}, ge::DT_FLOAT, {}, ge::DT_FLOAT, {}, ge::DT_FLOAT, {}, ge::DT_FLOAT, {}, ge::DT_FLOAT, {}, ge::DT_FLOAT, {}, ge::DT_FLOAT, "ep_group", 384, 0, 256, "tp_group", 2, 0, 0, 1, 32, 0, 0, 1, "", 0, 0, 0, false, 0},
    {"moe_distribute_dispatch_test_tiling_ep_world_size_72", "", "Ascend910_93", 20, 196608, 0, {}, ge::DT_FLOAT, {}, ge::DT_FLOAT, {}, ge::DT_FLOAT, {}, ge::DT_FLOAT, {}, ge::DT_FLOAT, {}, ge::DT_FLOAT, {}, ge::DT_FLOAT, {}, ge::DT_FLOAT, {}, ge::DT_FLOAT, {}, ge::DT_FLOAT, {}, ge::DT_FLOAT, {}, ge::DT_FLOAT, {}, ge::DT_FLOAT, {}, ge::DT_FLOAT, "ep_group", 72, 0, 216, "tp_group", 2, 0, 0, 1, 18, 0, 0, 1, "", 0, 0, 0, false, 0},
    {"moe_distribute_dispatch_test_tiling_x_active_mask_2dims", "", "Ascend910_93", 20, 196608, 0, {}, ge::DT_FLOAT, {}, ge::DT_FLOAT, {}, ge::DT_FLOAT, {}, ge::DT_FLOAT, {}, ge::DT_FLOAT, {}, ge::DT_FLOAT, {}, ge::DT_FLOAT, {}, ge::DT_FLOAT, {}, ge::DT_FLOAT, {}, ge::DT_FLOAT, {}, ge::DT_FLOAT, {}, ge::DT_FLOAT, {}, ge::DT_FLOAT, {}, ge::DT_FLOAT, "ep_group", 72, 0, 216, "tp_group", 2, 0, 0, 1, 18, 0, 0, 1, "", 0, 0, 0, false, 0},
    {"moe_distribute_dispatch_test_tiling_elastic_info", "", "Ascend910_93", 20, 196608, 0, {}, ge::DT_FLOAT, {}, ge::DT_FLOAT, {}, ge::DT_FLOAT, {}, ge::DT_FLOAT, {}, ge::DT_FLOAT, {}, ge::DT_FLOAT, {}, ge::DT_FLOAT, {}, ge::DT_FLOAT, {}, ge::DT_FLOAT, {}, ge::DT_FLOAT, {}, ge::DT_FLOAT, {}, ge::DT_FLOAT, {}, ge::DT_FLOAT, {}, ge::DT_FLOAT, "ep_group", 72, 0, 216, "tp_group", 1, 0, 0, 1, 18, 0, 0, 1, "", 0, 0, 0, false, 0},
    {"moe_distribute_dispatch_test_zeroComputeExpertNum_invalid", "", "Ascend910_93", 20, 196608, 0, {}, ge::DT_FLOAT, {}, ge::DT_FLOAT, {}, ge::DT_FLOAT, {}, ge::DT_FLOAT, {}, ge::DT_FLOAT, {}, ge::DT_FLOAT, {}, ge::DT_FLOAT, {}, ge::DT_FLOAT, {}, ge::DT_FLOAT, {}, ge::DT_FLOAT, {}, ge::DT_FLOAT, {}, ge::DT_FLOAT, {}, ge::DT_FLOAT, {}, ge::DT_FLOAT, "ep_group", 8, 0, 7, "tp_group", 1, 0, 0, 1, 1, 0, 0, 1, "", 0xFFFFFFFF, 2, 3, false, 0},
};

TEST_P(MoeDistributeDispatchV2TilingParam, general_case)
{
    const auto &param = GetParam();
    TestOneParamCase(param);
}

INSTANTIATE_TEST_SUITE_P(
    general_cases_params,
    MoeDistributeDispatchV2TilingParam,
    ::testing::ValuesIn(cases_params),
    [](const ::testing::TestParamInfo<MoeDistributeDispatchV2TilingTestParam> &info) {
        std::string name = info.param.case_name;
        for (char &c : name) { if (!std::isalnum(static_cast<unsigned char>(c)) && c != '_') c = '_'; }
        return name;
    });
} // anonymous namespace
} // namespace MoeDistributeDispatchV2UT
