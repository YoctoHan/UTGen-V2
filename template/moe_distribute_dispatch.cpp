/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd. All rights reserved.
 */
#include <iostream>
#include <vector>
#include <string>
#include <gtest/gtest.h>
#include "mc2_tiling_case_executor.h"

namespace MoeDistributeDispatchUT {
namespace {
template <typename T>
auto build_from(const T &value) { return Ops::Transformer::AnyValue::CreateFrom<T>(value); }

gert::StorageShape make_shape(const std::initializer_list<int64_t> &input_shape)
{
    if (input_shape.size() == 0) { return gert::StorageShape{}; }
    return gert::StorageShape{input_shape, input_shape};
}

struct MoeDistributeDispatchTilingTestParam {
    // 平台信息
    uint64_t inputTotalNum;
    std::string case_name;
    std::string compile_info;
    std::string soc_version;
    uint64_t coreNum;
    uint64_t ubSize;
    uint64_t tilingDataSize;

    std::initializer_list<int64_t> input0_shape;
    ge::DataType input0_dtype;
    std::initializer_list<int64_t> input1_shape;
    ge::DataType input1_dtype;
    std::initializer_list<int64_t> input2_shape;
    ge::DataType input2_dtype;
    std::initializer_list<int64_t> input3_shape;
    ge::DataType input3_dtype;
    std::initializer_list<int64_t> input4_shape;
    ge::DataType input4_dtype;
    std::initializer_list<int64_t> output0_shape;
    ge::DataType output0_dtype;
    std::initializer_list<int64_t> output1_shape;
    ge::DataType output1_dtype;
    std::initializer_list<int64_t> output2_shape;
    ge::DataType output2_dtype;
    std::initializer_list<int64_t> output3_shape;
    ge::DataType output3_dtype;
    std::initializer_list<int64_t> output4_shape;
    ge::DataType output4_dtype;
    std::initializer_list<int64_t> output5_shape;
    ge::DataType output5_dtype;

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
    bool has_expect_tiling_key;
    uint64_t expect_tiling_key;
};

class MoeDistributeDispatchTilingParam : public ::testing::TestWithParam<MoeDistributeDispatchTilingTestParam> {
protected:
    static void SetUpTestCase() { std::cout << "SetUp" << std::endl; }
    static void TearDownTestCase() { std::cout << "TearDown" << std::endl; }
};

void TestOneParamCase(const MoeDistributeDispatchTilingTestParam &param)
{
    struct MoeDistributeDispatchCompileInfo {} compileInfo;

    // 输入列表映射
    std::vector<std::pair<std::initializer_list<int64_t>, ge::DataType>> shapeDtypeList = {
        {param.input0_shape, param.input0_dtype},
        {param.input1_shape, param.input1_dtype},
        {param.input2_shape, param.input2_dtype},
        {param.input3_shape, param.input3_dtype},
        {param.input4_shape, param.input4_dtype},
    };

    std::vector<gert::TilingContextPara::TensorDescription> inputList;
    for (uint64_t i = 0; i < param.inputTotalNum && i < shapeDtypeList.size(); ++i) {
        inputList.push_back({make_shape(shapeDtypeList[i].first), shapeDtypeList[i].second, ge::FORMAT_ND});
    }

    std::vector<gert::TilingContextPara::TensorDescription> outputList;
    outputList.push_back({make_shape(param.output0_shape), param.output0_dtype, ge::FORMAT_ND});
    outputList.push_back({make_shape(param.output1_shape), param.output1_dtype, ge::FORMAT_ND});
    outputList.push_back({make_shape(param.output2_shape), param.output2_dtype, ge::FORMAT_ND});
    outputList.push_back({make_shape(param.output3_shape), param.output3_dtype, ge::FORMAT_ND});
    outputList.push_back({make_shape(param.output4_shape), param.output4_dtype, ge::FORMAT_ND});
    outputList.push_back({make_shape(param.output5_shape), param.output5_dtype, ge::FORMAT_ND});

    gert::TilingContextPara tilingContextPara("MoeDistributeDispatch", inputList, outputList,
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
        },
        &compileInfo, param.soc_version, param.coreNum, param.ubSize, param.tilingDataSize);

    Mc2Hcom::MockValues hcomTopologyMockValues{{"rankNum", 8}};
    if (param.has_expect_tiling_key) {
        Mc2ExecuteTestCase(tilingContextPara, hcomTopologyMockValues, ge::GRAPH_SUCCESS, param.expect_tiling_key);
    } else {
        Mc2ExecuteTestCase(tilingContextPara, hcomTopologyMockValues);
    }
}

TEST_P(MoeDistributeDispatchTilingParam, general_case)
{
    const auto &param = GetParam();
    TestOneParamCase(param);
}

INSTANTIATE_TEST_SUITE_P(
    general_cases_params,
    MoeDistributeDispatchTilingParam,
    ::testing::ValuesIn(cases_params),
    [](const ::testing::TestParamInfo<MoeDistributeDispatchTilingTestParam> &info) {
        std::string name = info.param.case_name;
        for (char &c : name) { if (!std::isalnum(static_cast<unsigned char>(c)) && c != '_') c = '_'; }
        return name;
    });
} // anonymous namespace
} // namespace MoeDistributeDispatchUT
