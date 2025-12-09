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

MoeDistributeDispatchTilingTestParam cases_params[] = {
    {2, "moe_distribute_dispatch_test_tiling_0", "", "Ascend910_93", 20, 196608, 0, {32, 7168}, ge::DT_FLOAT16, {32, 8}, ge::DT_INT32, {}, ge::DT_FLOAT16, {}, ge::DT_FLOAT, {}, ge::DT_FLOAT, {576, 7168}, ge::DT_FLOAT16, {576}, ge::DT_FLOAT16, {256}, ge::DT_INT32, {1}, ge::DT_INT64, {288}, ge::DT_INT32, {2}, ge::DT_INT32, "ep_group", 288, 8, 256, "tp_group", 2, 0, 0, 1, 32, 0, 0, 1, false, 0},
    {2, "moe_distribute_dispatch_test_tiling_1", "", "Ascend910_93", 20, 196608, 0, {16, 7160}, ge::DT_FLOAT16, {16, 8}, ge::DT_INT32, {}, ge::DT_FLOAT16, {}, ge::DT_FLOAT, {}, ge::DT_FLOAT, {576, 7160}, ge::DT_FLOAT16, {576}, ge::DT_FLOAT, {128}, ge::DT_INT32, {1}, ge::DT_INT64, {288}, ge::DT_INT32, {2}, ge::DT_INT32, "ep_group", 288, 0, 256, "tp_group", 2, 0, 0, 1, 32, 0, 0, 1, false, 0},
    {2, "moe_distribute_dispatch_test_tiling_2", "", "Ascend910_93", 20, 196608, 0, {16, 7160}, ge::DT_FLOAT16, {16, 8}, ge::DT_INT32, {}, ge::DT_FLOAT16, {}, ge::DT_FLOAT, {}, ge::DT_FLOAT, {576, 7160}, ge::DT_FLOAT16, {576}, ge::DT_FLOAT, {128}, ge::DT_INT32, {1}, ge::DT_INT64, {288}, ge::DT_INT32, {2}, ge::DT_INT32, "ep_group", 288, 0, 256, "tp_group", 2, 1024, 0, 1, 32, 0, 0, 1, false, 0},
    {2, "moe_distribute_dispatch_test_tiling_3", "", "Ascend910_93", 20, 196608, 0, {16, 7160}, ge::DT_FLOAT16, {16, 8}, ge::DT_INT32, {}, ge::DT_FLOAT16, {}, ge::DT_FLOAT, {}, ge::DT_FLOAT, {576, 7160}, ge::DT_FLOAT16, {576}, ge::DT_FLOAT, {128}, ge::DT_INT32, {1}, ge::DT_INT64, {288}, ge::DT_INT32, {2}, ge::DT_INT32, "ep_group", 288, 0, 256, "tp_group", 2, 0, 0, 1, 31, 0, 0, 0, false, 0},
    {3, "moe_distribute_dispatch_test_tiling_4", "", "Ascend910_93", 20, 196608, 0, {16, 7168}, ge::DT_FLOAT16, {16, 8}, ge::DT_INT32, {33, 7168}, ge::DT_FLOAT, {}, ge::DT_FLOAT, {}, ge::DT_FLOAT, {576, 7168}, ge::DT_INT8, {576}, ge::DT_FLOAT, {128}, ge::DT_INT32, {1}, ge::DT_INT64, {288}, ge::DT_INT32, {2}, ge::DT_INT32, "ep_group", 288, 0, 257, "tp_group", 2, 0, 0, 1, 31, 1, 0, 0, false, 0},
    {2, "moe_distribute_dispatch_test_tiling_5", "", "Ascend910_93", 20, 196608, 0, {16, 7168}, ge::DT_FLOAT16, {16, 8}, ge::DT_INT32, {}, ge::DT_FLOAT16, {}, ge::DT_FLOAT, {}, ge::DT_FLOAT, {576, 7168}, ge::DT_FLOAT16, {576}, ge::DT_FLOAT, {128}, ge::DT_INT32, {1}, ge::DT_INT64, {288}, ge::DT_INT32, {2}, ge::DT_INT32, "ep_group", 288, 0, 256, "tp_group", 2, 0, 0, 1, 32, 10, 0, 0, false, 0},
    {2, "moe_distribute_dispatch_test_tiling_7", "", "Ascend910_93", 20, 196608, 0, {32, 7168}, ge::DT_FLOAT16, {32, 8}, ge::DT_INT32, {}, ge::DT_FLOAT16, {}, ge::DT_FLOAT, {}, ge::DT_FLOAT, {576, 7168}, ge::DT_FLOAT16, {576}, ge::DT_FLOAT, {256}, ge::DT_INT32, {1}, ge::DT_INT64, {288}, ge::DT_INT32, {2}, ge::DT_INT32, "ep_group", 288, 0, 256, "tp_group", 2, -1, 0, 1, 32, 2, 0, 0, false, 0},
    {2, "moe_distribute_dispatch_test_tiling_8", "", "Ascend910_93", 20, 196608, 0, {32, 7168}, ge::DT_FLOAT16, {32, 8}, ge::DT_INT32, {}, ge::DT_FLOAT16, {}, ge::DT_FLOAT, {}, ge::DT_FLOAT, {576, 7168}, ge::DT_FLOAT16, {576}, ge::DT_FLOAT, {256}, ge::DT_INT32, {1}, ge::DT_INT64, {288}, ge::DT_INT32, {2}, ge::DT_INT32, "ep_group", 288, 1, 256, "", 2, 1024, 1, 1, 32, 2, 1, 1, false, 0},
    {2, "moe_distribute_dispatch_test_tiling_9", "", "Ascend910_93", 20, 196608, 0, {16, 7168}, ge::DT_FLOAT16, {16, 8}, ge::DT_INT32, {}, ge::DT_FLOAT16, {}, ge::DT_FLOAT, {}, ge::DT_FLOAT, {576, 7168}, ge::DT_INT8, {576}, ge::DT_FLOAT, {128}, ge::DT_INT32, {1}, ge::DT_INT64, {288}, ge::DT_INT32, {1}, ge::DT_INT32, "ep_group", 288, 0, 256, "", 2, -1, 0, 1, 32, 2, 0, 0, false, 0},
    {2, "moe_distribute_dispatch_test_tiling_10", "", "Ascend910_93", 20, 196608, 0, {32, 7168}, ge::DT_FLOAT16, {32, 8}, ge::DT_INT32, {}, ge::DT_FLOAT16, {}, ge::DT_FLOAT, {}, ge::DT_FLOAT, {576, 7168}, ge::DT_FLOAT16, {576}, ge::DT_FLOAT, {256}, ge::DT_INT32, {1}, ge::DT_INT64, {288}, ge::DT_INT32, {2}, ge::DT_INT32, "ep_group", 288, 256, 256, "tp_group", 2, 0, 0, 1, 32, 0, 0, 1, false, 0},
    {2, "moe_distribute_dispatch_test_tiling_A2_ShapeAndEp_rank_id", "", "Ascend910B", 48, 196608, 0, {8, 7160}, ge::DT_FLOAT16, {8, 8}, ge::DT_INT32, {}, ge::DT_FLOAT16, {}, ge::DT_FLOAT, {}, ge::DT_FLOAT, {2048, 7168}, ge::DT_INT8, {2048}, ge::DT_FLOAT, {64}, ge::DT_INT32, {8}, ge::DT_INT64, {256}, ge::DT_INT32, {1}, ge::DT_INT32, "ep_group", 32, 33, 256, "", 0, 0, 0, 1, 0, 2, 0, 0, false, 0},
    {2, "moe_distribute_dispatch_test_tiling_A2_moe_expert_num", "", "Ascend910B", 48, 196608, 0, {8, 7168}, ge::DT_FLOAT16, {8, 8}, ge::DT_INT32, {}, ge::DT_FLOAT16, {}, ge::DT_FLOAT, {}, ge::DT_FLOAT, {2048, 7168}, ge::DT_INT8, {2048}, ge::DT_FLOAT, {64}, ge::DT_INT32, {8}, ge::DT_INT64, {256}, ge::DT_INT32, {1}, ge::DT_INT32, "ep_group", 32, 0, 257, "", 0, 0, 0, 1, 0, 2, 0, 0, false, 0},
    {2, "moe_distribute_dispatch_test_tiling_ep_world_size_384", "", "Ascend910_93", 20, 196608, 0, {32, 7168}, ge::DT_FLOAT16, {32, 8}, ge::DT_INT32, {}, ge::DT_FLOAT16, {}, ge::DT_FLOAT, {}, ge::DT_FLOAT, {576, 7168}, ge::DT_FLOAT16, {576}, ge::DT_FLOAT, {256}, ge::DT_INT32, {1}, ge::DT_INT64, {288}, ge::DT_INT32, {2}, ge::DT_INT32, "ep_group", 384, 0, 256, "tp_group", 2, 0, 0, 1, 32, 0, 0, 1, false, 0},
    {2, "moe_distribute_dispatch_test_tiling_ep_world_size_72", "", "Ascend910_93", 20, 196608, 0, {32, 7168}, ge::DT_FLOAT16, {32, 8}, ge::DT_INT32, {}, ge::DT_FLOAT16, {}, ge::DT_FLOAT, {}, ge::DT_FLOAT, {576, 7168}, ge::DT_FLOAT16, {576}, ge::DT_FLOAT, {256}, ge::DT_INT32, {1}, ge::DT_INT64, {288}, ge::DT_INT32, {2}, ge::DT_INT32, "ep_group", 72, 0, 216, "tp_group", 2, 0, 0, 1, 18, 0, 0, 1, false, 0},
};

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
