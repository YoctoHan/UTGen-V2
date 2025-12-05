/**
 * This program is free software, you can redistribute it and/or modify.
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "../../../op_host/op_tiling/allto_allv_grouped_mat_mul_tiling.h"

#include <iostream>
#include <gtest/gtest.h>

#include "mc2_tiling_case_executor.h"

using namespace std;

namespace AlltoAllvGroupedMatMulUT {

namespace {

template <typename T>
auto build_from(const T &value)
{
    return Ops::Transformer::AnyValue::CreateFrom<T>(value);
}
struct TestParam {
    string test_name{};
    std::vector<std::pair<string, string>> tiling_params_str_pair{};
    std::vector<std::pair<string, std::vector<int64_t>>> tiling_params_vec_pair{};
    std::vector<std::pair<size_t, ge::DataType>> tiling_dTypes_pair{};
    ge::graphStatus status;
};

struct TilingParams {
    uint64_t BSK{4096};
    uint64_t BS{2048};
    uint64_t K{2};
    uint64_t H1{7168};
    uint64_t H2{7168};
    uint64_t A{4096};
    uint64_t N1{4096};
    uint64_t N2{64};
    uint64_t ep_world_size{8};
    uint64_t e{4};
    uint64_t commOut;
    uint64_t aivCoreNum{40};
    uint64_t aicCoreNum{20};
    uint64_t totalUbSize{196608};
    uint64_t gmm_weight_dim1{7168};
    uint64_t gmm_y_dim1{4096};
    uint64_t mm_weight_dim0{7168};
    bool trans_gmm_weight{false};
    bool trans_mm_weight{false};
    bool permute_out_flag{false};
    bool is_Need_MM{true};
    std::string group{"group"};
    std::vector<int64_t> send_counts{128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128,
                                     128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128};
    std::vector<int64_t> recv_counts{128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128,
                                     128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128};
};

std::unordered_map<string, std::function<void(TilingParams& tiling_params, const string& value_str)>>
    tiling_params_str_handlers = {
        {"BSK", [](TilingParams& tiling_params, const string& value_str) { tiling_params.BSK = std::stoi(value_str); }},
        {"BS", [](TilingParams& tiling_params, const string& value_str) { tiling_params.BS = std::stoi(value_str); }},
        {"K", [](TilingParams& tiling_params, const string& value_str) { tiling_params.K = std::stoi(value_str); }},
        {"H1", [](TilingParams& tiling_params, const string& value_str) { tiling_params.H1 = std::stoi(value_str); }},
        {"H2", [](TilingParams& tiling_params, const string& value_str) { tiling_params.H2 = std::stoi(value_str); }},
        {"A", [](TilingParams& tiling_params, const string& value_str) { tiling_params.A = std::stoi(value_str); }},
        {"N1", [](TilingParams& tiling_params, const string& value_str) { tiling_params.N1 = std::stoi(value_str); }},
        {"N2", [](TilingParams& tiling_params, const string& value_str) { tiling_params.N2 = std::stoi(value_str); }},
        {"ep_world_size", [](TilingParams& tiling_params,
                             const string& value_str) { tiling_params.ep_world_size = std::stoi(value_str); }},
        {"e", [](TilingParams& tiling_params, const string& value_str) { tiling_params.e = std::stoi(value_str); }},
        {"gmm_weight_dim1", [](TilingParams& tiling_params,
                               const string& value_str) { tiling_params.gmm_weight_dim1 = std::stoi(value_str); }},
        {"gmm_y_dim1",
         [](TilingParams& tiling_params, const string& value_str) { tiling_params.gmm_y_dim1 = std::stoi(value_str); }},
        {"mm_weight_dim0", [](TilingParams& tiling_params,
                              const string& value_str) { tiling_params.mm_weight_dim0 = std::stoi(value_str); }},
        {"trans_gmm_weight", [](TilingParams& tiling_params,
                                const string& value_str) { tiling_params.trans_gmm_weight = value_str == "true"; }},
        {"trans_mm_weight", [](TilingParams& tiling_params,
                               const string& value_str) { tiling_params.trans_mm_weight = value_str == "true"; }},
        {"permute_out_flag", [](TilingParams& tiling_params, const string& value_str) {
             tiling_params.permute_out_flag = value_str == "true";
         }},
        {"is_Need_MM", [](TilingParams& tiling_params, const string& value_str) {
             tiling_params.is_Need_MM = value_str == "true";
         }}
        };

std::unordered_map<string, std::function<void(TilingParams& tiling_params, const std::vector<int64_t> value_vec)>>
    tiling_params_vec_handlers = {
        {"send_counts", [](TilingParams& tiling_params,
                           const std::vector<int64_t> value_vec) { tiling_params.send_counts = value_vec; }},
        {"recv_counts", [](TilingParams& tiling_params, const std::vector<int64_t> value_vec) {
             tiling_params.recv_counts = value_vec;
         }}};

bool has_any_target_key(
    const std::vector<std::pair<std::string, std::string>>& params,
    const std::vector<std::string>& targets
) {
    return std::any_of(
        params.begin(),
        params.end(),
        [&targets](const auto& p) {
            return std::find(targets.begin(), targets.end(), p.first) != targets.end();
        }
    );
}

// 提取：初始化 tiling_params
void InitializeTilingParams(
    const TestParam& test_param,
    TilingParams& tiling_params
) {
    for (auto& kv : test_param.tiling_params_str_pair) {
        if (tiling_params_str_handlers.count(kv.first) != 0) {
            tiling_params_str_handlers[kv.first](tiling_params, kv.second);
        }
    }

    for (auto& kv : test_param.tiling_params_vec_pair) {
        if (tiling_params_vec_handlers.count(kv.first) != 0) {
            tiling_params_vec_handlers[kv.first](tiling_params, kv.second);
        }
    }
}

std::unique_ptr<gert::TilingContextPara::TensorDescription> CreateTensorShape(
    gert::StorageShape shape,
    ge::DataType dtype,
    ge::Format format
) {
    return std::unique_ptr<gert::TilingContextPara::TensorDescription>(
        new gert::TilingContextPara::TensorDescription(
            shape,  // 这里用大括号构造
            dtype,
            format
        )
    );
}

std::vector<gert::TilingContextPara::TensorDescription> CreateInputTensors(
    const TilingParams& tiling_params,
    const std::unique_ptr<gert::TilingContextPara::TensorDescription>& mm_x_shape,
    const std::unique_ptr<gert::TilingContextPara::TensorDescription>& mm_weight_shape
) {
    return {
        {{{tiling_params.BSK, tiling_params.H1}, {tiling_params.BSK, tiling_params.H1}},
         ge::DT_FLOAT16, ge::FORMAT_ND},
        {{{tiling_params.e, tiling_params.gmm_weight_dim1, tiling_params.N1}, {tiling_params.e, tiling_params.gmm_weight_dim1, tiling_params.N1}},
         ge::DT_FLOAT16, ge::FORMAT_ND},
        {{}, ge::DT_FLOAT16, ge::FORMAT_ND}, // placeholder
        {{}, ge::DT_FLOAT16, ge::FORMAT_ND}, // placeholder
        *mm_x_shape,
        *mm_weight_shape, 
    };
}

std::vector<gert::TilingContextPara::TensorDescription> CreateOutputTensors(
    const TilingParams& tiling_params,
    const std::unique_ptr<gert::TilingContextPara::TensorDescription>& mm_y_shape
) {
    return {
        {{{tiling_params.A, tiling_params.gmm_y_dim1}, {tiling_params.A, tiling_params.gmm_y_dim1}},
         ge::DT_FLOAT16, ge::FORMAT_ND},
        *mm_y_shape,
        {{{tiling_params.A, tiling_params.H1}, {tiling_params.A, tiling_params.H1}},
         ge::DT_FLOAT16, ge::FORMAT_ND},
    };
}

std::vector<std::pair<std::string, Ops::Transformer::AnyValue>> CreateAttrs(
    const TestParam& test_param,
    const TilingParams& tiling_params
) {
    return {
        {"group", Ops::Transformer::AnyValue::CreateFrom<std::string>(tiling_params.group)},
        {"ep_world_size", Ops::Transformer::AnyValue::CreateFrom<int64_t>(tiling_params.ep_world_size)},
        {"send_counts", Ops::Transformer::AnyValue::CreateFrom<vector<int64_t>>(tiling_params.send_counts)},
        {"recv_counts", Ops::Transformer::AnyValue::CreateFrom<vector<int64_t>>(tiling_params.recv_counts)},
        {"trans_gmm_weight", Ops::Transformer::AnyValue::CreateFrom<bool>(false)},
        {"trans_mm_weight", Ops::Transformer::AnyValue::CreateFrom<bool>(false)},
        {"permute_out_flag", Ops::Transformer::AnyValue::CreateFrom<bool>(tiling_params.permute_out_flag)}
    };
}

class AlltoAllvGroupedMatMulTiling : public testing::TestWithParam<TestParam>
{
protected:
    static void SetUpTestCase()
    {
        std::cout << "AlltoAllvGroupedMatMulTiling Test SetUp" << std::endl;
    }

    static void TearDownTestCase()
    {
        std::cout << "AlltoAllvGroupedMatMulTiling Test TearDown" << std::endl;
    }

public:
    std::vector<int64_t> send_counts{128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128,
                                     128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128};
    std::vector<int64_t> recv_counts{128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128,
                                     128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128};
};

void TestOneParamCase(const TestParam &test_param)
{
    struct AlltoAllvGroupedMatMulCompileInfo {};
    AlltoAllvGroupedMatMulCompileInfo compileInfo;

    std::string socVersion = "Ascend910_93";
    uint64_t coreNum = 20;
    uint64_t ubSize = 196608;
    uint64_t tilingDataSize = 8192;

    TilingParams tiling_params;
    InitializeTilingParams(test_param, tiling_params);

    std::vector<std::string> targets = {"BS", "H2", "mm_weight_dim0", "N2"};

    auto mm_x_shape = CreateTensorShape({{tiling_params.BS, tiling_params.H2}, {tiling_params.BS, tiling_params.H2}},
                                        ge::DT_FLOAT16, ge::FORMAT_ND);
    auto mm_weight_shape =
        CreateTensorShape({{tiling_params.mm_weight_dim0, tiling_params.N2}, {tiling_params.mm_weight_dim0, tiling_params.N2}},
                          ge::DT_FLOAT16, ge::FORMAT_ND);
    auto mm_y_shape = CreateTensorShape({{tiling_params.BS, tiling_params.N2}, {tiling_params.BS, tiling_params.N2}},
                                        ge::DT_FLOAT16, ge::FORMAT_ND);

    if (!(has_any_target_key(test_param.tiling_params_str_pair, targets) || tiling_params.is_Need_MM == false)) {
        mm_x_shape->shape_ = {};
        mm_weight_shape->shape_ = {};
        mm_y_shape->shape_ = {};
    }

    gert::TilingContextPara tilingContextPara(
        "AlltoAllvGroupedMatMul",
        CreateInputTensors(tiling_params, mm_x_shape, mm_weight_shape),
        CreateOutputTensors(tiling_params, mm_y_shape),
        {
            {"group", build_from<std::string>(tiling_params.group)},
            {"ep_world_size", build_from<int64_t>(tiling_params.ep_world_size)},
            {"send_counts", build_from<vector<int64_t>>(tiling_params.send_counts)},
            {"recv_counts", build_from<vector<int64_t>>(tiling_params.recv_counts)},
            {"trans_gmm_weight", build_from<bool>(false)},
            {"trans_mm_weight", build_from<bool>(false)},
            {"permute_out_flag", build_from<bool>(tiling_params.permute_out_flag)},
        },
        &compileInfo, socVersion, coreNum, ubSize, tilingDataSize);

    Mc2Hcom::MockValues hcomTopologyMockValues{{"rankNum", 8}};
    if (test_param.status == ge::GRAPH_FAILED) {
        Mc2ExecuteTestCase(tilingContextPara, hcomTopologyMockValues);
    } else {
        uint64_t expectTilingKey = 1000UL;
        if (test_param.test_name == "Test_no_MM") {
            expectTilingKey = 256UL;
        }
        Mc2ExecuteTestCase(tilingContextPara, hcomTopologyMockValues, ge::GRAPH_SUCCESS, expectTilingKey);
    }
}

TEST_P(AlltoAllvGroupedMatMulTiling, general_case)
{
    if (!IsOpImplRegistryAvailable()) {
        GTEST_SKIP() << "Skip test: OpImplSpaceRegistryV2 is null on host.";
    }
    const auto &param = GetParam();
    TestOneParamCase(param);
}


INSTANTIATE_TEST_SUITE_P(
    AlltoAllvGroupedMatMul,
    AlltoAllvGroupedMatMulTiling,
    testing::ValuesIn(test_params),
    [](const testing::TestParamInfo<AlltoAllvGroupedMatMulTiling::ParamType> &info) {
        std::string name = info.param.test_name;
        for (char &c : name) {
            if (!std::isalnum(static_cast<unsigned char>(c)) && c != '_') {
                c = '_';
            }
        }
        return name;
    });

} // anonymous namespace

} // allto_allv_grouped_mat_mul_ut