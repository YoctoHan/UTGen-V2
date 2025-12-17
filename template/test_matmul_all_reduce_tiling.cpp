/**
 * Copyright (c) Huawei Technologies Co., Ltd. 2025. All rights reserved.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <iostream>
#include <cctype>
#include <gtest/gtest.h>
#include "mc2_tiling_case_executor.h"

namespace MatmulAllReduceUT {
template <typename T>
auto build_from(const T& value){
    return Ops::Transformer::AnyValue::CreateFrom<T>(value);
}

struct MatmulAllReduceTilingTestParam {
    uint64_t inputTotalNum;
    string case_name;
    string compile_info;
    string soc_version;
    uint64_t coreNum;
    uint64_t ubSize;
    uint64_t tilingDataSize;

    std::initializer_list<int64_t> x1_shape;
    std::initializer_list<int64_t> x2_shape;
    std::initializer_list<int64_t> bias_shape;
    std::initializer_list<int64_t> x3_shape;
    std::initializer_list<int64_t> antiquant_scale_shape;
    std::initializer_list<int64_t> antiquant_offset_shape;
    std::initializer_list<int64_t> dequant_scale_shape;
    std::initializer_list<int64_t> pertoken_scale_shape;
    std::initializer_list<int64_t> comm_quant_scale_1_shape;
    std::initializer_list<int64_t> comm_quant_scale_2_shape;
    std::initializer_list<int64_t> output_shape;

    ge::DataType x1_dtype;
    ge::DataType x2_dtype;
    ge::DataType bias_dtype;
    ge::DataType x3_dtype;
    ge::DataType antiquant_scale_dtype;
    ge::DataType antiquant_offset_dtype;
    ge::DataType dequant_scale_dtype;
    ge::DataType pertoken_scale_dtype;
    ge::DataType comm_quant_scale_1_dtype;
    ge::DataType comm_quant_scale_2_dtype;
    ge::DataType output_dtype;

    bool is_trans_a;
    bool is_trans_b;

    uint64_t expectTilingKey;
};

class MatmulAllReduceTilingParam : public ::testing::TestWithParam<MatmulAllReduceTilingTestParam> {
protected:
    static void SetUpTestCase()
    {
        std::cout << "MatmulAllReduceTiling SetUp" << std::endl;
    }

    static void TearDownTestCase()
    {
        std::cout << "MatmulAllReduceTiling TearDown" << std::endl;
    }
};

gert::StorageShape make_shape(const std::initializer_list<int64_t>& input_shape){
    if (input_shape.size() == 0){
        return gert::StorageShape{};
    }
    return gert::StorageShape{input_shape, input_shape};
}

void TestOneParamCase(const MatmulAllReduceTilingTestParam& param){
    struct MatmulAllReduceCompileInfo {};
    MatmulAllReduceCompileInfo compileInfo;

    std::vector<pair<std::initializer_list<int64_t>, ge::DataType>> shapeDtypeList = {
    {param.x1_shape, param.x1_dtype}, 
    {param.x2_shape, param.x2_dtype}, 
    {param.bias_shape, param.bias_dtype}, 
    {param.x3_shape, param.x3_dtype}, 
    {param.antiquant_scale_shape, param.antiquant_scale_dtype}, 
    {param.antiquant_offset_shape, param.antiquant_offset_dtype}, 
    {param.dequant_scale_shape, param.dequant_scale_dtype}, 
    {param.pertoken_scale_shape, param.pertoken_scale_dtype}, 
    {param.comm_quant_scale_1_shape, param.comm_quant_scale_1_dtype}, 
    {param.comm_quant_scale_2_shape, param.comm_quant_scale_2_dtype}
    };

    std::vector<gert::TilingContextPara::TensorDescription> inputList;
    for (int i = 0; i < param.inputTotalNum; i++){
        inputList.push_back({make_shape(shapeDtypeList[i].first), shapeDtypeList[i].second, ge::FORMAT_ND});
    }

    gert::TilingContextPara tilingContextPara("MatmulAllReduce", inputList,
        {
            {{param.output_shape, param.output_shape}, param.output_dtype, ge::FORMAT_ND},
        },
        {
            {"group", build_from<std::string>("group")},
            {"reduce_op", build_from<std::string>("sum")},
            {"is_trans_a", build_from<bool>(param.is_trans_a)},
            {"is_trans_b", build_from<bool>(param.is_trans_b)},
            {"comm_turn", build_from<int64_t>(0)},
            {"antiquant_group_size", build_from<int64_t>(0)},
            {"group_size", build_from<int64_t>(0)},
            {"y_dtype", build_from<int64_t>(0)},
            {"comm_quant_mode", build_from<int64_t>(0)}
        },
        &compileInfo, param.soc_version, param.compile_info, param.tilingDataSize);

    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, param.expectTilingKey);
}

TEST_P(MatmulAllReduceTilingParam, general_case)
{
    Mc2Hcom::MockValues hcomTopologyMockValues{{"rankNum", 8}};
    Mc2Hcom::MC2HcomTopologyMocker::GetInstance().SetValues(hcomTopologyMockValues);

    const auto &param = GetParam();
    TestOneParamCase(param);

    Mc2Hcom::MC2HcomTopologyMocker::GetInstance().Reset();
}

INSTANTIATE_TEST_SUITE_P(
    general_cases_params,
    MatmulAllReduceTilingParam,
    ::testing::ValuesIn(cases_params),
    [](const ::testing::TestParamInfo<MatmulAllReduceTilingTestParam> &info) {
        std::string name = info.param.case_name;
        for (char &c : name) {
            if (!std::isalnum(static_cast<unsigned char>(c)) && c != '_') {
                c = '_';
            }
        }
        return name;
    });

} // namespace MatmulAllReduceUT