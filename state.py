from typing import TypedDict, List, Optional, Dict, Any, Literal
from config import OperatorName, OpType, OPS_TRANSFORMERS_DIR
import os
from pathlib import Path


class WorkflowState(TypedDict):
    # ========== 步骤1：输入信息 ==========
    operator_name: Literal["all_gather_matmul", "matmul_all_reduce", "distribute_barrier"]
    op_type: Literal["op_host"]

    # ========== 步骤2：模型输出 ==========
    input_path: str

    # ========== 步骤3：模板相关 ==========
    def_file_path: str
    template_file_path: str

    # ========== 步骤4：输出信息 ==========
    output_path: str


def create_initial_state(operator_name: OperatorName, operator_type: OpType) -> WorkflowState:
    """
    根据算子名称和类型生成初始状态，包括：
    - def_file_path: 指向 ops-transformer 仓库中的 *_def.cpp
      例如: /workspace/ops-transformer-dev/mc2/all_gather_matmul/op_host/all_gather_matmul_def.cpp
    - template_file_path: 当前工程下的模板输出路径
      例如: /workspace/UTGen-V2/template/all_gather_matmul.cpp
    - output_path: 当前工程下最终 UT 代码输出路径
      例如: /workspace/UTGen-V2/target/test_all_gather_matmul_tiling.cpp
    """
    # ops-transformer 仓库中的 def.cpp
    op_name = operator_name.value
    op_type = operator_type.value

    def_file_path = os.path.join(
        OPS_TRANSFORMERS_DIR,
        "mc2",
        op_name,
        op_type,
        f"{op_name}_def.cpp",
    )

    # 本工程根目录
    project_root = Path(__file__).parent.absolute()

    # 模板和最终 UT 代码路径
    template_file_path = project_root / "template" / f"{op_name}.cpp"
    output_path = project_root / "target" / f"test_{op_name}_tiling.cpp"

    input_path = project_root / "input" / f"{op_name}.jsonl"

    return WorkflowState(
        operator_name=op_name,
        op_type=op_type,
        input_path=str(input_path),
        def_file_path=str(def_file_path),
        template_file_path=str(template_file_path),
        output_path=str(output_path),
    )

