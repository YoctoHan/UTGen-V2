import argparse
import sys
from pathlib import Path
from enum import Enum

_current_dir = Path(__file__).parent.absolute()
if str(_current_dir) not in sys.path:
    sys.path.insert(0, str(_current_dir))

from langgraph.graph import StateGraph, END

from state import WorkflowState, create_initial_state
from config import OperatorName, OpType
from nodes.generate_template import generate_template
from nodes.generate_unit_test import generate_unit_test


def create_workflow():
    workflow = StateGraph[WorkflowState, None, WorkflowState, WorkflowState](WorkflowState)
    
    workflow.add_node("generate_template", generate_template)
    workflow.add_node("generate_unit_test", generate_unit_test)
    
    workflow.set_entry_point("generate_template")
    
    workflow.add_edge("generate_template", "generate_unit_test")
    
    return workflow.compile()
    

def main():
    parser = argparse.ArgumentParser(
        description=(
            "UTGen-V2\n"
            "\n"
            "示例:\n"
            "  python workflow.py -n all_gather_matmul -t op_host\n"
            "  python workflow.py -n matmul_all_reduce -t op_host\n"
            "  python workflow.py -n distribute_barrier -t op_host\n"
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "-n",
        "--operator_name",
        dest="operator_name",
        type=OperatorName,
        choices=[op.value for op in OperatorName],
        required=True,
        help=(
            "指定算子名称，可选值: "
            f"{', '.join(op.value for op in OperatorName)}"
        ),
    )
    parser.add_argument(
        "-t",
        "--op_type",
        dest="op_type",
        type=OpType,
        choices=[t.value for t in OpType],
        required=True,
        help=(
            "指定算子类型，可选值: "
            f"{', '.join(t.value for t in OpType)}"
        ),
    )

    # 如果用户没有传任何参数，优先展示帮助信息，而不是报错格式
    if len(sys.argv) == 1:
        parser.print_help()
        return

    args = parser.parse_args()
    
    app = create_workflow()
    
    initial_state = create_initial_state(args.operator_name, args.op_type)
    
    result = app.invoke(initial_state)
    
    print("\n" + "="*80)
    print("✅ Workflow 完成！")
    print("="*80)
    print(f"最终状态: {result.get('operator_name', 'N/A')} - {result.get('op_type', 'N/A')}")
    return result


if __name__ == "__main__":
    main()

