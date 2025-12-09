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


def check_template_existence(state: WorkflowState) -> str:
    """检查模板文件是否存在，决定下一步操作"""
    template_path = Path(state["template_file_path"])
    if template_path.exists():
        print(f"模板文件已存在: {template_path}，跳过生成模板步骤")
        return "generate_unit_test"
    else:
        print(f"模板文件不存在: {template_path}，开始生成模板")
        return "generate_template"


def create_workflow():
    workflow = StateGraph[WorkflowState, None, WorkflowState, WorkflowState](WorkflowState)
    
    workflow.add_node("generate_template", generate_template)
    workflow.add_node("generate_unit_test", generate_unit_test)
    
    # 使用条件入口点代替固定入口点
    workflow.set_conditional_entry_point(
        check_template_existence,
        {
            "generate_template": "generate_template",
            "generate_unit_test": "generate_unit_test"
        }
    )
    
    workflow.add_edge("generate_template", "generate_unit_test")
    workflow.add_edge("generate_unit_test", END)
    
    return workflow.compile()
    

def get_available_operators():
    """从 input 目录动态获取可用的算子名称（基于 .jsonl 文件名）"""
    input_dir = _current_dir / "input"
    if not input_dir.exists():
        return []
    # 获取所有 .jsonl 文件的主文件名
    return sorted([f.stem for f in input_dir.glob("*.jsonl")])


def main():
    available_ops = get_available_operators()
    
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
        type=str,
        choices=available_ops,
        required=True,
        help=(
            "指定算子名称，将从 input/ 目录自动扫描可用算子。\n"
            "可选值: "
            f"{', '.join(available_ops)}"
        ),
    )
    parser.add_argument(
        "-t",
        "--op_type",
        dest="op_type",
        type=str,
        default="op_host",
        help="指定算子类型，默认为 op_host",
    )

    # 如果用户没有传任何参数，优先展示帮助信息，而不是报错格式
    if len(sys.argv) == 1:
        parser.print_help()
        return

    args = parser.parse_args()
    
    app = create_workflow()
    
    # 兼容 Enum 和 str，但这里 args.op_type 已经是 str
    initial_state = create_initial_state(args.operator_name, args.op_type)
    
    result = app.invoke(initial_state)
    
    print("\n" + "="*80)
    print("✅ Workflow 完成！")
    print("="*80)
    print(f"最终状态: {result.get('operator_name', 'N/A')} - {result.get('op_type', 'N/A')}")
    return result


if __name__ == "__main__":
    main()
