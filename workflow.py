#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
UTGen-V2 工作流脚本

将 template/ 目录中的模板文件与 input/ 目录中的 JSONL 测试数据合并,
生成完整的 C++ 单元测试文件到 outputs/ 目录。

命名规则:
  - input 文件: {op_name}.jsonl
  - template 文件: test_{op_name}_tiling.cpp
  - output 文件: test_{op_name}_tiling.cpp (在 outputs/ 目录)

用法:
  python workflow.py                    # 处理所有 input 文件
  python workflow.py -n all_gather_matmul  # 只处理指定算子
  python workflow.py --list             # 列出所有可用的算子
"""

import argparse
import sys
from pathlib import Path
from typing import List, Optional

# 项目根目录
PROJECT_ROOT = Path(__file__).parent.absolute()
INPUT_DIR = PROJECT_ROOT / "input"
TEMPLATE_DIR = PROJECT_ROOT / "template"
OUTPUT_DIR = PROJECT_ROOT / "outputs"
TARGET_DIR = PROJECT_ROOT / "target"  # 用于验证

# 导入核心生成逻辑
sys.path.insert(0, str(PROJECT_ROOT))
from nodes.generate_unit_test import generate_unit_test
import re


def remove_registry_check(content: str) -> str:
    """
    移除 IsOpImplRegistryAvailable 检查代码块。
    
    移除形如:
        if (!IsOpImplRegistryAvailable()) {
            GTEST_SKIP() << "Skip test: OpImplSpaceRegistryV2 is null on host.";
        }
    """
    # 匹配整个 if 块，包括可能的不同缩进
    pattern = r'\s*if\s*\(\s*!IsOpImplRegistryAvailable\(\)\s*\)\s*\{[^}]*\}\s*\n?'
    content = re.sub(pattern, '\n', content)
    return content


def get_available_operators() -> List[str]:
    """
    从 input 目录获取所有可用的算子名称。
    返回 JSONL 文件的主文件名列表。
    """
    if not INPUT_DIR.exists():
        return []
    return sorted([f.stem for f in INPUT_DIR.glob("*.jsonl")])


def get_matching_template(op_name: str) -> Optional[Path]:
    """
    根据算子名称找到对应的模板文件。
    
    命名规则: {op_name}.jsonl -> test_{op_name}_tiling.cpp
    """
    template_file = TEMPLATE_DIR / f"test_{op_name}_tiling.cpp"
    if template_file.exists():
        return template_file
    return None


def process_operator(op_name: str, verbose: bool = True) -> bool:
    """
    处理单个算子，生成对应的单元测试文件。
    
    Args:
        op_name: 算子名称 (如 "all_gather_matmul")
        verbose: 是否打印详细信息
    
    Returns:
        是否成功
    """
    # 构建路径
    input_path = INPUT_DIR / f"{op_name}.jsonl"
    template_path = get_matching_template(op_name)
    output_path = OUTPUT_DIR / f"test_{op_name}_tiling.cpp"
    
    # 检查文件存在性
    if not input_path.exists():
        print(f"❌ 输入文件不存在: {input_path}")
        return False
    
    if template_path is None:
        print(f"❌ 模板文件不存在: test_{op_name}_tiling.cpp")
        return False
    
    if verbose:
        print(f"📝 处理算子: {op_name}")
        print(f"   输入: {input_path}")
        print(f"   模板: {template_path}")
        print(f"   输出: {output_path}")
    
    # 构建状态字典 (兼容原有的 generate_unit_test 函数)
    state = {
        "operator_name": op_name,
        "op_type": "op_host",
        "input_path": str(input_path),
        "template_file_path": str(template_path),
        "output_path": str(output_path),
        "def_file_path": "",  # 不需要，模板已存在
    }
    
    try:
        # 确保输出目录存在
        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        
        # 调用核心生成逻辑
        result = generate_unit_test(state)
        
        # 后处理：移除 IsOpImplRegistryAvailable 检查
        output_path = Path(result['output_path'])
        if output_path.exists():
            content = output_path.read_text(encoding='utf-8')
            content = remove_registry_check(content)
            output_path.write_text(content, encoding='utf-8')
        
        if verbose:
            print(f"   ✅ 生成成功: {result['output_path']}")
        
        return True
    except Exception as e:
        print(f"   ❌ 生成失败: {e}")
        return False


def process_all_operators(verbose: bool = True) -> tuple:
    """
    处理所有可用的算子。
    
    Returns:
        (成功数, 失败数)
    """
    operators = get_available_operators()
    
    if not operators:
        print("⚠️  没有找到任何输入文件")
        return 0, 0
    
    print(f"\n🚀 开始处理 {len(operators)} 个算子...\n")
    print("=" * 60)
    
    success_count = 0
    fail_count = 0
    failed_ops = []
    
    for op_name in operators:
        if process_operator(op_name, verbose):
            success_count += 1
        else:
            fail_count += 1
            failed_ops.append(op_name)
        print()
    
    print("=" * 60)
    print(f"\n📊 处理完成: 成功 {success_count}, 失败 {fail_count}")
    
    if failed_ops:
        print(f"❌ 失败的算子: {', '.join(failed_ops)}")
    
    return success_count, fail_count


def list_operators():
    """列出所有可用的算子及其状态"""
    operators = get_available_operators()
    
    if not operators:
        print("⚠️  没有找到任何输入文件")
        return
    
    print(f"\n📋 可用算子列表 ({len(operators)} 个):\n")
    print(f"{'算子名称':<45} {'模板':<8} {'目标文件':<8}")
    print("-" * 65)
    
    for op_name in operators:
        template_exists = "✅" if get_matching_template(op_name) else "❌"
        target_exists = "✅" if (TARGET_DIR / f"test_{op_name}_tiling.cpp").exists() else "❌"
        print(f"{op_name:<45} {template_exists:<8} {target_exists:<8}")
    
    print()


def verify_output(op_name: str) -> bool:
    """
    验证生成的输出与目标文件是否一致（忽略空行差异）
    """
    output_path = OUTPUT_DIR / f"test_{op_name}_tiling.cpp"
    target_path = TARGET_DIR / f"test_{op_name}_tiling.cpp"
    
    if not output_path.exists():
        print(f"❌ 输出文件不存在: {output_path}")
        return False
    
    if not target_path.exists():
        print(f"⚠️  目标文件不存在: {target_path}")
        return False
    
    # 读取并规范化内容（移除空行进行比较）
    def normalize(content: str) -> List[str]:
        return [line.rstrip() for line in content.split('\n') if line.strip()]
    
    output_lines = normalize(output_path.read_text(encoding="utf-8"))
    target_lines = normalize(target_path.read_text(encoding="utf-8"))
    
    if output_lines == target_lines:
        return True
    
    # 如果不一致，打印差异
    print(f"❌ 输出与目标不一致: {op_name}")
    print(f"   输出行数: {len(output_lines)}, 目标行数: {len(target_lines)}")
    
    return False


def verify_all_outputs() -> tuple:
    """验证所有生成的输出"""
    operators = get_available_operators()
    
    print(f"\n🔍 验证 {len(operators)} 个算子的输出...\n")
    
    match_count = 0
    mismatch_count = 0
    
    for op_name in operators:
        output_path = OUTPUT_DIR / f"test_{op_name}_tiling.cpp"
        if output_path.exists():
            if verify_output(op_name):
                print(f"  ✅ {op_name}")
                match_count += 1
            else:
                mismatch_count += 1
        else:
            print(f"  ⏭️  {op_name} (未生成)")
    
    print(f"\n📊 验证完成: 匹配 {match_count}, 不匹配 {mismatch_count}")
    return match_count, mismatch_count


def main():
    parser = argparse.ArgumentParser(
        description="UTGen-V2 - 单元测试生成工作流",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  python workflow.py                       # 处理所有算子
  python workflow.py -n all_gather_matmul  # 只处理指定算子
  python workflow.py --list                # 列出所有可用算子
  python workflow.py --verify              # 验证生成结果与目标一致
        """
    )
    
    parser.add_argument(
        "-n", "--operator-name",
        dest="operator_name",
        type=str,
        default=None,
        help="指定要处理的算子名称。不指定则处理所有算子。"
    )
    
    parser.add_argument(
        "-t", "--op-type",
        dest="op_type",
        type=str,
        default="op_host",
        help="指定算子类型 (向后兼容参数，当前版本忽略此参数)"
    )
    
    parser.add_argument(
        "--list",
        action="store_true",
        help="列出所有可用的算子"
    )
    
    parser.add_argument(
        "--verify",
        action="store_true",
        help="验证生成的输出与目标文件是否一致"
    )
    
    parser.add_argument(
        "-q", "--quiet",
        action="store_true",
        help="静默模式，减少输出"
    )
    
    args = parser.parse_args()
    
    # 列出算子
    if args.list:
        list_operators()
        return
    
    # 验证输出
    if args.verify:
        verify_all_outputs()
        return
    
    # 处理算子
    if args.operator_name:
        # 处理单个算子
        available = get_available_operators()
        if args.operator_name not in available:
            print(f"❌ 未知的算子: {args.operator_name}")
            print(f"可用的算子: {', '.join(available)}")
            sys.exit(1)
        
        success = process_operator(args.operator_name, verbose=not args.quiet)
        sys.exit(0 if success else 1)
    else:
        # 处理所有算子
        success, fail = process_all_operators(verbose=not args.quiet)
        sys.exit(0 if fail == 0 else 1)


if __name__ == "__main__":
    main()
