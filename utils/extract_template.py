#!/usr/bin/env python3
"""
从 target 目录的完整测试文件中提取模板骨架。
移除 COMPILE_INFO 常量定义和 cases_params 数组，保留代码框架。

用法:
    python extract_template.py <target_file> <output_template>
    python extract_template.py --all  # 处理所有文件
"""

import re
import sys
from pathlib import Path


def extract_template_from_target(target_content: str) -> str:
    """
    从目标文件内容中提取模板骨架：
    1. 移除 const ... COMPILE_INFO = ...; 定义
    2. 移除 StructName cases_params[] = {...}; 定义
    3. 移除 "// 用例列表集" 开头的注释（这些会被生成逻辑重新添加）
    4. 保留其他代码
    """
    lines = target_content.split('\n')
    result_lines = []
    i = 0
    
    while i < len(lines):
        line = lines[i]
        
        # 检查是否是 COMPILE_INFO 定义（可能是 const string 或 const std::string）
        if re.match(r'^const\s+(std::)?string\s+COMPILE_INFO\s*=', line):
            # 跳过 COMPILE_INFO 定义行
            # 如果是多行定义，继续跳过直到分号结束
            while i < len(lines) and not lines[i].rstrip().endswith(';'):
                i += 1
            i += 1  # 跳过最后一行（包含分号）
            # 跳过 COMPILE_INFO 后的空行
            while i < len(lines) and lines[i].strip() == '':
                i += 1
            continue
        
        # 检查是否是 "// 用例列表集" 开头的注释（这些会被生成逻辑重新添加）
        if line.strip().startswith('// 用例列表集') or line.strip().startswith('// 用例参数列表'):
            # 跳过这个注释及其后续相关注释
            while i < len(lines) and lines[i].strip().startswith('//'):
                i += 1
            continue
        
        # 检查是否是 cases_params 数组定义
        if re.match(r'^([\w:]+)\s+cases_params\s*\[\s*\]\s*=\s*\{', line):
            # 跳过整个数组定义，直到找到 };
            brace_count = line.count('{') - line.count('}')
            while i < len(lines):
                if brace_count == 0 and lines[i].rstrip().endswith('};'):
                    i += 1
                    break
                i += 1
                if i < len(lines):
                    brace_count += lines[i].count('{') - lines[i].count('}')
            # 跳过数组定义后的空行
            while i < len(lines) and lines[i].strip() == '':
                i += 1
            continue
        
        result_lines.append(line)
        i += 1
    
    result = '\n'.join(result_lines)
    
    # 确保在 TEST_P 前有恰好 3 个换行符（即 2 个空行）
    # 这样与模板格式一致
    result = re.sub(r'\n+(TEST_P\()', r'\n\n\n\1', result)
    
    return result


def process_file(target_path: Path, template_path: Path) -> bool:
    """处理单个文件"""
    if not target_path.exists():
        print(f"目标文件不存在: {target_path}")
        return False
    
    target_content = target_path.read_text(encoding='utf-8')
    template_content = extract_template_from_target(target_content)
    
    template_path.parent.mkdir(parents=True, exist_ok=True)
    template_path.write_text(template_content, encoding='utf-8')
    
    print(f"已生成模板: {template_path}")
    return True


def process_all():
    """处理所有目标文件"""
    project_root = Path(__file__).parent.parent
    target_dir = project_root / "target"
    template_dir = project_root / "template"
    
    if not target_dir.exists():
        print(f"目标目录不存在: {target_dir}")
        return
    
    for target_file in sorted(target_dir.glob("*.cpp")):
        template_file = template_dir / target_file.name
        process_file(target_file, template_file)


def main():
    if len(sys.argv) == 2 and sys.argv[1] == "--all":
        process_all()
    elif len(sys.argv) == 3:
        target_path = Path(sys.argv[1]).resolve()
        template_path = Path(sys.argv[2]).resolve()
        process_file(target_path, template_path)
    else:
        print(__doc__)
        sys.exit(1)


if __name__ == "__main__":
    main()
