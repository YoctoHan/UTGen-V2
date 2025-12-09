#!/bin/bash

# 获取脚本所在目录的绝对路径
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
WORKFLOW_PY="$SCRIPT_DIR/workflow.py"
INPUT_DIR="$SCRIPT_DIR/input"

# 初始化计数器
total=0
success=0
fail=0
failed_ops=()

echo "========================================================"
echo "UTGen-V2 批量执行工具"
echo "Work Dir: $SCRIPT_DIR"
echo "========================================================"

# 检查 input 目录
if [ ! -d "$INPUT_DIR" ]; then
    echo "Error: input directory not found at $INPUT_DIR"
    exit 1
fi

# 遍历 input 目录下的所有 .jsonl 文件
# sort 确保执行顺序一致
for file in $(ls "$INPUT_DIR"/*.jsonl | sort); do
    # 检查文件是否存在（处理空目录情况）
    [ -e "$file" ] || continue

    # 获取不带路径和扩展名的文件名作为算子名称
    filename=$(basename -- "$file")
    op_name="${filename%.*}"

    ((total++))

    echo ""
    echo "--------------------------------------------------------"
    echo "[$total] 正在处理算子: $op_name"
    echo "CMD: python3 workflow.py -n $op_name -t op_host"
    echo "--------------------------------------------------------"

    # 执行 workflow.py
    python3 "$WORKFLOW_PY" -n "$op_name" -t op_host

    # 检查执行结果
    if [ $? -eq 0 ]; then
        ((success++))
        echo "✅ 算子 $op_name 执行成功"
    else
        ((fail++))
        failed_ops+=("$op_name")
        echo "❌ 算子 $op_name 执行失败"
    fi
done

echo ""
echo "========================================================"
echo "执行结果汇总"
echo "========================================================"
echo "总计执行: $total"
echo "成功数量: $success"
echo "失败数量: $fail"

if [ $fail -gt 0 ]; then
    echo "--------------------------------------------------------"
    echo "以下算子执行失败:"
    for op in "${failed_ops[@]}"; do
        echo " - $op"
    done
    echo "--------------------------------------------------------"
    exit 1
else
    echo "🎉 所有算子均执行成功！"
    exit 0
fi

