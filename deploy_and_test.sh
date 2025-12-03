#!/bin/bash

# 脚本功能：将 target 目录下的测试文件复制到正确位置，编译并执行测试
# 使用方法: ./deploy_and_test.sh [--build-only] [--test-only]

set -e

# 配置路径
UTGEN_TARGET_DIR="/workspace/UTGen-V2/target"
OPS_TRANSFORMER_DIR="/workspace/ops-transformer-dev"
MC2_DIR="${OPS_TRANSFORMER_DIR}/mc2"

# 颜色输出
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# 日志函数
log_info() {
    echo -e "${GREEN}[INFO]${NC} $1"
}

log_warn() {
    echo -e "${YELLOW}[WARN]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# 从文件名提取算子名称
# 例如: test_all_gather_matmul_tiling.cpp -> all_gather_matmul
extract_op_name() {
    local filename="$1"
    # 移除 test_ 前缀和 _tiling.cpp 后缀
    local op_name="${filename#test_}"
    op_name="${op_name%_tiling.cpp}"
    echo "$op_name"
}

# 复制测试文件到目标位置
deploy_test_files() {
    log_info "开始部署测试文件..."
    
    local deployed_ops=()
    
    # 遍历 target 目录下的所有 .cpp 文件
    for file in "${UTGEN_TARGET_DIR}"/*.cpp; do
        if [[ ! -f "$file" ]]; then
            log_warn "没有找到 .cpp 文件"
            continue
        fi
        
        local filename=$(basename "$file")
        local op_name=$(extract_op_name "$filename")
        local target_dir="${MC2_DIR}/${op_name}/tests/ut/op_host"
        
        # 检查目标目录是否存在
        if [[ ! -d "$target_dir" ]]; then
            log_warn "目标目录不存在: $target_dir，尝试创建..."
            mkdir -p "$target_dir"
        fi
        
        log_info "复制 $filename -> $target_dir/"
        cp "$file" "$target_dir/"
        
        deployed_ops+=("$op_name")
    done
    
    # 导出已部署的算子列表供后续使用
    DEPLOYED_OPS=("${deployed_ops[@]}")
    
    if [[ ${#DEPLOYED_OPS[@]} -eq 0 ]]; then
        log_error "没有部署任何测试文件"
        exit 1
    fi
    
    log_info "成功部署 ${#DEPLOYED_OPS[@]} 个测试文件"
    echo "部署的算子: ${DEPLOYED_OPS[*]}"
}

# 编译指定的算子
build_ops() {
    log_info "开始编译..."
    
    cd "${OPS_TRANSFORMER_DIR}"
    
    # 构建 ops 参数，用逗号分隔
    local ops_list=$(IFS=','; echo "${DEPLOYED_OPS[*]}")
    
    log_info "执行编译命令: bash build.sh -u --ophost --ops=\"${ops_list}\" --noexec"
    
    bash build.sh -u --ophost --ops="${ops_list}" --noexec
    
    log_info "编译完成"
}

# 执行测试
run_tests() {
    log_info "开始执行测试..."
    
    cd "${OPS_TRANSFORMER_DIR}"
    
    # 设置 BUILD_PATH 环境变量（测试框架需要）
    export BUILD_PATH="${OPS_TRANSFORMER_DIR}/build"
    log_info "设置 BUILD_PATH=${BUILD_PATH}"
    
    # 查找 transformer_op_host_ut 可执行文件
    local ut_binary=$(find . -name "transformer_op_host_ut" -type f -executable 2>/dev/null | head -1)
    
    if [[ -z "$ut_binary" ]]; then
        # 尝试常见路径
        if [[ -f "./build/bin/transformer_op_host_ut" ]]; then
            ut_binary="./build/bin/transformer_op_host_ut"
        elif [[ -f "./output/bin/transformer_op_host_ut" ]]; then
            ut_binary="./output/bin/transformer_op_host_ut"
        elif [[ -f "./build/tests/ut/framework_normal/op_host/transformer_op_host_ut" ]]; then
            ut_binary="./build/tests/ut/framework_normal/op_host/transformer_op_host_ut"
        else
            log_error "找不到 transformer_op_host_ut 可执行文件"
            exit 1
        fi
    fi
    
    log_info "执行测试: $ut_binary --gtest_filter='*Tiling*:-*InferShape*'"
    "$ut_binary" --gtest_filter='*Tiling*:-*InferShape*'
    
    log_info "测试完成"
}

# 显示帮助信息
show_help() {
    echo "用法: $0 [选项]"
    echo ""
    echo "选项:"
    echo "  --deploy-only    仅部署文件，不编译和测试"
    echo "  --build-only     仅部署和编译，不执行测试"
    echo "  --test-only      仅执行测试（假设文件已部署和编译）"
    echo "  -h, --help       显示此帮助信息"
    echo ""
    echo "默认行为: 部署 -> 编译 -> 测试"
}

# 主函数
main() {
    local do_deploy=true
    local do_build=true
    local do_test=true
    
    # 解析参数
    while [[ $# -gt 0 ]]; do
        case "$1" in
            --deploy-only)
                do_build=false
                do_test=false
                shift
                ;;
            --build-only)
                do_test=false
                shift
                ;;
            --test-only)
                do_deploy=false
                do_build=false
                shift
                ;;
            -h|--help)
                show_help
                exit 0
                ;;
            *)
                log_error "未知参数: $1"
                show_help
                exit 1
                ;;
        esac
    done
    
    log_info "============================================"
    log_info "UTGen 测试部署和执行脚本"
    log_info "============================================"
    
    if $do_deploy; then
        deploy_test_files
    else
        # 如果不部署，需要从现有文件获取算子列表
        DEPLOYED_OPS=()
        for file in "${UTGEN_TARGET_DIR}"/*.cpp; do
            if [[ -f "$file" ]]; then
                local filename=$(basename "$file")
                local op_name=$(extract_op_name "$filename")
                DEPLOYED_OPS+=("$op_name")
            fi
        done
    fi
    
    if $do_build; then
        build_ops
    fi
    
    if $do_test; then
        run_tests
    fi
    
    log_info "============================================"
    log_info "全部完成!"
    log_info "============================================"
}

# 运行主函数
main "$@"

