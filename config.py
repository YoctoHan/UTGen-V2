from enum import Enum

class OperatorName(str, Enum):
    MATMUL_ALL_REDUCE = "matmul_all_reduce"
    ALL_GATHER_MATMUL = "all_gather_matmul"
    DISTRIBUTE_BARRIER = "distribute_barrier"
    ALL_GATHER_MATMUL_V2 = "all_gather_matmul_v2"
    ALL_TO_ALL_ALL_GATHER_BATCH_MAT_MUL = "all_to_all_all_gather_batch_mat_mul"
    BATCH_MAT_MUL_REDUCE_SCATTER_ALLTO_ALL = "batch_mat_mul_reduce_scatter_allto_all"
    GROUPED_MAT_MUL_ALL_REDUCE = "grouped_mat_mul_all_reduce"
    MATMUL_REDUCE_SCATTER = "matmul_reduce_scatter"
    MATMUL_REDUCE_SCATTER_V2 = "matmul_reduce_scatter_v2"
    MOE_DISTRIBUTE_COMBINE = "moe_distribute_combine"
    MOE_DISTRIBUTE_COMBINE_ADD_RMS_NORM = "moe_distribute_combine_add_rms_norm"
    MOE_DISTRIBUTE_COMBINE_V2 = "moe_distribute_combine_v2"
    MOE_DISTRIBUTE_DISPATCH = "moe_distribute_dispatch"
    MOE_DISTRIBUTE_DISPATCH_V2 = "moe_distribute_dispatch_v2"


class OpType(str, Enum):
    OP_HOST = "op_host"


OPS_TRANSFORMERS_DIR = "/workspace/ops-transformer-dev"
