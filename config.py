from enum import Enum

class OperatorName(str, Enum):
    MATMUL_ALL_REDUCE = "matmul_all_reduce"
    ALL_GATHER_MATMUL = "all_gather_matmul"
    DISTRIBUTE_BARRIER = "distribute_barrier"


class OpType(str, Enum):
    OP_HOST = "op_host"


OPS_TRANSFORMERS_DIR = "/workspace/ops-transformer-dev"