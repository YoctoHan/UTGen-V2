python workflow.py -n all_gather_matmul -t op_host
python workflow.py -n matmul_all_reduce -t op_host
python workflow.py -n distribute_barrier -t op_host