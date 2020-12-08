import os

print(os.path.dirname(os.path.realpath(__file__)))

proj_root = os.path.dirname(os.path.realpath(__file__))
results_root = os.path.join(proj_root, "graph_rl_results")