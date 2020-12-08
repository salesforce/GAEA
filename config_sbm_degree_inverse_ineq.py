import rl.graph_edit_rl as grl
import rl.graph_includes as graph_inc
import numpy.random as npr
import numpy as np
from config_global import *


config["N"] = 200
n2 = int(config["N"]/2)
config["sbm-p"]  = [[0.1, 0.01], [0.01, 0.1]]
config["sizes"] = [n2, n2]
config["use_same_flow_matrix"] = True													# when communities share the same flow matrix
config["immunized nodes"] = {"lambda:degree": {"ascending":True, "fn": graph_inc.degree_scaled, "fn params":{}, "count": 3, "subset":range(n2)}}
config["graph fn"] = graph_inc.get_graph                                               #graph generator function
config["graph params"] = {"model": "sbm", "im": config["immunized nodes"],
                          "params": {"sizes": config["sizes"],
                                     "p": config["sbm-p"],
                                     "fn_edges":[graph_inc.uniform_weight,
                                                 graph_inc.uniform_weight]}}          #graph generator params, takes function family and immunization params
                                                                                            # Barabasi Albert preferential attachment graph
                                                                                            # x,y are node degrees of incident nodes, red weights wrt x*y, black wrt 1/(x*y)

config["states fn"] = "init_node_states"                                               #initial state creation-untrained
config["states params"] = {}                                                           #idk we might need params
config["graph edit fn"] = "get_graph_edit_model"                                       # graph edit model
config["graph edit params"] = {}
                                 #initial particle locations (e.g. random)
config["particle params"] = [{"node_range": range(0, n2)}, {"node_range": range(n2, config["N"])}]

config["particles MC"] = [{"node_range": range(0, n2), "particles":10000}, {"node_range": range(n2, config["N"]), "particles":10000}]

config["state train fn"] = "init_states_basic"                                         #training the initial states
config["state train params"] = {"epochs":100}
config["steps_per_epoch"] = 10

# "fn_edges": [lambda G, x, y, sizes, p: p[np.digitize(x, [0] + list(np.cumsum(sizes))) - 1][
#     np.digitize(y, [0] + list(np.cumsum(sizes))) - 1],
#              lambda G, x, y, sizes, p: p[np.digitize(x, [0] + list(np.cumsum(sizes))) - 1][
#                  np.digitize(y, [0] + list(np.cumsum(sizes))) - 1]]}}
