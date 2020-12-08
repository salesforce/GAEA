import rl.graph_edit_rl as grl
import rl.graph_includes as graph_inc
import numpy.random as npr
import numpy as np
from config_global import *

config["N"] = 400
config["cl-exp"] = 2.5
                                           #immunized node selection (e.g. 5 random)
config["graph fn"] = graph_inc.get_graph                                               #graph generator function
config["immunized nodes"] = {"lambda:degree": {"ascending":True, "fn": graph_inc.degree_scaled, "fn params":{}, "count": 3}}
config["graph params"] = {"model": "cl", "im": config["immunized nodes"],
                          "params": {"N":config["N"], "exp_val":config["cl-exp"],
                                      "fn_edges":[graph_inc.weight_inv_degree, graph_inc.weight_degree]}}         #graph generator params, takes function family and immunization params
                                                                                            # Barabasi Albert preferential attachment graph
                                                                                          # x,y are node degrees of incident nodes, red weights wrt x*y, black wrt 1/(x*y)
config["states fn"] = "init_node_states"                                               #initial state creation-untrained
config["states params"] = {}                                                           #idk we might need params
config["graph edit fn"] = "get_graph_edit_model"                                       # graph edit model
config["graph edit params"] = {}
                                                   #params for this function
config["state train fn"] = "init_states_basic"                                         #training the initial states
config["state train params"] = {}







