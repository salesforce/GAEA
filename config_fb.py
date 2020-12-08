import rl.graph_edit_rl as grl
import rl.graph_includes as graph_inc
import numpy.random as npr
import numpy as np
import os
import glob
from paths_inc import *
import joblib as jl
import scipy.io as sio
from config_global import *

#inds = [13, 50, 26, 55, 58,  1,  7, 24]
# files[38] -- MIT


def add_fb_config(config, file):

    d = sio.loadmat(file)
    config["particles MC"] = [{"particles":1000}, {"particles":1000}]
    config["group names"] = ["female", "male"]
    #config["budget fraction"] = 0.01
    #config["budget"] = int(np.floor(np.sum(d["A"].nnz)*config["budget fraction"]))
    config["budget"] = 500
    config["use_same_flow_matrix"] = True
    # when communities share the same flow matrix                                                           #iterations in the graph editing loop
    config["immunized nodes"] = {}

    #config["immunized nodes"] = {"lambda:degree": {"ascending":True, "fn": graph_inc.degree_scaled, "fn params":{}, "count": 2}}                                              #immunized node selection (e.g. 5 random)
    config["graph fn"] = graph_inc.get_graph                                               #graph generator function
    config["graph params"] = {"model": "fb", "im": config["immunized nodes"],
                              "params": {"d": d, "definition":"upper bias", "samples":3}}          #graph generator params, takes function family and immunization params
                                                                                                # Barabasi Albert preferential attachment graph
                                                                                                # x,y are node degrees of incident nodes, red weights wrt x*y, black wrt 1/(x*y)
    config["states fn"] = "init_node_states"                                               #initial state creation-untrained
    config["states params"] = {}                                                           #idk we might need params
    config["graph edit fn"] = "get_graph_edit_model"                                       # graph edit model
    config["graph edit params"] = {}
    config["particle fn"] = "init_particle_locs_fb"                                    #initial particle locations (e.g. random)
    config["particle params"] = [{"d": d, "pop_key":1,"particles":config["particles"]},  #female
                                 {"d": d, "pop_key":2, "particles":config["particles"]}] #male


    #config["particle params r"] ={"df_nodes": dff_nodes, "df_tract":dff_tracts, "pop_key":"white"}
    #config["particle params b"] ={"df_nodes": dff_nodes, "df_tract":dff_tracts, "pop_key":"black"}   #params for this function
    #config["state train fn"] = "init_states_basic"                                         #training the initial states
    config["state train params"] = {}
    config["model_type"] = ['bootstrapped', 'fullydifferntiable'][model_type_idx]                                                        #iterations in the graph editing loop

    config["T"] = 3
    config['temp_decay_factor']=0.995
    config['lagrangian_lamdba_growth']=10
    
    config["constraint_coeff_mu"] = [10000, 1e-14]
    config["constraint_epoch_start_schedule"] = [0, 400, 600]
    config["epochs"] = [3,1500][model_type_idx]
    config["batch_size"] = 25
    config["gamma"] = 0.7 
    config['fair_loss_error'] = ['mae','mae2','mae3','mape','rmape','smape','zscore'][0]
    config['var_coeff'] = 0.


    return config



