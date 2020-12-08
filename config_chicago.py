import rl.graph_edit_rl as grl
import rl.graph_includes as graph_inc
import numpy.random as npr
import numpy as np
import os
from paths_inc import *
import joblib as jl
from config_global import *
import geopandas as gpd
import collections as c
import shapely
import fiona



def add_chicago_config(config, select="full"):
    bb ={"full":shapely.geometry.box(-90, 39, -85, 45),
         "south-large": shapely.geometry.box(-87.762800, 41.945000, -87.506549, 41.742000),
         "south": shapely.geometry.box(-87.762800, 41.632756, -87.506549, 41.742000),
         "north": shapely.geometry.box(-87.922857, 41.945000, -87.506549, 42.039490),
         "north-large": shapely.geometry.box(-87.922857, 41.800000, -87.506549, 42.039490)}

    config["group names"] = ["white", "black", "latino"]
    config["batch_size"] = 30
    config["budget"] = 50
    config["budget fraction"] = 0.1
    data_path = os.path.join(proj_root ,"data" ,"networks")
    out_nodes = os.path.join(data_path, "chicago_nodes_new.gz")
    out_edges = os.path.join(data_path, "chicago_edges_new.gz")
    out_tracts = os.path.join(data_path, "chicago_tracts_new.gz")
    config["use_same_flow_matrix"] = True
    dff_nodes, index = graph_inc.filter_geo_data(jl.load(out_nodes), bb[select], get_index=True)
    dff_edges = graph_inc.map_edges(graph_inc.filter_geo_data(jl.load(out_edges), bb[select]), index)
    dff_tracts = graph_inc.filter_geo_data(jl.load(out_tracts), bb[select], reindex=False)
    config["budget"] = int(np.floor(len(dff_edges)*config["budget fraction"]))


    config["chicago-tag"] = "all"
    config["batch size state"] = 50  # batch size for initial state  training
    config["batch size train"] = 50
    # when communities share the same flow matrix                                                           #iterations in the graph editing loop
    config["immunized nodes"] = {"geo": {"df_nodes": dff_nodes, "count": np.inf}, "large mask": True}

    #config["immunized nodes"] = {"lambda:degree": {"ascending":True, "fn": graph_inc.degree_scaled, "fn params":{}, "count": 2}}                                              #immunized node selection (e.g. 5 random)
    config["graph fn"] = graph_inc.get_graph                                               #graph generator function
    config["graph params"] = {"model": "geo", "im": config["immunized nodes"], "mask_params": {"order": 2},
                              "params": {"df_edges": dff_edges, "weight_fn": graph_inc.weight_identity, "fn_edges":[graph_inc.geo_edge_map, graph_inc.geo_edge_map, graph_inc.geo_edge_map]}}          #graph generator params, takes function family and immunization params
                                                                                                # Barabasi Albert preferential attachment graph
                                                                                                # x,y are node degrees of incident nodes, red weights wrt x*y, black wrt 1/(x*y)

    config["states fn"] = "init_node_states"                                               #initial state creation-untrained
    config["states params"] = {}                                                           #idk we might need params
    config["graph edit fn"] = "get_graph_edit_model"                                       # graph edit model
    config["graph edit params"] = {}
    config["particle fn"] = "init_particle_locs_geo"                                    #initial particle locations (e.g. random)

    config["particle params"] = [{"df_nodes": dff_nodes, "df_tract":dff_tracts, "pop_key":"white", "particles":50000},
                                 {"df_nodes": dff_nodes, "df_tract":dff_tracts, "pop_key":"black", "particles":50000},
                                 {"df_nodes": dff_nodes, "df_tract":dff_tracts, "pop_key":"latino", "particles":50000}]
    #config["particle params r"] ={"df_nodes": dff_nodes, "df_tract":dff_tracts, "pop_key":"white"}
    #config["particle params b"] ={"df_nodes": dff_nodes, "df_tract":dff_tracts, "pop_key":"black"}   #params for this function
    #config["state train fn"] = "init_states_basic"                                         #training the initial states
    config["state train params"] = {}
    config["model_type"] = ['bootstrapped', 'fullydifferntiable'][model_type_idx]                                                        #iterations in the graph editing loop
    
    config["particles"] = 50000
    config["constraint_coeff_mu"] = [0.1, 1e-06]
    config["constraint_epoch_start_schedule"] = [0, 0, 200]

    config["particles MC"] = [{"df_nodes": dff_nodes, "df_tract":dff_tracts, "pop_key":"white", "particles":2000},
                                 {"df_nodes": dff_nodes, "df_tract":dff_tracts, "pop_key":"black", "particles":2000},
                                 {"df_nodes": dff_nodes, "df_tract":dff_tracts, "pop_key":"latino", "particles":2000}]
    #config["particle params r"] ={"df_nodes": dff_nodes, "df_tract":dff_tracts, "pop_key":"white"}
    #config["particle params b"] ={"df_nodes": dff_nodes, "df_tract":dff_tracts, "pop_key":"black"}   #params for this function
    #config["state train fn"] = "init_states_basic"                                         #training the initial states
    config["state train params"] = {}
    config["model_type"] = ['bootstrapped', 'fullydifferntiable'][model_type_idx]                                                        #iterations in the graph editing loop
    config["particles"] = 10000
    config["constraint_coeff_mu"] = [1, 1e-06]
    config["constraint_epoch_start_schedule"] = [0, 100, 200]
    
    config['lagrangian_lamdba_growth']=0.0001
    config['lagrangian_lamdba2_growth']=0.01
    
    config["epochs"] = [3,300][model_type_idx]
    
    return config


