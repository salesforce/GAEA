import argparse

import gc
import time
import collections as c
import sys
import glob
import os
import tensorflow as tf
import rl.graph_includes as graph_inc
import datetime
import pandas as pd
import numpy as np
from  scipy import stats
from collections import defaultdict
from keras import backend as K
import joblib as jl
import itertools as it
import matplotlib.pyplot as plt
import warnings
from numba import cuda
import plotting as rlp
import scipy.io as sio
from tensorflow.compat.v1 import ConfigProto
from tensorflow.compat.v1 import InteractiveSession
warnings.filterwarnings("ignore")

def reset_keras(device=0):
    cuda.select_device(device)
    cuda.close()
    print(gc.collect()) # if it's done something you should see a number being outputted

    K.clear_session()
    sess = K.get_session()
    sess.close()
    # use the same config as you used to create the session
    config = tf.ConfigProto()
    config.gpu_options.per_process_gpu_memory_fraction = 1
    config.gpu_options.visible_device_list = "0"
    K.set_session(tf.Session(config=config))

def set_device(gpu):
    os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID";
    os.environ["CUDA_VISIBLE_DEVICES"]=str(gpu);

def init_keras(device=0):
    set_device(device)
    config = tf.ConfigProto()
    config.gpu_options.allow_growth=True
    if device:
        K.set_session(tf.Session(
            config=tf.ConfigProto(intra_op_parallelism_threads = 14, inter_op_parallelism_threads = 14)))
    else:
        sess = tf.Session(config=config)
        K.set_session(sess)
    
if __name__ == "__main__": 
    # set it up
    parser = argparse.ArgumentParser(description='Run both baseline and the RL method')
    parser.add_argument("--exp", type=str, default="edit", help="experiment to run:edit and facility_placement")
    parser.add_argument("--graph", type=str, default="fb", help="graph to run experiments on:synthetic,chicago and fb")
    parser.add_argument("--school", type=str, default="Caltech36", help="Applicable only when graph:fb .Select the " \
                        + "school name:Caltech36,Mich67,Reed98")
    
    args = parser.parse_args()
    run_type = args.graph
    experiment_type = args.exp
    school = args.school
    
#     run_type = {
#     0:"synthetic",
#     1:'chicago',
#     2:'fb', 
#     3:"ba-degree-ineq",
#     4:'ba-degree-inverse-ineq',
#     5:'cl-degree-ineq',
#     6:"cl-degree-inverse-ineq" , 
#     7:"sbm-degree-ineq", 
#     8:"sbm-degree-inverse-ineq", 
#     9:"er-degree-ineq", 
#     10:"er-degree-inverse-ineq"}[2]

    #experiment_type = "edit"
    debug = ["build",'train'][1]
    file_tag = "secondorder-fixed"
    fb_minmax = [0, 3000]

    init_keras(0)
    
    from paths_inc import *
    #
    print('run_type:',run_type)
    #methods = ["sbm-degree-ineq"]
    if run_type == "synthetic":
        methods = ["ba-degree-ineq", "ba-degree-inverse-ineq", "sbm-degree-ineq", "sbm-degree-inverse-ineq","cl-degree-ineq", "cl-degree-inverse-ineq" , "er-degree-ineq", "er-degree-inverse-ineq"]
    elif run_type  == "chicago":
        methods = ["south-large"]
    elif run_type == "fb":
        methods = [(os.path.basename(os.path.splitext(f)[0]), f) for f in glob.glob(os.path.join(proj_root, "data", "facebook100","") + "*.mat")]
        #a = ["Mich67"]
        #methods = [method for method in methods if method[0] in a]
    
        ds = [sio.loadmat(method[1]) for method in methods]
        methods = [method for d, method in zip(ds, methods) if d["A"].shape[0] <= fb_minmax[1] and d["A"].shape[0] >= fb_minmax[0] and
                   np.max([np.sum(d["local_info"][:, 1] == 1)/len(d["local_info"][:, 1]),
                           np.sum(d["local_info"][:, 1] == 2)/len(d["local_info"][:, 1])]) < .80]
        
        #a = ["MIT8", "UChicago30", "Carnegie49", "Vermont70", "Caltech36", "Yale4","Columbia2", "Dartmouth6"]
        a = [school] #["Columbia2", "Dartmouth6"]
        methods = [method for method in methods if method[0] in a]
        print('methods:',methods)
    else:
        methods = [run_type]
        

    r = c.defaultdict(dict)
    
    for method in methods:
        if method == "kregular-inverse-degree-ineq":   # top-1 degree node reward on BA graph
            from config_kregular_degree_ineq import *
        if method == "kregular-degree-ineq":   # top-1 degree node reward on BA graph
            from config_kregular_degree_ineq import *
        elif method == "ba-degree-ineq":   # top-1 degree node reward on BA graph
            from config_ba_degree_ineq import *
        elif method == "cl-degree-ineq":   # top-1 degree node reward on BA graph
            from config_cl_degree_ineq import *
        elif method == "er-degree-ineq":   # top-1 degree node reward on BA graph
            from config_er_degree_ineq import *
        elif method == "karate-degree-ineq":
            from config_karate_degree_ineq import *
        elif method == "sbm-degree-ineq":
            from config_sbm_degree_ineq import *
        elif method == "cl-degree-inverse-ineq":   # top-1 degree node reward on BA graph
            from config_cl_degree_inverse_ineq import *
        elif method == "ba-degree-inverse-ineq": #bottom sampled nodes (k=1) by degree, BA graph
            from config_ba_degree_inverse_ineq import *
        elif method == "er-degree-inverse-ineq": #bottom sampled nodes (k=1) by degree, BA graph
            from config_er_degree_inverse_ineq import *
        elif method == "sbm-degree-inverse-ineq": #bottom sampled nodes (k=1) by degree, BA graph
            from config_sbm_degree_inverse_ineq import *
        elif run_type == "chicago":
            from config_chicago import *
            config = add_chicago_config(config, method)
        elif run_type == "fb":
            from config_fb import *
            config = add_fb_config(config, method[1])
    
        config["no training"] = False
        config["evaluation iters"] = 1
        config["sample graphs iter"] = False
        config["param schedule"] = False
        config["evaluation baseline"] = True
    
        if experiment_type == "facility_placement":
            import rl.facility_placement_rl as grl
            from config_facility_placement import *
        else:
            import rl.graph_edit_rl as grl
    
    
        eval_log = c.defaultdict(list)
        start_time = '{:%Y%m%d%H%M%S}'.format(datetime.datetime.now())
        if "build" in debug:
            model = grl.FlowGraphEditRL(config)
            break
        model = grl.FlowGraphEditRL(config)  # build graph and model
        if "train" in debug:
            _, history = model.train()
            net_gs = model.net_gs[0]
            net_gs_trained = model.net_gs_trained
            immunized = model.immunized_ids
            locs = model.locs
            mask = model.mask
            budget = model.budget
            #reset_keras(model)
            model = None
    
            reset_keras()
            #gc.collect()
            #K.clear_session()
    
            jl.dump({"history": history.history, "config": config,
                     "graph before": net_gs, "graph trained": net_gs_trained},
                    os.path.join(results_root, str(run_type) + '_' +  str(method[0]) + '_' +file_tag+'_'+ start_time + ".gz"))
    
            paths_orig = graph_inc.shortest_paths(net_gs, immunized, locs)
            paths_train = graph_inc.shortest_paths(net_gs_trained, immunized, locs)
            paths_orig_flat = np.array([i for i in list(it.chain(*paths_orig)) if not np.isnan(i)])
            paths_train_flat = np.array([i for i in list(it.chain(*paths_train)) if not np.isnan(i)])
    
            gini_orig = graph_inc.gini(np.max(paths_orig_flat) - paths_orig_flat)
            gini_train = graph_inc.gini(np.max(paths_train_flat) - paths_train_flat)
    
            means_orig = {k:v for v,k in zip([np.nanmean(r) for r in paths_orig], ["white", "black", "latino"])}
            means_train = {k:v for v,k in zip([np.nanmean(r) for r in paths_train], ["white", "black", "latino"])}
            edits = len(np.where(np.array(net_gs_trained >= 0.5).astype(np.int) - net_gs.astype(np.bool).astype(int))[0])
    
            gs, ret = graph_inc.do_edge_add_heuristic(net_gs,immunized, locs, k=edits, edge_k=25,A=mask)
            paths_baseline = graph_inc.shortest_paths(gs, immunized, locs)
            paths_baseline_flat = np.array([i for i in list(it.chain(*paths_baseline)) if not np.isnan(i)])
            gini_baseline = graph_inc.gini(np.max(paths_baseline_flat) - paths_baseline_flat)
            means_baseline = {k: v for v, k in zip([np.nanmean(r) for r in paths_baseline], ["white", "black", "latino"])}
    
            r = {"edits":{"budget":budget, "edit": edits},
                 "gini":{"original": gini_orig, "edit":gini_train,
                         "baseline": gini_baseline},
                 "mean": {(method, "original"): means_orig,
                          (method, "edit"): means_train,
                          (method, "baseline"): means_baseline}}
            print(r)
            jl.dump({"history": history.history, "config": config,
                     "graph before": net_gs, "graph trained": net_gs_trained, "results":r},
                    os.path.join(results_root, str(run_type) + '_' +  str(method[0]) + '_' +file_tag+'_'+ start_time + ".gz"))
    
        # if not debug:
        #     # for ttt in range(config["T"],config["T_max"]):
        #     #     setattr(model, "T", ttt)
        #     #     ret_tune = model.significance_replicates(args={"replicates": config["evaluation replicates"]})
        #     #     v1 = np.mean(ret_tune["measure r"])
        #     #     v2 =  np.mean(ret_tune["measure b"])
        #     #     print("Finding episode length: " + str(ttt)+ ", " + str((v1,v2,config["flow th"])))
        #     #     if v1 > config["flow th"] or v2 > config["flow th"]:
        #     #         break
        #     for i in range(config["evaluation iters"]):  # number of unique graphs
        #         if model is None or config["sample graphs iter"]:
        #             model = grl.FlowGraphEditRL(config) #build graph and model
        #         else:
        #             model.reset()
        #         if config["param schedule"]:  #this updates parameters with a list of param dictionaries of length config["evaluation iters"], e.g. changing budget
        #             for k,v in config["param schedule"][i]:
        #                 setattr(model,k,v)
        #         #_, history = model.train()
        #         gs, _  = graph_inc.select_nodes(model.net_gs, model.particle_fn, model.particle_mc_params, model.immunized_ids, config["budget"])
        #
        #         ret_orig = model.significance_replicates(args={"walk_iters":100})
        #
        #         #graph_random_walks(net_r, rewards, starts_r, max_walk, walks, fn)
        #         #ret_model = model.significance_replicates(state="edit", args={"walk_iters":100})
        #         if config["evaluation baseline"]:
        #             ret_baseline = model.significance_replicates(nets= gs, args={"walk_iters":100})
        #         else:
        #             ret_baseline = c.defaultdict(lambda:np.nan)
        #
        #         ret_model = ret_baseline
        #         results = {"orig":ret_orig, "baseline": ret_baseline, "model": ret_model}
        #         t = time.time()
        #         print({ "gamma": model.gamma, "budget": model.budget, "T":model.T, "epochs": model.epochs})
        #         df = pd.DataFrame(columns=['info','baseline','model','orig'])
        #
        #         rows = defaultdict(dict)
        #         for k,v in results.items():
        #             for i,measure in enumerate(v["measures"]):
        #                 vv = np.mean(measure)
        #                 eval_log[(k, "flow:{}".format(i))].append(vv)
        #                 rows[i][k]=vv
        #                 rows[i]['info']='group-{} flow'.format(i)
        #
        #         for k,row in rows.items():
        #             df = df.append(row,ignore_index=True)
        #
        #         row = {'info':'ANOVA'}
        #         for k, v in results.items():
        #             vs = v["measures"]
        #             #print('vs:',vs)
        #             _,pvlaue = stats.f_oneway(*vs)
        #             eval_log[(k, "p-ttest")].append(pvlaue)
        #             row[k]=pvlaue
        #         df = df.append(row,ignore_index=True)
        #
        #         row = {'info': 'Gini'}
        #         for k,v in results.items():
        #             vs = v["measures"]
        #             vv = graph_inc.gini(np.array(list(it.chain(*vs))))
        #             eval_log[(k, "gini")].append(vv)
        #             row[k]=vv
        #         df = df.append(row, ignore_index=True)
        #
        #         row = {'info':'kruskal'}
        #         for k, v in results.items():
        #             vs = v["measures"]
        #             #print('vs:',vs)
        #             _,pvlaue = stats.kruskal(*vs)
        #             eval_log[(k, "kruskal")].append(pvlaue)
        #             row[k]=pvlaue
        #         df = df.append(row, ignore_index=True)
        #
        #         hypothesis_difference_fraction = config['hypothesis_difference_fraction']
        #         row = {'info':'p-ttest-dff'}
        #         for k, v in results.items():
        #                 vns = []
        #                 N = config["particles MC"][0]["particles"]
        #                 R = v["measures"][0]
        #                 B = v["measures"][1]
        #                 for measure in v["measures"]:
        #                     vns.append(np.var(R,ddof=1) / N)
        #
        #                 vn1 = np.var(R,ddof=1) / N
        #                 vn2 = np.var(B,ddof=1) / N
        #
        #                 vns = np.array(vns)
        #
        #                 np.sum(vns)**2 / (np.sum(vns**2)/(N-1))
        #                 dof = ((vn1 + vn2)**2) / ((vn1**2) / (N - 1) + (vn2**2) / (N - 1))
        #
        #                 hypothesis_difference = hypothesis_difference_fraction * np.mean([np.mean(R),np.mean(B)])
        #                 tscore = np.max([0,np.abs(np.mean(R)-np.mean(B)) - hypothesis_difference])/np.sqrt(vn1+vn2)
        #                 pvalue = stats.t.sf(np.abs(tscore), dof) * 2
        #
        #                 eval_log[(k, "p-ttest-dff")].append(pvalue)
        #                 row[k]=pvalue
        #         df = df.append(row,ignore_index=True)
        #         print(df)
        #
        #         for k, v in results.items():
        #             plt.figure()
        #             for vv in v["measures"]:
        #                 plt.hist(vv, alpha=0.5)
        #             plt.title("Measure distribution: "+ k)
        #             plt.savefig(os.path.join(results_root, "model_dist_{}_{}_{}.png".format(method, k,config['budget'], start_time)))
        #
        #         print("Time :" + str((time.time() - t) / 60))
        #         jl.dump({"df":df, "history":history.history, "simulations":results , "config": config, "results": eval_log, "graph before": model.net_gs, "graph trained": model.net_gs_trained }, os.path.join(results_root, str(method)  +'_'+ start_time + ".gz"))
        #         rlp.print_scatter(results_root, config["budget"])