
import rl.graph_edit_rl as grl
import rl.graph_includes as graph_inc
import numpy.random as npr
import numpy as np
config = {}

model_type_idx = {'bootstrapped':0,
				  'fullydifferntiable':1}['fullydifferntiable']

config["immunized nodes"] = {"lambda:degree": {"fn": graph_inc.degree_scaled, "fn params":{}, "count": 3}}
config["epochs"] = [3,400][model_type_idx]
config["model_type"] = ['bootstrapped', 'fullydifferntiable'][model_type_idx]
config["use_same_flow_matrix"] = True
config["N"] = 400
config["batch size state"] = 200                                                        #batch size for initial state  training
config["batch size train"] = 200                                                         #batch for graph edit training
config["particles"] = 200                                                                # number of particles
config["gamma"] = 0.99                                                                   #fairness penalty weight
config["budget"] = 400                                                                  #number of edits
config["constraint_coeff_mu"] = [0.1, 1e-06]											# Coeff of squared constraint
config["constraint_coeff_mu_update_factors"] = [1.02, 1.02]								# Update factor of Coeff of squared constraint
config["constraint_epoch_start_schedule"] = [0, 100, 200]
config["constraint_epsilon"] = [0.0, 0.0] #[0.03, 0.99]
config["hypothesis_difference_fraction"] = 0.05
config["constraint_scheme"] = ['squared','lagrangian_augmentation'][1]					# For unconstrainted optimization
config["edit_bias_init_mu"]=0
config["main_obj_optimizer_p"]=2
config["alternate_loss"] = True

config["beta"] = 10
config["lr"] = 0.01
config["T"] = 10
config["T_max"] = 30
config["train iters"] = 300
config["steps_per_epoch"] = 10
config["evaluation replicates"] = 100
config["fair_loss_type"] = 'squared'
config["flow th"] = 0.0
config["hypothesis_difference_fraction"] = 0.05
config['temp_decay_factor']=0.999

config['exp_name']='exp_5e-9_0.9995_NoMeanaugLag-500'

config["group names"] = ["black", "red"]

config["fair_loss_p"] = 1.
config["fair_loss_normalize_variance"] = False

config["particle fn"] = "init_particle_locs_random"
config["particle params"] =[{}, {}]
config["particles MC"] = [{"particles":1000}, {"particles":1000}]

#config["particle params r"] ={"particle_map": {"fn params":{}, "fn":graph_inc.degree_scaled}, "inverse":True}
#config["particle params b"] ={"particle_map": {"fn params":{}, "fn":graph_inc.degree_scaled}, "inverse":False}

config["hidden_dims"]=10
config["edge_kernel_size"]=5
config["edge_num_layers"]=0
config["batch_size"]=50
config['model_num_exp']=10

config['lagrangian_lamdba_growth']=0.005
config['lagrangian_lamdba2_growth']=0.1
config['fair_loss_error'] = ['mae','mae2','mape','rmape','smape','zscore'][0]

