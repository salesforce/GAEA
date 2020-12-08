from config_global import *

config["epochs"] = [3, 600][model_type_idx]
config["budget"] = 21  # number of edits
config["constraint_coeff_mu"] = [0.005, 0.000001]  # Coeff of squared constraint
config["constraint_epoch_start_schedule"] = [20, 40, 100]