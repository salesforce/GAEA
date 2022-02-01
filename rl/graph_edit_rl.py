#!/usr/bin/env python
# coding: utf-8

# In[1]:

#config = tf.ConfigProto()
#config.gpu_options.allow_growth = True
#sess = tf.Session(config=config)

#K.set_session(sess)
import itertools as it
import joblib as jl
from tensorflow.keras import backend as K
from tensorflow.keras.callbacks import Callback
from tensorflow.keras.constraints import NonNeg as nonneg
import tensorflow.keras.layers as kl
import tensorflow.keras.models as km
from tensorflow.keras.optimizers import Adam
import numpy as np
import numpy.random as npr
from rl import graph_includes as graph_inc
from tensorflow.keras.callbacks import TensorBoard
from datetime import datetime
import time
import copy
from rl import model_util

class ParameterSchedule(Callback):
    def __init__(self, 
                 constraint_coeff_mu, 
                 constraint_coeff_mu_update_factors, 
                 constraint_scheme,
                 fair_loss_normalize_variance, 
                 fair_loss_p,
                 budget,
                 N,
                 constraint_epoch_start_schedule,
                 constraint_epsilon,
                 alternate_loss,
                 data_gen,
                 config):
        self.constraint_scheme = constraint_scheme
        self.constraint_epoch_start_schedule = constraint_epoch_start_schedule
        self.constraint_epsilon = constraint_epsilon
        
        self.constraint_coeff_mu = constraint_coeff_mu
        
        self.alternate_loss = alternate_loss
        self.flow_loss_switch = K.variable(1)
        self.fair_loss_switch = K.variable(1)
        self.budget_loss_switch = K.variable(1)
        
        self.fair_loss_mu = K.variable(0)#constraint_coeff_mu[0])
        self.edit_loss_mu = K.variable(0)#constraint_coeff_mu[1])
        
        self.fair_loss_mu_update_factor = constraint_coeff_mu_update_factors[0]
        self.edit_loss_mu_update_factor = constraint_coeff_mu_update_factors[1]
        
        self.fair_loss_lambda = K.variable(0)
        self.edit_loss_lambda = K.variable(0)
        
        self.fair_loss_lambda_value = 0
        self.edit_loss_lambda_value = 0
        
        self.fair_loss_normalize_variance = fair_loss_normalize_variance
        self.fair_loss_p = fair_loss_p
        
        self.budget = budget
        self.N = N
        
        self.epoch_act = 0
        self.batch_act = 0
        
        self.data_gen = data_gen
        self.config = config
        self.A_mask = None
        
        K.set_value(self.flow_loss_switch,1)
        K.set_value(self.fair_loss_switch,1)
        K.set_value(self.budget_loss_switch,1)
        
        self.mask = None
        
        self.temp = np.ones((1,1))
        
    def set_mask(self,A,mask):
        self.A = A
        self.mask = mask
        
    def set_temp(self,temp_var,temp_decay_factor):
        self.temp_var=temp_var
        self.temp_decay_factor = temp_decay_factor
        
        
    
    def _schedule(self, min_batch, epoch):
        if epoch>self.constraint_epoch_start_schedule[0]:
            K.set_value(self.fair_loss_mu, self.constraint_coeff_mu[0])
            
            if self.alternate_loss:
                if min_batch%2==0:
                    K.set_value(self.flow_loss_switch,1)
                    K.set_value(self.fair_loss_switch,0)
                    K.set_value(self.budget_loss_switch,0)
                else:
                    K.set_value(self.flow_loss_switch,0)
                    K.set_value(self.fair_loss_switch,1)
                    K.set_value(self.budget_loss_switch,0)
        
        
        if epoch>self.constraint_epoch_start_schedule[1]:
            K.set_value(self.edit_loss_mu, self.constraint_coeff_mu[1])
            if min_batch%3==0:
                K.set_value(self.budget_loss_switch,1)
                K.set_value(self.flow_loss_switch,1)
                K.set_value(self.fair_loss_switch,1)
              
        print('>>>>>>>>>>>>>>>>>')
        print('flow_loss_switch:{}\tfair_loss_switch:{}\tbudget_loss_switch:{}'.format(
            K.get_value(self.flow_loss_switch),
            K.get_value(self.fair_loss_switch),
            K.get_value(self.budget_loss_switch
            )))
        print('>>>>>>>>>>>>>>>>>')
        
    
    # customize your behavior
    def on_epoch_end(self, epoch, logs={}):
        if epoch>1:
            K.set_value(self.A,np.expand_dims(self.mask, axis=(0,-1)))
        else:
            K.set_value(self.flow_loss_switch,1)
            K.set_value(self.fair_loss_switch,1)
            K.set_value(self.budget_loss_switch,1)
            
        if 'lagrangian_augmentation' in self.constraint_scheme:
            outputs = [[],[]]
            if self.epoch_act>self.constraint_epoch_start_schedule[0] or self.epoch_act>self.constraint_epoch_start_schedule[1]:
                for _ in range(self.config['model_num_exp']):
                    output = self.model.predict(next(self.data_gen)[0])
                    outputs[0].extend(output[0])
                    outputs[1].extend(output[1])
                
            outputs[0] = np.array(outputs[0])
            outputs[1] = np.array(outputs[1])
                
            if self.epoch_act>self.constraint_epoch_start_schedule[0]:
                # Learning proxy lagrangian for fair loss
                vs = outputs[0]
                
                Vs = [np.mean(vs[i]) for i in range(len(vs))]
                fairlosses = np.sum(np.abs(Vs - np.mean(Vs)))**self.fair_loss_p
                
                if self.fair_loss_normalize_variance:
                    V_variances = np.array([np.var(vs[i]) for i in range(len(vs))])
                    fairlosses = fairlosses / np.sqrt(np.sum(V_variances/self.N)) 
                    
                #fairlosses = np.mean(fairlosses)
                fairlosses = 2 * K.get_value(self.fair_loss_mu) * fairlosses
                
                K.set_value(self.fair_loss_lambda, (K.get_value(self.fair_loss_lambda) + fairlosses)/2)
            
            if self.epoch_act>self.constraint_epoch_start_schedule[1]:
                # Learning proxy lagrangian for edit loss
                Es = outputs[1]
                edit_loss = 2 * K.get_value(self.edit_loss_mu) * np.mean(np.sum(Es, axis=(1,)) - self.budget)
                
                K.set_value(self.edit_loss_lambda, (K.get_value(self.edit_loss_lambda) + self.config['lagrangian_lamdba_growth'] * edit_loss))
                
        if self.epoch_act>self.constraint_epoch_start_schedule[2]:
            self.temp *= self.temp_decay_factor
            K.set_value(self.temp_var, self.temp)
        print('\n-------\n temp:',self.temp,'\n-------')
                
        if 'squared' in self.constraint_scheme:
            K.set_value(self.fair_loss_mu, K.get_value(self.fair_loss_mu) * self.fair_loss_mu_update_factor)
            K.set_value(self.edit_loss_mu, K.get_value(self.edit_loss_mu) * self.edit_loss_mu_update_factor)
        print('fair_loss_mu:{},edit_loss_mu:{}'.format(K.get_value(self.fair_loss_mu), K.get_value(self.edit_loss_mu)))
        print('fair_loss_lambda:{},edit_loss_lambda:{}'.format(K.get_value(self.fair_loss_lambda), K.get_value(self.edit_loss_lambda)))
        
        
        self.epoch_act+=1
        self._schedule(self.epoch_act,self.epoch_act)
        
        
        
        
    def on_batch_end(self, batch, logs=None):
        self.batch_act+=1
        

            
            
        
class FlowGraphEditRL():
    def __init__(self,config={}):
        ##graph
        self.config = config
        im_def = self.map_defaults(config, "immunized nodes", {"random":5})
        graph_fn = self.map_defaults(config, "graph fn", graph_inc.get_graph)
        graph_params = self.map_defaults(config, "graph params", {"model": "karate", "im": im_def})
        start = time.time()
        g_s,mask, im = graph_fn(**graph_params)
        print('Time to load graph:',time.time()-start)
        self.immunized_ids = im
        self.net_gs = g_s
        self.N = len(self.net_gs[0])
        self.mask = mask    

        #states
        states_fn = self.enforce_fn(self.map_defaults(config, "states fn","init_node_states"))
        states_params = self.map_defaults(config, "states params", {})
        self.states = states_fn(**states_params)
        self.num_states = len(self.states)

        #params
        self.num_groups = self.map_defaults(config, "num_groups", 2)
        self.T = self.map_defaults(config, "T", 4)
        self.train_iters = self.map_defaults(config, "train iters", 100)
        self.batch_size_state = self.map_defaults(config, "batch size state", 100)
        self.batch_size_train = self.map_defaults(config, "batch size train", 36)
        self.epochs = self.map_defaults(config, "epochs", 20)
        self.particles = self.map_defaults(config, "particles", 50)
        self.gamma = self.map_defaults(config, "gamma", 0.9)
        self.budget = self.map_defaults(config, "budget", 5)
        self.temp_decay_factor = self.map_defaults(config, "temp_decay_factor", 0.02)
        self.alternate_loss = self.map_defaults(config, "alternate_loss", False)
        
        self.constraint_coeff_mu = self.map_defaults(config, "constraint_coeff_mu", [0.1,0.1])
        self.constraint_coeff_mu_update_factors = self.map_defaults(config, "constraint_coeff_mu_update_factors", [1.01,1.01])
        self.constraint_coeff_lambdas = self.map_defaults(config, "constraint_coeff_lambdas", [0,0])
        self.beta_update_factor = self.map_defaults(config, "beta_update_factor", 1.01)
        
        self.main_obj_optimizer_p = self.map_defaults(config, "main_obj_optimizer_p", 1)
        self.edit_bias_init_mu = self.map_defaults(config, "edit_bias_init_mu", 0)
        
        self.constraint_epoch_start_schedule = self.map_defaults(config, "constraint_epoch_start_schedule", [0,0])
        
        self.lr = self.map_defaults(config, "lr", 0.01)
        self.model_type = self.map_defaults(config, "model_type", 'bootstrapped')
        self.steps_per_epoch = self.map_defaults(config, "steps_per_epoch", 10)
        self.use_same_flow_matrix = self.map_defaults(config, "use_same_flow_matrix", False)
        self.constraint_scheme = self.map_defaults(config, "constraint_scheme", 'squared')
        self.fair_loss_p = self.map_defaults(config, "fair_loss_p", 1)
        self.fair_loss_normalize_variance = self.map_defaults(config, "fair_loss_normalize_variance", False)
        self.constraint_epsilon = self.map_defaults(config, "constraint_epsilon", [0.,0.])
        
        #particles
        self.particle_fn = self.enforce_fn(self.map_defaults(config, "particle fn","init_particle_locs_random"))
        self.exp_name  = self.map_defaults(config, "exp_name", 'exp')
        self.method = self.map_defaults(config, "method","exp")

        self.particle_params = self.map_defaults(config, "particle params", [{}, {}])
        self.particle_mc_params =  self.map_defaults(config,"particles MC", [{"particles":10000}, {"particles":10000}])
        self.num_groups = len(self.particle_params)
        #self.particle_params_b = self.map_defaults(config, "particle params b", {})
        self.locs = [self.particle_fn(**s) for s in  self.particle_params]
        #self.loc_r =  self.particle_fn(**self.particle_params_r)
        #self.loc_b = self.particle_fn(**self.particle_params_b)

        self.immunized_nodes = np.array(np.zeros(self.N))
        self.immunized_nodes[[im]] = 1

        #training
        if ("no training" not in config or config['no training']==False):

            self.param_schedule = ParameterSchedule(self.constraint_coeff_mu,
                                                    self.constraint_coeff_mu_update_factors,
                                                    self.constraint_scheme,
                                                    self.fair_loss_normalize_variance,
                                                    self.fair_loss_p,
                                                    self.budget,
                                                    self.N,
                                                    self.constraint_epoch_start_schedule,
                                                    self.constraint_epsilon,
                                                    self.alternate_loss,
                                                    self.get_df_data_gen(),
                                                    self.config)

            timestamp = datetime.now().strftime('%Y-%m-%d_%H-%M-%S.%f')[:-3]
            self.tbCallBack = TensorBoard(log_dir='../Graph/{}_{}_{}'.format(self.exp_name, self.method, timestamp))
            states_fn = self.enforce_fn(self.map_defaults(config, "state train fn", "init_states_basic"))
            states_params = self.map_defaults(config, "state train params", {})
            #self.model_net_r = None
            #self.model_net_b = None
            self.dense_gs = None

            self.net_gs_trained = None
            #self.net_r_trained = None
            #self.net_b_trained = None
            self.train_history = None
            states_fn(**states_params)
            self.states_fn = states_fn
            self.states_params = states_params

            graph_edit_fn = self.enforce_fn(self.map_defaults(config, "graph edit fn","get_graph_edit_model"))
            graph_edit_params = self.map_defaults(config, "graph edit params", {})
            self.graph_edit_fn = graph_edit_fn
            self.graph_edit_params = graph_edit_params
            self.model_graph_edit_train, self.model_graph_edit_predict = graph_edit_fn(**graph_edit_params)

        #print(config)

    def reset(self):
        self.model_graph_edit_train, self.model_graph_edit_predict = self.graph_edit_fn(**self.graph_edit_params)
        self.states_fn(**self.states_params)

    def significance_replicates(self, args={}, state=None, nets=None):
        if not nets:
            nets = self.net_gs
        if state=="edit":
            nets = self.net_gs_trained
        return graph_inc.evaluate_model_stat_replicates(nets, rewards=self.immunized_ids, fn_replicates=self.particle_fn, params_replicates=self.particle_mc_params, **args)
    def init_states_basic(self, epochs=None):
        if epochs is None:
            epochs = self.epochs
            
        self.dense_gs = []
        for i in range(self.num_groups):
            _, dense_g = self.get_flow_model(self.net_gs[i])
            self.dense_gs.append(dense_g)
            
    def value_iteration(self,P,R,T,gamma=0.9,epochs=5):
        V = np.zeros((T,self.num_states))
        for _ in range(epochs):
            for t in reversed(range(T)):
                if t<T-1:
                     V[t] = R + gamma * np.dot(P,V[t+1]) 
                else:
                     V[t] = R
        return V

    def train(self):
        if self.model_type == 'bootstrapped':
            return self.train_bootstrapped()
        elif self.model_type == 'fullydifferntiable':
            return self.train_fullydifferntiable()
        else:
            raise Exception('select model type')
        
    def get_df_data_gen(self):
        Y = [np.array([0] * self.config['batch_size']), np.array([0] * self.config['batch_size'])]#, np.array([self.mask] * self.config['batch_size'])]
        
        i = 0
        temp = np.ones((1,1))*1.0
        
        def batch_repeat(a):
            al = np.ones(a.ndim,dtype=int)
            al[0] = self.config['batch_size']
            return np.tile(a,tuple(al))
        
        R = batch_repeat(np.expand_dims(np.array(self.immunized_nodes.tolist()), axis=0)) 
        W = batch_repeat(np.expand_dims(np.array(self.net_gs[0]),axis=(0,-1)))
        R_mat = batch_repeat(np.expand_dims(np.diag(self.immunized_nodes), axis = (0,-1)))
        A = batch_repeat(np.expand_dims(self.mask, axis=(0,-1)))
        
        
        while True:
            locs = []
            for s in  self.particle_params:
                s = copy.deepcopy(s)
                s['particles'] = self.config['batch_size']
                locs.append(self.particle_fn(**s))
            
            state_gs = []
            for loc in locs:
                state_g = []
                for k in loc:
                    v = np.zeros(self.N)
                    v[k] = 1
                    state_g.append(v)
                state_g = np.asarray(state_g)
                state_gs.append(state_g)

            if self.param_schedule.epoch_act>self.constraint_epoch_start_schedule[2]:
                temp *= self.temp_decay_factor
            i+=1
            yield (state_gs,Y)
            

    def train_fullydifferntiable(self):

        s0_all = np.zeros((self.num_states,self.num_states))
        for i in range(self.num_states):
            s0_all[i,i] = 1

        df_data_gen = self.get_df_data_gen()
        
        history = self.model_graph_edit_train.fit_generator(df_data_gen,
                                       steps_per_epoch = self.steps_per_epoch,
                                       epochs=self.epochs, 
                                       callbacks=[self.param_schedule,self.tbCallBack],
                                       verbose=1)

        
        net_g_updated_partials = [[] for _ in range(self.num_groups)]
        print('#_# net_g_updated_partials:',net_g_updated_partials)
        for i in range(0,len(s0_all),1):
            print('Predicted :{} of {}'.format(i,len(s0_all)))
            s0_sub=s0_all[i:min(i+1,len(s0_all))]
            
            E, Prs = self.model_graph_edit_predict.predict([s0_sub for _ in range(self.num_groups)])
            
            break
        print('E',E[0].shape)

        # ### Positions of edits (red folloed by black)
        #np.round(self.model_graph_edit.predict([S0_rs, S0_bs, Rs, temp])[1][:1], 0)
        self.net_gs_trained = E[0]
        self.train_history = history.history
        return E[0], history
    

    def train_bootstrapped(self):
        ret = []
        net_r_updated = self.net_r
        net_b_updated = self.net_b
        
        s0_all = np.zeros((self.num_states,self.num_states))
        for i in range(self.num_states):
            s0_all[i,i] = 1

        for i in range(10):
            for j in range(self.train_iters):
                Vr = []
                Vb = []
                
                Vtr = self.value_iteration(net_r_updated,self.immunized_nodes,self.T,gamma=self.gamma,epochs=1)
                Vtb = self.value_iteration(net_b_updated,self.immunized_nodes,self.T,gamma=self.gamma,epochs=1)
        
                opt_T = self.T
                for t in range(1,opt_T,1):
                #for t in range(1,2,1):
                   for v in range(len(self.loc_r)):
                      Vr.append(Vtr[t])
                   for v in range(len(self.loc_b)):
                      Vb.append(Vtb[t])
                
                temp = 1 / (i * j + 1)
                #print('temp:', temp)
                temp = np.array([temp] * self.particles*(opt_T-1))
                Rs = np.array([self.immunized_nodes.tolist()] * self.particles*(opt_T-1))

                S0_rs = np.array(self.state_r *(opt_T-1))
                S0_bs = np.array(self.state_b *(opt_T-1))

                Vrs = np.array(Vr)
                Vbs = np.array(Vb)

                history = self.model_graph_edit.fit([S0_rs, S0_bs, Vrs, Vbs, Rs, temp],
                                               [np.array([0] * self.particles*(opt_T-1)),
                                                np.array([self.mask]*self.particles*(opt_T-1))],
                                                epochs=self.epochs,
                                                batch_size=self.batch_size_train, 
                                                verbose=1)

                Prs = self.model_graph_edit.predict([s0_all,
                                                s0_all,
                                                Vrs[:self.num_states],
                                                Vbs[:self.num_states],
                                                Rs[:self.num_states],
                                                temp[:self.num_states]])
                net_r_updated = Prs[2][:self.num_states].tolist()
                net_b_updated = Prs[3][:self.num_states].tolist()
                
                #nodes = Prs[0][:self.num_states].to

        # ### State transitions of Red particle Before Vs After
        #Pr
        #np.round(self.model_graph_edit.predict([S0_rs, S0_bs, Vrs, Vbs, Rs, temp])[2][:self.num_states - 1], 2)

        # ### State transitions of Black particle Before Vs After
        #Pb
        #np.round(self.model_graph_edit.predict([S0_rs, S0_bs, Vrs, Vbs, Rs, temp])[3][:self.num_states - 1], 2)

        # ### Positions of edits (red folloed by black)
        #np.round(self.model_graph_edit.predict([S0_rs, S0_bs, Vrs, Vbs, Rs, temp])[1][:1], 0)
        self.net_r_trained = net_r_updated
        self.net_b_trained = net_b_updated
        self.train_history = history.history
        return net_b_updated, net_r_updated, history

    def init_particle_locs_degree(self, particle_map, nodes=None, inverse=False):
        if nodes is None:
            nodes = range(self.N)
        nodes = [i for i in nodes if i not in self.immunized_ids]
        mask = np.array(self.mask)
        degrees = np.array([len(mask[i].nonzero()[0]) for i in nodes])
        if inverse:
            degrees = (1 / degrees)

        p = particle_map["fn"](degrees, **particle_map["fn params"])
        return npr.choice(nodes, p=p, size=self.particles, replace=True)


    def init_particle_locs_random(self, node_range = None, particles=None):
        if particles is None:
            particles = self.particles
        if node_range is None:
            node_range = range(self.N)
        return npr.choice([v for v in node_range if v not in self.immunized_ids],particles, replace=True)
    
    def init_particle_locs_geo(self, df_nodes, df_tract, pop_key="white", particles=None):
        if particles is None:
            particles = self.particles

        total = np.sum([v for v in df_tract[pop_key]])
        p = [v/total for v in df_tract[pop_key]]

        ti_map = {}
        for ti in df_tract.index:
            ti_map[ti] = list(df_nodes.index[df_nodes["tracts"] == ti])

        tids = npr.choice(df_tract.index, particles*4, p=p, replace=True)
        ret = []
        for tid in tids:
            a = ti_map[tid]
            if a:
                aa = npr.choice(a)
                if aa not in self.immunized_ids:
                    ret.append(aa)
            if len(ret)==particles:
                break
        return ret
    def init_particle_locs_fb(self, d, pop_key=1, particles=None): # 1 = female, 2 = male
        if particles is None:
            particles = self.particles
        years = d["local_info"][:, 5]
        years_clean = [x for x,y in zip(*np.unique(years, return_counts=True)) if y > len(years)*.1 and x != 0]
        inds = [v for v in np.where(np.bitwise_and(d["local_info"][:, 1] == pop_key, d["local_info"][:, 5]==np.max(years_clean)))[0] if v not in self.immunized_ids]
        return npr.choice(inds, min(particles, len(inds)), replace=False)

    def init_node_states(self):
        return np.eye(self.N, dtype=int).tolist()

    def enforce_fn(self, default):
        if isinstance(default, str):
            default = getattr(self, default)
        return default

    def map_defaults(self, config, key, default):
        if key not in config:
            config[key] = default
            return default
        else:
            return config[key]

    def get_flow_model(self,P_init):
        s = kl.Input((self.num_states,))
        Dense_W = kl.Dense(self.num_states, use_bias=False,kernel_constraint=nonneg(),weights = [np.array(P_init)])
        logit = Dense_W(s)
        #P = kl.Activation('softmax')(logit)
        P = kl.Lambda(lambda z:z/K.sum(z,axis=-1,keepdims=True))(logit)

        model = km.Model(inputs=[s], outputs=P)

        #model.summary()
        model.compile(
            loss='categorical_crossentropy',
            optimizer='adam'
        )
        return model, Dense_W

    def get_bs_particle_graphs(self, s, Dense_W, V, R, temp):
        Dense_W(s)
        W = Dense_W.weights[0]

        Dense_W_ = kl.Dense(self.num_states, use_bias=False, kernel_constraint=nonneg())
        Dense_W_(s)
        W_ = Dense_W_.weights[0]

        s_embed = kl.Dense(self.num_states, activation='tanh')(s)
        Dense_E = kl.Dense(self.num_states * self.num_states)
        null_input = kl.Lambda(lambda z: 0 * z[:,0:1])(s_embed)
        E_logit = kl.Dense(self.num_states,activation='tanh')(null_input)
        E_logit = Dense_E(E_logit)
        E_logit = kl.Lambda(lambda z: z / temp)(E_logit)
        E = kl.Activation('sigmoid')(E_logit)
        E = kl.Reshape((self.num_states, self.num_states))(E)

        E = kl.Lambda(lambda z: self.mask * z)(E)

        Dense_W.trainable = False
        logit_1 = Dense_W(s)

        logit_2 = kl.Lambda(lambda z: W_ * (z))(E)
        logit_2 = kl.Dot(axes=1)([logit_2, s])

        logit = kl.Add()([logit_1, logit_2])
        #P = kl.Activation('softmax')(logit)
        P = kl.Lambda(lambda z:z/K.sum(z,axis=-1,keepdims=True))(logit)
        V_ = kl.Dot(axes=-1)([P, V])

        r = kl.Dot(axes=-1)([s, R])

        v = kl.Lambda(lambda z: z[0] + self.gamma * z[1])([r, V_])
        return W, W_, E, P, V_, v

    def get_fd_particle_type_sub_graph(self, s, P_transpose, R, T):
        
        vs = []
        for i in range(T):
            r = kl.Lambda(lambda z:K.sum(z[0]*z[1],axis=-1,keepdims=True))([s,R])
            v = kl.Lambda(lambda z:(self.gamma**i)*z)(r)
            vs.append(v)
            
            s = kl.Dot(axes=(-1))([P_transpose,s])
            
        v = kl.Add()(vs)
        return v

        

    def get_graph_edit_model(self): #Done
        if self.model_type == 'bootstrapped':
            return self.get_graph_edit_bootstrapped_model()
        elif self.model_type == 'fullydifferntiable':
            return self.get_graph_edit_fullydifferntiable_model()
        else:
            raise Exception('select model type')

    def get_graph_edit_fullydifferntiable_model(self): #Done
        
        sgs = []
        for i in range(self.num_groups):
            sgs.append(kl.Input((self.num_states,),name='sg{}'.format(i)))
            
            
        #R = kl.Input((self.num_states,))
        #print('self.immunized_nodes.tolist():',self.immunized_nodes.tolist())
        R = np.expand_dims(np.array(self.immunized_nodes.tolist()), axis=0) 
        self.temp_var = K.variable(np.ones((1,1)))
        self.param_schedule.set_temp(self.temp_var, self.temp_decay_factor)
        W = np.expand_dims(np.array(self.net_gs[0]),axis=(0,-1))
        Wd = np.expand_dims(np.zeros(np.array(self.net_gs[0]).shape),axis=(0,-1))
        R_mat = np.expand_dims(np.diag(self.immunized_nodes), axis = (0,-1))
        A = np.expand_dims(np.zeros(self.mask.shape), axis=(0,-1))
        
        W[W>0] = 1.
        print('# of editable edges:',np.sum(self.mask))
        print('Budget:',self.budget)
        print('=========================================')
        
        R = kl.Input(tensor=K.constant(R), name="R")
        temp = kl.Input(tensor=self.temp_var, name="temp")
        W = kl.Input(tensor=K.constant(W), name="W")
        Wd = kl.Input(tensor=K.constant(Wd), name="Wd")
        R_mat = kl.Input(tensor=K.constant(R_mat), name="R_mat")
        E_features = kl.Concatenate(axis=-1)([R_mat,W])
        A = kl.Input(tensor=K.variable(A), name="A")
        self.param_schedule.set_mask(A, self.mask)

        dense_W = kl.Dense(self.num_states, name='dense_W')
        null_input = kl.Lambda(lambda z:0*z)(sgs[0])
        dense_W(null_input)
        W_d = dense_W.weights[0]
        
        yWd = kl.Lambda(lambda z:K.squeeze(z,axis=-1))(Wd)
        yWd = kl.Lambda(lambda z:z+W_d)(yWd)
        yWd = kl.Lambda(lambda z:K.expand_dims(z, axis=-1))(yWd)
        E_features = kl.Concatenate()([E_features,yWd])

        edge_inp = E_features#kl.Concatenate()([R,])
        for i in range(self.config["edge_num_layers"]):
            edge_inp = kl.Conv2D(self.config['hidden_dims'],
                                 self.config["edge_kernel_size"],
                                 strides=1,
                                 dilation_rate=i+1,
                                 activation='tanh',
                                 padding='same')(edge_inp)
#         edge_inp = kl.Concatenate()([edge_inp,yWd])
        new_edges = kl.Conv2D(1,
                 1,
                 strides=1,
                 activation=None,
                 padding='same')(edge_inp)
        
          
        new_edges = kl.Lambda(lambda z:z -K.log(-1*K.log(K.random_uniform(shape=K.shape(z), minval=0., maxval=1.0) + K.epsilon()) + K.epsilon()))(new_edges)
        new_edges = kl.Lambda(lambda z:z[0]/z[1])([new_edges,temp])
        
        E = kl.Activation('sigmoid')(new_edges)
        #E = kl.Reshape((self.num_states,self.num_states))(E)
        E = kl.Lambda(lambda z:z[0]*z[1]*(1-z[2]))([E,A,W])
        
        W_effect = kl.Lambda(lambda z:K.squeeze(z[0] + z[1],axis=-1))([W,E])
        P = kl.Lambda(lambda z:z/K.sum(z,axis=-1,keepdims=True))(W_effect)
        p_transpose = kl.Permute((2,1))(P)
         
        vgs = []
        for i in range(self.num_groups):
            vg = self.get_fd_particle_type_sub_graph(sgs[i],p_transpose,R,self.T)
            vgs.append(vg)

        vs = kl.Concatenate(name='value')(vgs)
        num_edits = kl.Lambda(lambda z:K.expand_dims(K.sum(z[0]*z[1],axis=(-1,-2,-3)),axis=-1))([A,E])
        num_edits = kl.Layer(name='edit')(num_edits)
        inps = sgs+[temp,R,W,R_mat,A,Wd]

        model = km.Model(inputs=inps, outputs=[vs,num_edits])
        model.layers[-1].trainable_weights.extend([W_d])
        model.summary()
        
        model.compile(
            loss={
                'value':self.fair_flow_loss,
                'edit':self.budget_loss,
            }, 
            loss_weights={
                'value':1,
                'edit':1
            },
            optimizer=Adam(lr=self.lr),
            metrics = {
                'value':[model_util.get_flow_loss(self.main_obj_optimizer_p), model_util.get_fair_loss(self.config), model_util.diff_bw_groups, model_util.mean_value],
                'edit':[model_util.get_num_edits_exceeded(self.budget)]
            }
        )
        
        model_predict = km.Model(inputs=inps, outputs=[W_effect,P])
        return model, model_predict

    def get_graph_edit_bootstrapped_model(self): #Done
        sr = kl.Input((self.num_states,))
        sb = kl.Input((self.num_states,))
        Vr = kl.Input((self.num_states,))
        Vb = kl.Input((self.num_states,))
        R = kl.Input((self.num_states,))
        temp = kl.Input((1,))

        Wr, Wr_, Er, Pr, Vr_, vr = self.get_bs_particle_graphs(sr, self.dense_r, Vr, R, temp)
        Wb, Wb_, Eb, Pb, Vb_, vb = self.get_bs_particle_graphs(sb, self.dense_b, Vb, R, temp)

        vs = kl.Concatenate(name='value')([vr, vb])

        Er = kl.Lambda(lambda z: K.expand_dims(z, axis=1))(Er)
        Eb = kl.Lambda(lambda z: K.expand_dims(z, axis=1))(Eb)
        Es = kl.Concatenate(name='edit', axis=1)([Er, Eb])

        model = km.Model(inputs=[sr, sb, Vr, Vb, R, temp], outputs=[vs, Es, Pr, Pb])
        model.layers[-1].trainable_weights.extend([Wr_, Wb_])
        model.layers[-1].non_trainable_weights.extend([Wr, Wb])

        model.summary()

        model.compile(
            loss={
                'value': self.fair_flow_loss,
                'edit': self.budget_loss,
            },
            loss_weights={
                'value': 1,
                'edit': 1
            },
            optimizer=Adam(lr=self.lr),
            metrics={
                'value': [model_util.get_flow_loss(self.main_obj_optimizer_p), model_util.get_fair_loss(self.config)]
            }
        )
        return model
    
    def num_edits_exceeded_(self, y_true, y_pred):
        edits_exceeded = K.clip(K.mean(y_pred) - self.budget, 0 , 1e20)
        return edits_exceeded
    
    def budget_loss(self, y_true, y_pred):
        edit_loss = model_util.get_num_edits_exceeded(self.budget)(y_true, y_pred)
        #edit_loss = K.clip(edit_loss, self.constraint_epsilon[1] , 1e20) - self.constraint_epsilon[1]
        
        if 'squared' in self.constraint_scheme:
            edit_loss = self.param_schedule.edit_loss_mu * (edit_loss**2)
        elif 'lagrangian_augmentation' in self.constraint_scheme:
            edit_loss = self.param_schedule.edit_loss_mu * (edit_loss**2) + 2 * self.param_schedule.edit_loss_lambda * edit_loss
        
        return edit_loss * self.param_schedule.budget_loss_switch
        #return K.mean(K.clip(K.sum(y_pred, axis=(1, 2, 3)) - self.budget, 0, 1e20), axis=0)

    def fair_loss_spectral_norm(self, y_true, y_pred):
        fair_loss = K.max(y_pred,axis=-1)-K.min(y_pred,axis=-1)
        print('^_^ fair_loss:',K.int_shape(fair_loss))
        return fair_loss
           
    def weighted_fair_loss(self, y_true, y_pred): 
        #fair_loss = self.fair_loss_spectral_norm(y_true, y_pred)
        fair_loss = model_util.get_fair_loss(self.config)(y_true, y_pred)
        fair_loss = K.clip(fair_loss,self.constraint_epsilon[0],1e20)-self.constraint_epsilon[0]
        
        if 'squared' in self.constraint_scheme:
            fair_loss = self.param_schedule.fair_loss_mu * (fair_loss**2)
        elif 'lagrangian_augmentation' in self.constraint_scheme:
            fair_loss = self.param_schedule.fair_loss_mu * (fair_loss**2) + 2 * self.param_schedule.fair_loss_lambda * fair_loss
        else:
            raise Exception('Invalid constraint_scheme:',self.constraint_scheme)
        
        return fair_loss
        #return K.square(K.mean(y_pred[:, 0]) - K.mean(y_pred[:, 1]))
        
    def fair_flow_loss(self, y_true, y_pred):
        return self.param_schedule.flow_loss_switch * model_util.get_flow_loss(self.main_obj_optimizer_p)(y_true, y_pred) + self.param_schedule.fair_loss_switch * self.weighted_fair_loss(y_true, y_pred)

