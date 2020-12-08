#!/usr/bin/env python
# coding: utf-8

# In[1]:

#config = tf.ConfigProto()
#config.gpu_options.allow_growth = True
#sess = tf.Session(config=config)

#K.set_session(sess)
import itertools as it
import joblib as jl
from keras import backend as K
from keras.callbacks import Callback
from keras.constraints import maxnorm, nonneg
import keras.layers as kl
import keras.models as km
from keras.optimizers import Adam
import numpy as np
import numpy.random as npr
from rl import graph_includes as graph_inc
from keras.callbacks import TensorBoard
import time


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
                 data_gen):
        
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
        
    # customize your behavior
    def on_epoch_end(self, epoch, logs={}):
        if 'squared' in self.constraint_scheme:
            K.set_value(self.fair_loss_mu, K.get_value(self.fair_loss_mu) * self.fair_loss_mu_update_factor)
            K.set_value(self.edit_loss_mu, K.get_value(self.edit_loss_mu) * self.edit_loss_mu_update_factor)
        print('fair_loss_mu:{},edit_loss_mu:{}'.format(K.get_value(self.fair_loss_mu), K.get_value(self.edit_loss_mu)))
        print('fair_loss_lambda:{},edit_loss_lambda:{}'.format(K.get_value(self.fair_loss_lambda), K.get_value(self.edit_loss_lambda)))
        
        if self.epoch_act>self.constraint_epoch_start_schedule[0]:
            K.set_value(self.fair_loss_mu, self.constraint_coeff_mu[0])
            
            if self.alternate_loss:
                if self.epoch_act%2==0:
                    K.set_value(self.flow_loss_switch,1)
                    K.set_value(self.fair_loss_switch,0)
                    K.set_value(self.budget_loss_switch,0)
                else:
                    K.set_value(self.flow_loss_switch,0)
                    K.set_value(self.fair_loss_switch,1)
                    K.set_value(self.budget_loss_switch,0)
        
        
        if self.epoch_act>self.constraint_epoch_start_schedule[1]:
            K.set_value(self.edit_loss_mu, self.constraint_coeff_mu[1])
            if self.epoch_act%3==0:
                K.set_value(self.budget_loss_switch,1)
                K.set_value(self.flow_loss_switch,0)
                K.set_value(self.fair_loss_switch,0)
              
        if self.epoch_act>self.constraint_epoch_start_schedule[0] or self.epoch_act>self.constraint_epoch_start_schedule[1]:
            print('>>>>>>>>>>>>>>>>>')
            print('flow_loss_switch:{}\tfair_loss_switch:{}\tbudget_loss_switch:{}'.format(
                K.get_value(self.flow_loss_switch),
                K.get_value(self.fair_loss_switch),
                K.get_value(self.budget_loss_switch
                )))
            print('>>>>>>>>>>>>>>>>>')
        
        self.epoch_act+=1
        
        
        
        
    def on_batch_end(self, batch, logs=None):
        self.batch_act+=1
        if 'lagrangian_augmentation' in self.constraint_scheme:
            if self.epoch_act>self.constraint_epoch_start_schedule[0] or self.epoch_act>self.constraint_epoch_start_schedule[1]:
                outputs = self.model.predict(next(self.data_gen)[0])
                
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
                edit_loss = 2 * K.get_value(self.edit_loss_mu) * np.mean(np.sum(Es, axis=(-1)) - self.budget)
                
                K.set_value(self.edit_loss_lambda, (K.get_value(self.edit_loss_lambda) + edit_loss))
            
            
        
class FlowGraphEditRL():
    def __init__(self,config={}):
        ##graph
        im_def = self.map_defaults(config, "immunized nodes", {"random":5})
        graph_fn = self.map_defaults(config, "graph fn", graph_inc.get_graph)
        graph_params = self.map_defaults(config, "graph params", {"model": "karate", "im": im_def})
        g_s,mask, im = graph_fn(**graph_params)
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
                                                self.get_df_data_gen())
        
        #particles
        self.particle_fn = self.enforce_fn(self.map_defaults(config, "particle fn","init_particle_locs_random"))
        self.exp_name  = self.map_defaults(config, "exp_name", 'exp')
        self.method = self.map_defaults(config, "method","exp")
        
        self.tbCallBack = TensorBoard(log_dir='../Graph/{}_{}_{}'.format(self.exp_name,self.method,int(time.time())))

        self.particle_params = self.map_defaults(config, "particle params", [{}, {}])
        self.particle_mc_params =  self.map_defaults(config,"particles MC", [{"particles":10000}, {"particles":10000}])
        self.num_groups = len(self.particle_params)
        #self.particle_params_b = self.map_defaults(config, "particle params b", {})
        self.locs = [self.particle_fn(**s) for s in  self.particle_params]
        #self.loc_r =  self.particle_fn(**self.particle_params_r)
        #self.loc_b = self.particle_fn(**self.particle_params_b)

        self.immunized_nodes = np.array(np.zeros(self.N))
        self.immunized_nodes[[im]] = 1

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
        self.model_graph_edit = graph_edit_fn(**graph_edit_params)

        print(config)

    def reset(self):
        self.model_graph_edit = self.graph_edit_fn(**self.graph_edit_params)
        self.states_fn(**self.states_params)

    def significance_replicates(self, args={}, state=None, nets=None):
        if not nets:
            nets = self.net_gs
        if state=="edit":
            nets = self.net_gs_trained
#            net_r = self.net_r_trained
#            net_b = self.net_b_trained
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
        if self.model_type == 'fullydifferntiable':
            return self.train_fullydifferntiable()
        else:
            raise Exception('select model type')
        
    def get_df_data_gen(self):
        Y = [np.array([0] * self.particles), np.array([self.mask] * self.particles)]
        
        i = 0
        temp = 1
        while True:
            
            locs = [self.particle_fn(**s) for s in  self.particle_params]
            
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
            temp_inp = np.array([temp] * self.particles)
            i+=1
            if i%100 == 0:
                print('\n-------\nT_T temp:',temp,'\n-------')
                
            yield (state_gs+[temp_inp],Y)
            

    def train_fullydifferntiable(self):

        s0_all = np.zeros((self.num_states,self.num_states))
        for i in range(self.num_states):
            s0_all[i,i] = 1

        df_data_gen = self.get_df_data_gen()
        
        history = self.model_graph_edit.fit_generator(df_data_gen,
                                       steps_per_epoch = self.steps_per_epoch,
                                       epochs=self.epochs, 
                                       callbacks=[self.param_schedule,self.tbCallBack],
                                       verbose=1)

        
        net_g_updated_partials = [[] for _ in range(self.num_groups)]
        for i in range(0,len(s0_all),10):
            s0_sub=s0_all[i:min(i+10,len(s0_all))]
            
            Prs = self.model_graph_edit.predict([s0_sub for _ in range(self.num_groups)]+
                                                [np.ones((self.num_states))])
            
            
            rewards = Prs[1][0]
            break
            #np.round(self.model_graph_edit.predict([S0_rs, S0_bs, Rs, temp])[2][:self.num_states - 1], 2)
        
        rewards = np.round(rewards,0)
        print('rewards:',rewards)
        # ### Positions of edits (red folloed by black)
        #np.round(self.model_graph_edit.predict([S0_rs, S0_bs, Rs, temp])[1][:1], 0)
        
        self.immunized_ids = np.where(rewards>1e-7)[0].tolist()
        self.net_gs_trained = self.net_gs
        self.train_history = history.history
        return self.net_gs, history

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
        tids = npr.choice([v for v in range(len(df_tract)) if v not in self.immunized_ids], particles*10, p=p, replace=True)
        ret = []
        for tid in tids:
            a = [i for i,v in enumerate(df_nodes["tracts"]) if v == tid]
            if a:
                ret.append(npr.choice(a))
            if len(ret)==particles:
                break
        return ret

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
        Dense_W = kl.Dense(self.num_states, use_bias=False,W_constraint=nonneg(),weights = [np.array(P_init)])
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

    def get_fd_particle_type_sub_graph(self,s,Dense_W,Dense_R,temp,T):
        Dense_W(s)
        W = Dense_W.weights[0]

        s_embed = kl.Dense(self.num_states,activation='tanh')(s)
        
        null_input = kl.Lambda(lambda z:0*z[:,0:1])(s_embed)
        R_logit = Dense_R(null_input)
        
        #Gumble
        #E_logit = kl.Lambda(lambda z:K.log(K.abs(z)+K.epsilon()) - K.log(-1*K.log(K.random_normal(shape=K.shape(E_logit), mean=0., stddev=1.0) + K.epsilon()) + K.epsilon()))(E_logit)
        R_logit = kl.Lambda(lambda z:z -K.log(-1*K.log(K.random_uniform(shape=K.shape(R_logit), minval=0., maxval=1.0) + K.epsilon()) + K.epsilon()))(R_logit)
        #E_logit = kl.Lambda(lambda z:K.log(K.abs(z)+K.epsilon()))(E_logit)
        
        R_logit = kl.Lambda(lambda z:z/temp)(R_logit)
        R = kl.Activation('sigmoid')(R_logit)
        
        
        mask = np.ones((1,self.num_states))
        mask[0,self.num_states-1]=0
        
        R = kl.Lambda(lambda z:mask*z)(R)
        Dense_W.trainable = False
        
        vs = []
        for i in range(T):
            #r = kl.Dot(axes=-1)([s,R])
            r = kl.Lambda(lambda z:K.sum(z[0]*z[1],axis=-1,keepdims=True))([s,R])
            v = kl.Lambda(lambda z:(self.gamma**i)*z)(r)
            vs.append(v)
            
            s = kl.Lambda(lambda z:(z[0]-z[1]*z[0])+(1-mask)*z[2])([s,R,r])
            
            logit_1 = Dense_W(s)
            logit = logit_1
            s = kl.Lambda(lambda z:K.abs(z)/K.sum(K.abs(z),axis=-1,keepdims=True))(logit)
            #s = kl.Lambda(lambda z:K.dot(z[0], z[1][0]))([s,W_effect])
            
            if i==0:
                P_0 = s

        v = kl.Add()(vs)
        
        print('v:',K.int_shape(v))
        
        return R,v

        

    def get_graph_edit_model(self): #Done
        if self.model_type == 'bootstrapped':
            return self.get_graph_edit_bootstrapped_model()
        elif self.model_type == 'fullydifferntiable':
            return self.get_graph_edit_fullydifferntiable_model()
        else:
            raise Exception('select model type')

    def get_graph_edit_fullydifferntiable_model(self): #Done
        
        sgs = []
        for _ in range(self.num_groups):
            sgs.append(kl.Input((self.num_states,)))
            
            
        #R = kl.Input((self.num_states,))
        temp = kl.Input((1,))


        def edit_bias_init(shape, dtype=None):
            return K.ones(shape)*self.edit_bias_init_mu
        
        Dense_R = kl.Dense(self.num_states, bias_initializer=edit_bias_init)
                
        
        vgs = []
        for i in range(self.num_groups):
            R,vg = self.get_fd_particle_type_sub_graph(sgs[i],self.dense_gs[i],Dense_R,temp,self.T)
            vgs.append(vg)

        vs = kl.Concatenate(name='value')(vgs)
        
        def iden_init(shape, dtype=None):
            return K.eye(self.num_states)
        
        Iden = kl.Dense(self.num_states, name='edit',kernel_initializer=iden_init,use_bias=False)
        R = Iden(R)
        Iden.trainable = False
        

        model = km.Model(inputs=sgs+[temp], outputs=[vs,R])
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
                'value':[self.flow_loss,self.fair_loss,self.diff_bw_groups,self.mean_value],
                'edit':[self.num_edits_exceeded]
            }
        )
        return model

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
                'value': [self.flow_loss, self.fair_loss]
            }
        )
        return model
    
    def num_edits_exceeded(self, y_true, y_pred):
        print('self.budget:',self.budget)
        edits_exceeded = K.clip(K.sum(y_pred, axis=(-1)) - self.budget, 0 , 1e20)
        return edits_exceeded
    
    def budget_loss(self, y_true, y_pred):
        edit_loss = self.num_edits_exceeded(y_true, y_pred)
        edit_loss = K.clip(edit_loss, self.constraint_epsilon[1] , 1e20) - self.constraint_epsilon[1]
        
        if 'squared' in self.constraint_scheme:
            edit_loss = self.param_schedule.edit_loss_mu * (edit_loss**2)
        elif 'lagrangian_augmentation' in self.constraint_scheme:
            edit_loss = self.param_schedule.edit_loss_mu * (edit_loss**2) + 2 * self.param_schedule.edit_loss_lambda * edit_loss
        
        return edit_loss * self.param_schedule.budget_loss_switch


    def diff_bw_groups(self, y_true, y_pred):
        N = K.int_shape(y_pred)[1]
        group_utilities = K.mean(y_pred,axis=0)
        mean_group_utilites = K.expand_dims(K.mean(group_utilities),axis=-1)
        diff_from_mean = K.abs(group_utilities-mean_group_utilites)
        diff_from_mean = K.sum(diff_from_mean)
        return diff_from_mean
            

    def fair_loss_1(self, y_true, y_pred):
        N = K.int_shape(y_pred)[1]
        group_utilities = K.mean(y_pred,axis=0)
        mean_group_utilites = K.expand_dims(K.mean(y_pred),axis=-1)
        fair_loss = K.abs(group_utilities-mean_group_utilites)
        N = K.int_shape(y_pred)[1]
        var_group_utilites = K.expand_dims(K.var(y_pred),axis=-1)
        if self.fair_loss_normalize_variance:
            variance = (K.sum(var_group_utilites)/N)**0.5
            fair_loss = fair_loss/variance
        
        return K.sum(fair_loss)
    
    def fair_loss(self, y_true, y_pred):
        fair_loss = K.sum(K.abs(y_pred - K.mean(y_pred,axis=-1,keepdims=True)),axis=-1)
        return fair_loss
    
    def fair_loss_spectral_norm(self, y_true, y_pred):
        fair_loss = K.max(y_pred,axis=-1)-K.min(y_pred,axis=-1)
        return fair_loss
           
    def weighted_fair_loss(self, y_true, y_pred): 
        #fair_loss = self.fair_loss_spectral_norm(y_true, y_pred)
        fair_loss = self.fair_loss(y_true, y_pred)
        fair_loss = K.clip(fair_loss,self.constraint_epsilon[0],1e20)-self.constraint_epsilon[0]
        
        if 'squared' in self.constraint_scheme:
            fair_loss = self.param_schedule.fair_loss_mu * (fair_loss**2)
        elif 'lagrangian_augmentation' in self.constraint_scheme:
            fair_loss = self.param_schedule.fair_loss_mu * (fair_loss**2) + 2 * self.param_schedule.fair_loss_lambda * fair_loss
        
        return fair_loss
        
    def erst_while_fair_loss(self, y_true, y_pred):
        return K.square(K.mean(y_pred[:, 0]) - K.mean(y_pred[:, 1]))
    
    def erst_while_fair_loss_take_2(self, y_true, y_pred):
        fair_loss = y_pred[:, 0] - y_pred[:, 1]
        if 'squared' in self.constraint_scheme:
            fair_loss = self.param_schedule.fair_loss_mu * K.square(fair_loss)
        elif 'lagrangian_augmentation' in self.constraint_scheme:
            fair_loss = self.param_schedule.fair_loss_mu * K.square(fair_loss) + 2 * self.param_schedule.fair_loss_lambda * K.abs(fair_loss)
        return fair_loss

    def mean_value(self,y_true, y_pred):
        return K.mean(y_pred)
    
    def flow_loss(self, y_true, y_pred):
        return -1 * (K.mean(y_pred, axis=-1))**self.main_obj_optimizer_p

    def fair_flow_loss(self, y_true, y_pred):
        return self.param_schedule.flow_loss_switch * self.flow_loss(y_true, y_pred) + self.param_schedule.fair_loss_switch * self.weighted_fair_loss(y_true, y_pred)

