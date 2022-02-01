from tensorflow.keras import backend as K

def get_fair_loss(config):
    def fair_loss(y_true, y_pred):
        v = y_pred
        vgs = K.mean(v,axis=-2)
        vg_mean = K.mean(vgs,axis=-1)
        vgs_var = K.var(vgs,axis=-1)        
        
        v_var = K.var(v,axis=-2)
        v_var_mean = K.mean(v_var,axis=-1)
        
        v_mean =  K.mean(v,axis=(-1,-2),keepdims=True)
        
        print('vgs:{},vg_mean:{}'.format(vgs,vg_mean))
        
        if config['fair_loss_error']=='mae':
            loss = K.sum(K.abs(vg_mean-vgs))
        elif config['fair_loss_error']=='mae2':
            print('v:',K.int_shape(v))
            print('v_mean:',K.int_shape(v_mean))
            loss = K.abs(v_mean-v)
            print('loss:',K.int_shape(loss))
            loss = K.mean(loss,axis=-1)
            print('loss:',K.int_shape(loss))
        elif config['fair_loss_error']=='mae3':
            loss = K.sum(K.abs(vg_mean-vgs)) + K.sum(K.abs(v_var_mean-v_var)) * config['var_coeff']
        elif config['fair_loss_error']=='mape':
            loss = K.sum(K.abs(vg_mean-vgs)/vg_mean)
        elif config['fair_loss_error']=='rmape':
            loss = K.sum(K.abs(vg_mean-vgs)/vgs)
        elif config['fair_loss_error']=='smape':
            loss = K.sum(K.abs(vg_mean-vgs)/K.abs(vg_mean+vgs))
        elif config['fair_loss_error']=='zscore':
            loss = K.sum(K.abs(vg_mean-vgs)/K.abs(vgs_var))
        else:
            raise Exception('Invalid selection for fair_loss_error:',config['fair_loss_error'])
        
        return loss
    return fair_loss



def diff_bw_groups(y_true, y_pred):
    N = K.int_shape(y_pred)[1]
    group_utilities = K.mean(y_pred,axis=0)
    mean_group_utilites = K.expand_dims(K.mean(group_utilities),axis=-1)
    diff_from_mean = K.abs(group_utilities-mean_group_utilites)
    diff_from_mean = K.sum(diff_from_mean)
    return diff_from_mean

def get_num_edits_exceeded(budget):
    def num_edits_exceeded(y_true, y_pred):
        edits_exceeded = K.clip(K.mean(y_pred) - budget, 0 , 1e20)
        return edits_exceeded
    return num_edits_exceeded
        
def mean_value(y_true, y_pred):
    return K.mean(y_pred)

def get_flow_loss(main_obj_optimizer_p):
    def flow_loss(y_true, y_pred):
        return -1 * K.pow((K.mean(y_pred, axis=-1)),main_obj_optimizer_p)
    return flow_loss      
