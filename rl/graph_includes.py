import numpy.random as npr
import numpy as np
import networkx as nx
import copy
import scipy.sparse as scs
import collections as c
import joblib as jl
import random
import time
import glob
import os
import scipy.io as sio
import igraph

def weight_identity(x):
    return 1
def uniform_weight(G, x, y, w=1):
    return w
def weight_edge(x):
    return 1 / (.00000001 + x)
def weight_inv_degree(G,x,y):
    return 1 / (G.degree[x] * G.degree[y])
def  weight_degree(G,x,y):
    return G.degree[x] * G.degree[y]
#add particles to graph
def add_particles(G, N, color_p=0.5, start_dist=None):
    inds = npr.choice(list(range(G["N"])), size=N,p=start_dist)
    for j,jj in enumerate(inds):
        #initialize state =0, binary color sampled from color_p probability
        G["particles"][jj].append({"id":j, "color": int(npr.binomial(1, size=1, p=color_p)), "state": 0})
    return G


#wrapper for inner dictionary
def init_graph(N, adj_dim=2):
    l = [scs.dok_matrix((N, N)) for _ in range(adj_dim)]
    ret = {"adjs" : l, "particles" : c.defaultdict(list), "immunized": set(), "N":N}
    return ret

#wrapper to translate networkx to dict (assumes undirected)
def networkx_wrapper(G, fn_edges=[lambda G, x,y: npr.exponential(2), lambda G, x,y: np.abs(npr.normal(0,1))], directed=True, self=False, aux = {}):


    ret = init_graph(np.max(G.nodes)+1, len(fn_edges))
    if self:
        for v in G.nodes:
            for li, f_i in zip(ret["adjs"], fn_edges):
                li[v, v] = f_i(G, v, v, **aux)

    for v1, v2 in G.edges:
        for li, f_i in zip(ret["adjs"], fn_edges):
            li[v1, v2] = f_i(G, v1, v2, **aux)
            if not directed:
                li[v2,v1] = li[v1,v2]
            else:
                li[v2,v1] = f_i(G, v1, v2, **aux)
    return ret

#build erdos reyni graph with N nodes at p edge probability
def build_er_graph(N=100, p=.1, fn_edges=[lambda G, x,y: npr.exponential(2), lambda G, x,y: np.abs(npr.normal(0,1))], directed=True, self=False):
    return networkx_wrapper(nx.erdos_renyi_graph(N, p), fn_edges=fn_edges, directed=directed, self=self )

#build standard karate club network
def build_karate_club():
    return networkx_wrapper(nx.karate_club_graph())

#build preferential attachment graph with N nodes and k attachment edges
def build_ba_graph(N=100, k=3, p=0.1, fn_edges=[lambda G, x,y: npr.exponential(2), lambda G, x,y: np.abs(npr.normal(0,1))], directed=True, self=False):
    return networkx_wrapper(nx.powerlaw_cluster_graph(N, k, p), fn_edges=fn_edges, directed=directed, self=self )

def build_kregular_graph(N=100, k=3, fn_edges=[lambda G, x,y: npr.exponential(2), lambda G, x,y: np.abs(npr.normal(0,1))], directed=True, self=False):
    return networkx_wrapper(nx.random_regular_graph(k, N), fn_edges=fn_edges, directed=directed, self=self )

def build_sbm_graph(sizes=[50, 50], p= [[0.9, 0.1], [0.1, 0.9]],fn_edges=[lambda G, x,y: npr.exponential(2), lambda G, x,y: np.abs(npr.normal(0,1))], directed=True, self=False):
    # if fn_edges is None:
    #     #edge functions which index into the the p matrix
    #     fn_edges = [lambda x,y,sizes, p: match_inds(x,y,sizes, p, 0) , lambda x,y,sizes, p: match_inds(x,y,sizes, p, 1)]
    # if fn_edges == "fixed":
    #     fn_edges = [lambda x,y,sizes, p: 0.5, lambda x,y,sizes, p:0.5]
    return networkx_wrapper(nx.stochastic_block_model(sizes=sizes, p=p, directed=directed, selfloops=self), fn_edges=fn_edges, directed=directed, self=self)


def build_fb_graph(d, fn_edges=[uniform_weight, uniform_weight], definition="upper bias", samples=5, major_th=0.05):

    A = nx.DiGraph(d["A"].todense())
    years = d["local_info"][:, 5]
    men_idx = d["local_info"][:, 1] == 2

    years_clean = [x for x, y in zip(*np.unique(years, return_counts=True)) if y > len(years)*.1 and x != 0]
    years_idx = years == np.nanmax(years_clean) - 4
    #inds = np.where(years == np.nanmax(years) - 4)[0]
    majors = [i for i in np.unique(d["local_info"][:, 2]) if i != 0]
    if definition == "major bias":
        scores = {i: np.sum(np.bitwise_and(d["local_info"][:, 2] == i, men_idx)) / np.sum(d["local_info"][:, 2] == i) for i in majors if
               np.sum(d["local_info"][:, 2] == i) > len(years)*major_th}
        bias_key, bias_value = sorted(scores.items(), key=lambda x: x[1])[::-1][0]
        inds_select = list(np.where(np.bitwise_and(np.bitwise_and(years_idx, men_idx), d["local_info"][:, 2] == bias_key)[0]))
    elif definition == "upper bias":
        rewards = {i:(np.sum(d["local_info"][list(A.neighbors(i)), 1] == 2))/(len(list(A.neighbors(i)))) for i in np.where(np.bitwise_and(years_idx, men_idx))[0] if len(list(A.neighbors(i)))> 30}
        inds_select = [x[0] for x in sorted(rewards.items(), key=lambda x: x[1])[::-1]][0:min(len(rewards), samples )]
    elif definition == "upper men":
        inds_select = list(np.where(np.bitwise_and(years_idx, men_idx))[0])[0:min(len(years_idx), samples)]


    #inds_select = list(npr.permutation(inds)[0:int(np.floor(len(inds)*im_frac))])
    return networkx_wrapper(A, self=False, directed=True, fn_edges=fn_edges), inds_select

def build_chunglu_graph(N, exp_val,fn_edges=[lambda G, x,y: npr.exponential(2), lambda G, x,y: np.abs(npr.normal(0,1))], directed=True, self=False):
    A = nx.expected_degree_graph([np.round(v) for v in truncated_power_law(N, exp_val)], selfloops=self)
    A.remove_nodes_from(list(nx.isolates(A)))
    A = nx.relabel_nodes(A, {k:v for k,v in zip(A.nodes, range(len(A.nodes)))})
    A = nx.DiGraph(A)

    return networkx_wrapper(A, self=self, directed=directed, fn_edges=fn_edges)

def geo_edge_map(G,x,y,weight_dict, weight_fn):
    return weight_fn(weight_dict[(x,y)])

def build_dataframe_graph(df, node_from="START_NODE", node_to="END_NODE", weight_key="LENGTH", fn_edges=[geo_edge_map, geo_edge_map], weight_fn=lambda x: 1/np.exp(x), directed=True, self=False):
    weights = list(df[weight_key])
    edges = [(int(r[1][node_from]), int(r[1][node_to])) for r in df.iterrows()]
    weights_dict = {k: v for k, v in zip(edges, weights)}
    return networkx_wrapper(nx.DiGraph(list(edges)), fn_edges=fn_edges, directed=directed, self=self,
                            aux={"weight_fn": weight_fn, "weight_dict": weights_dict})

def networkx_compress_labels(g):
    inds = {y: x for x, y in zip(list(range(len(g.nodes))), np.sort(g.nodes))}
    return nx.relabel.relabel_nodes(g, inds)

def build_geo_graph(df_edges, edge_key="name", weight_key="weight", fn_edges=[geo_edge_map, geo_edge_map], weight_fn=lambda x: 1/np.exp(x), directed=True, self=False):
    edges = df_edges[edge_key]
    weights = df_edges[weight_key]
    weights_dict = {k:v for k,v in zip(edges, weights)}
    return networkx_wrapper(nx.DiGraph(list(edges)), fn_edges=fn_edges, directed=directed, self=self, aux={"weight_fn": weight_fn, "weight_dict": weights_dict})

def filter_geo_data(df, bb, get_index=False, reindex=True):
    if get_index:
        idx = np.array([bb.contains(e) for e in df["geometry"]])
        inds = np.where(idx)[0]
        index = {y:x for x,y in zip(range(len(inds)), inds)}
        return df.drop(df[np.bitwise_not(idx)].index, inplace=False).reset_index(drop=True),index
    else:
        r = df.drop(df[[not bb.contains(e) for e in df["geometry"]]].index, inplace=False)
        if reindex:
            return r.reset_index(drop=True)
        else:
            return r


def map_edges(df, index):
    r = []
    for e in df["name"]:
        if e[0] in index and e[1] in index:
            r.append((index[e[0]], index[e[1]]))
        else:
            r.append((np.nan, np.nan))
    df.name = r
    return df

def truncated_power_law(size,exp_val, min_degree=1):
    a = [float(i)**(-exp_val) for i in range(min_degree, size)]
    return npr.choice(list(range(min_degree, size)), size, p=a/np.sum(a))



#run simulation one step, send particles to neighboring nodes with uniform 1-stay_p probability
def propagate_step(G):
    ret_part = c.defaultdict(list)
    for k,v in G["particles"].items():
        for vv in v:
            inds, weights = G["adjs"][vv["color"]][k].nonzero()[1], np.abs(list(G["adjs"][vv["color"]][k].values()))
            to_node = int(npr.choice(inds, p=weights/np.sum(weights), size=1)[0])
            if k in G["immunized"]:
                vv["state"] = 1
            ret_part[to_node].append(vv)
    return ret_part


def immunization_perf(G):
    rets = {"red imm": 0, "black imm": 0, "red miss":0, "black miss":0 }
    for k,v in G["particles"].items():
        for vv in v:
            if vv["state"]:
                if vv["color"]:
                    rets["red imm"] += 1
                else:
                    rets["black imm"] += 1
            else:
                if vv["color"]:
                    rets["red miss"] += 1
                else:
                    rets["black miss"] += 1
    return rets

def normalize_graph(g):
    g1 = g.sum(axis=1, keepdims=True)
    g1[np.isnan(g1)] = 1
    g1[g1 == 0] = 1
    return g / g1
def immunize(g, nodes=[0,1]):
    g = np.append(g, np.zeros([1, g.shape[1]]), axis = 0)
    g = np.append(g, np.zeros([g.shape[0], 1]), axis = 1)
    for i in list(nodes) + [g.shape[0]-1]:
        g[i, :] = 0
        g[i, -1] = 1
    return g

def degree_scaled(degrees, fn=lambda x:x):
    return np.array([fn(s) for s in degrees])/np.sum(degrees)
def match_inds(x,y, sizes, p,s):
    ind1 = np.digitize(x, [0] + list(np.cumsum(sizes))) - 1
    ind2 = np.digitize(y, [0] + list(np.cumsum(sizes))) - 1
    if ind1 == ind2:
        if s != ind1:
            ind1 += 1
    return p[abs(ind1 - ind2)]

def get_graph(params={}, model="ba", im={"random": 4}, mask_params={"order": 2}):
    if model == "ba":
        g = build_ba_graph(**params)
    elif model == "er":
        g = build_er_graph(**params)
    elif model == "karate":
        g = build_karate_club()
    elif model == "sbm":
        g = build_sbm_graph(**params)
    elif model == "geo":
        g = build_geo_graph(**params)
    elif model == "cl":
        g = build_chunglu_graph(**params)
    elif model == "kregular":
        g = build_kregular_graph(**params)
    elif model == "fb":
        g, im = build_fb_graph(**params)
    # elif model == "hardcode":
    #     Pr, Pb, mask = get_hardcoded_graph()
    #     return Pr, Pb, mask
    #
    degrees = np.array([len(s.nonzero()[1]) for s in g["adjs"][0]]) #assumes red and black have the same degrees

    if "random" in im:
        im = npr.choice(list(range(g["adjs"][0].shape[0])), im["random"])
    elif "fixed" in im:
        im = im["fixed"]
    elif "lambda:degree" in im:
        if "subset" in im["lambda:degree"]:
            nodes = im["lambda:degree"]["subset"]
            degrees = degrees[nodes]
        else:
            nodes = list(range(g["adjs"][0].shape[0]))
        if "ascending" in im["lambda:degree"] and im["lambda:degree"]["ascending"]:
            degrees = 1 / degrees
        p = im["lambda:degree"]["fn"](degrees, **im["lambda:degree"]["fn params"])  #pulled this out for readability because it's pretty nutty
        im = npr.choice(nodes, p=p, size=im["lambda:degree"]["count"], replace=False ) #this is tricky, this is a function on degree sequence
    elif "degree-desc" in im:
        im = np.argsort(degrees)[::-1][0:im["degree-desc"]]
    elif "degree-acc" in im:
        im = np.argsort(degrees)[0:im["degree-asc"]]
    elif "geo" in im:
        df_nodes = im["geo"]["df_nodes"]
        nodes = [i for i, v in enumerate(df_nodes["name"]) if not isinstance(v, tuple)]
        if "count" in im:
            im = npr.choice(nodes, size=min(len(nodes), im["geo"]["count"]), replace=False)
        else:
            im = nodes


    prs = [immunize(normalize_graph(gg.toarray()), im) for gg in g["adjs"]]

    im2 = np.append(im, [len(prs[0])-1])
    #Pr = immunize(normalize_graph(g["adjs"][0].toarray()), im).tolist()
    #Pb = immunize(normalize_graph(g["adjs"][1].toarray()), im).tolist()
    #if "large mask" in im:

    #else:
    #    mask = get_graph_mask(prs[0], im2)
#    if model in ["geo"]:
    mask = get_graph_mask_neigh(prs[0], **mask_params)
    # else:
    #     mask = get_graph_mask_full(prs[0], im, **mask_params)
    return prs, mask, im

def get_graph_mask(a, im):
    a = np.array(a)
    a[im, :] = np.nan
    a[a > 0] = 1
    a[np.isnan(a)] = 0
    return np.array(a, dtype=int).tolist()
def get_graph_mask_neigh(a, zero_last=True, order=2):
    a = np.array(a)
    if zero_last:
        a[:, -1] = 0
    g = igraph.Graph.Adjacency(a.astype(bool).tolist())
    g2 = igraph.Graph(directed=True)
    g2.add_vertices(g.vs.indices)
    neighs = g.neighborhood(g.vs.indices, order=order, mode="out", mindist=2)
    es = [(j, i) for j, neigh in zip(g.vs.indices, neighs) for i in neigh]
    g2.add_edges(es)
    return g2.get_adjacency_sparse().toarray()

def get_graph_mask_full(a, im= [], zero_last = True):
    a = np.array(a)
    aa = np.ones(a.shape) - np.eye(a.shape[0], a.shape[1])
    if zero_last:
        aa[:, -1] = 0
        aa[-1, :] = 0
    aa[im, :] = 0
    for x,y in zip(*np.nonzero(a)):
        aa[x,y] = 0
    return aa

def fb_clean_files(files):
    ret =[]
    for f in files:
        d = sio.loadmat(f)
        if np.sum(d["local_info"][:, 1] == 1) > 100 and np.sum(d["local_info"][:, 1] == 2) > 100:
            ret.append(f)
    return ret
def get_hardcoded_graph():
    Pr = [[0.8, 0.1, 0.1, 0.],
          [0., 0., 0., 1.],
          [0.8, 0.1, 0.1, 0.],
          [0., 0., 0., 1.]]

    Pb = [[0.1, 0.8, 0.1, 0.],
          [0., 0., 0., 1.],
          [0.1, 0.8, 0.1, 0.],
          [0., 0., 0., 1.]]

    mask = [[1, 1, 1, 0],
            [0, 0, 0, 0],
            [1, 1, 1, 0],
            [0, 0, 0, 0]]  # TODO: I dont understand this mask, is this thresholding non-zeros? why is row 2 and 4 different than Pb Pr??
    return Pr, Pb, mask

def random_walk(g, start, reward, max_walk=10, walks=1000): ####NP.ARRAY right now for speed. gets wrecked under scipy sparse
    if isinstance(g, list):
        g = np.array(g)
    if not len(g[start].nonzero()[0]):
        return [], []

    reward = set(reward)
    reward_ret = []
    lengths_ret = []

    for _ in range(walks):
        r = 0
        ret_iter = []
        i = start
        for _ in range(max_walk):
            ret_iter.append(i)
            if i in reward:
                r = 1
                break
            else:
                neigh = g[i].nonzero()[0]
                if len(neigh):
                    i = random.choices(neigh, weights=g[i][neigh], k= 1)[0]
                else:
                    break
        reward_ret.append(r)
        lengths_ret.append(len(ret_iter))
    return reward_ret, lengths_ret

def graph_random_walks(g, reward, starts, max_walk=10, walks=1000 ,fn= lambda x: np.mean(x), fn_2= lambda x:np.mean(x)):
    ret_walk = [random_walk(g, i, reward, max_walk=max_walk, walks=walks) for i in starts]
    return {"measures":np.array([fn(v[0]) for v in ret_walk]), "paths":np.array([fn(v[1]) for v in ret_walk])}

def permute(x, y):
    new = np.concatenate([x,y])
    np.random.shuffle(new)
    return new[:len(x)], new[len(x):]

def parallel_handler(net_r, rewards, starts_r, max_walk, walks, fn, test_key):
    return graph_random_walks(net_r, rewards, starts_r, max_walk, walks, fn)

def gini(array):
    """Calculate the Gini coefficient of a numpy array."""
    # based on bottom eq: http://www.statsdirect.com/help/content/image/stat0206_wmf.gif
    # from: http://www.statsdirect.com/help/default.htm#nonparametric_methods/gini.htm
    array = array.flatten() #all values are treated equally, arrays must be 1d
    if np.amin(array) < 0:
        array -= np.amin(array) #values cannot be negative
    array += 0.0000001 #values cannot be 0
    array = np.sort(array) #values must be sorted
    index = np.arange(1,array.shape[0]+1) #index per array element
    n = array.shape[0]#number of array elements
    return ((np.sum((2 * index - n  - 1) * array)) / (n * np.sum(array))) #Gini coefficient

def evaluate_model_stat_replicates(nets, rewards, fn_replicates, params_replicates, replicates=100, rand_iters=5000,  walk_iters=100, max_walk=10, walk_fn= lambda x: np.mean(x), test_key= "reward", test_type="both", test_fn= lambda x,y: np.nanmean(y) - np.nanmean(x)):
    t = time.time()

    paths = []
    values = []
    for net, p in zip(nets, params_replicates):
        params = []

        starts = fn_replicates(**p)
        #starts_r = fn_replicates(**params_replicates_r)
        #params.append((net, rewards, starts, max_walk, walk_iters, walk_fn, test_key))
        ret = parallel_handler(net, rewards, starts, max_walk, walk_iters, walk_fn, test_key)
        #ret = jl.Parallel(n_jobs=8)(jl.delayed(parallel_handler)(*v) for v in params)
        values.append(ret["measures"])
        paths.append(ret["paths"])

        #[float(vv["path lengths"]) for vv in ret])
        #params = []
    # for i in range(replicates):
    #     starts_b = fn_replicates(**params_replicates_b)
    #     params.append((net_b, rewards, starts_b, max_walk, walk_iters, walk_fn, test_key))
    #
    # ret = jl.Parallel(n_jobs=8)(jl.delayed(parallel_handler)(*v) for v in params)
    # b = [float(vv[test_key]) for vv in ret]
    # b_paths = [float(vv["path lengths"]) for vv in ret]

    # test = test_fn(r, b)
    # null_dist = np.array([test_fn(*permute(r, b)) for _ in range(rand_iters)])
    #
    # if test_type == "both":
    #     p_value = np.sum(np.abs(null_dist) > np.abs(test)) / len(null_dist)
    #     test = np.abs(test)
    # elif test_type == "+":
    #     p_value = np.sum(null_dist > test) / len(null_dist)
    # elif test_type == '-':
    #     p_value = np.sum(null_dist < test) / len(null_dist)
    return {'p': np.nan, "measures": values, 'test val': np.nan, 'null': [], 'paths': paths}
def shortest_paths(gs, im, node_samples, full=False):
    ret = []
    if not isinstance(gs, igraph.Graph):
        g = igraph.Graph.Adjacency(np.array(gs).astype(bool).tolist())
    else:
        g = gs
    uniques = np.array(np.unique(np.concatenate([np.array(v) for v in node_samples])), dtype=np.int)
    paths_idx = {k:v for k,v in zip(uniques, g.shortest_paths(uniques, target=im))}
    for ss in node_samples:
        paths = [paths_idx[s] for s in ss]
        if full:
            ret.append(paths)
        else:
            m = np.array([np.min(p) for p in paths], dtype=np.float)
            m[np.isinf(m)] = np.nan
            ret.append(m)
    return ret

def shortest_paths_edge_augment(g, im, locs, es):
    r = {}
    for e in es:
        g[e] = 1
        r[e] = np.nanmean(shortest_paths(g, im, locs, full=False)[0])
        g[e] = 0
    return r


def do_edge_add_heuristic(gs, im, locs, k, edge_k, A):
    if not isinstance(gs, igraph.Graph):
        gs = igraph.Graph.Adjacency(np.array(gs).astype(bool).tolist())
    ret = []
    for i in range(k):
        tt = time.time()
        ind = np.argmax([np.nanmean(v) for v in shortest_paths(gs, im, locs)])
        vs = npr.choice(locs[ind], min(edge_k, len(locs[ind])), replace=False)
        es = [e for e in [(v, npr.choice(np.where(A[v])[0] if np.any(A[v]) else [np.nan] )) for v in vs] if not np.isnan(e[1])]
        r = shortest_paths_edge_augment(gs, im, [locs[ind]], es)
        v = sorted(r.items(), key=lambda x: x[1])[0]
        ret.append([ind, v[0], v[1]])
        gs[ret[-1][1]] = 1
        print("Baseline (progress, time): " + str((str(i/k), time.time() - tt)))
    return gs, ret

# def evaluate_model_stat_test(gs, starts, rewards, rand_iters=10000, walk_iters=5000, max_walk=10, walk_fn= lambda x: np.mean(x), test_key= "reward", test_type="both" , test_fn= lambda x,y: np.nanmean(y) - np.nanmean(x)):
#
#     measures = [graph_random_walks(g, rewards, s, max_walk=max_walk, walks=walk_iters, fn=walk_fn) for g,s in zip(gs, starts)]
#     # test = test_fn(meas_r[test_key], meas_b[test_key])
#     # null_dist = np.array([test_fn(*permute(meas_r[test_key], meas_b[test_key])) for _ in range(rand_iters)])
#     # if test_type == "both":
#     #     p_value = np.sum(np.abs(null_dist) > np.abs(test))/len(null_dist)
#     # elif test_type == "+":
#     #     p_value = np.sum(null_dist > test) / len(null_dist)
#     # elif test_type == '-':
#     #     p_value = np.sum(null_dist < test) / len(null_dist)
#     return {'p': np.null, 'measures':measures  , 'test val': np.nan, 'null dist': [] }


def bfs(network, seed, until=None):
    to_visit = [seed]
    curr = 1
    prev = {}
    while (curr == 1 or len(to_visit)):
        next_visit = set()
        for s in to_visit:
            if s in network:
                neigh = list(network[s])
                for v in neigh:
                    if v not in prev:
                        next_visit.add(v)
                        prev[v] = s
                        if v in until:
                            r = [v]
                            while True:
                                r.insert(0, prev[r[0]])
                                if r[0] == seed:
                                    return r
        curr += 1
        to_visit = next_visit
    return []

def get_particle_paths(g, particles, rewards):
    ret = c.defaultdict(int)
    for p_iter in particles:
        path = bfs(g, p_iter, rewards)
        for v1, v2 in zip(path, path[1:]):
            ret[(v1, v2)] += 1
    return ret

def select_nodes(gs, fn_particles, particle_params, rewards, budget, replicates=20):
    gs = [np.array(gg) for gg in gs]
    w_edit = np.nanmax([np.nanmax(gg) for gg in gs])
    g_binary = {k: list(np.nonzero(v)[0]) for k, v in enumerate(gs[0])}
    paths = [c.defaultdict(int) for _ in gs]

    #for _ in range(replicates):
    particles = [fn_particles(**s) for s in particle_params]
        #particles_a = fn_particles(**particle_params_a)
        #particles_b = fn_particles(**particle_params_b)
    p_iters = [get_particle_paths(g_binary, ss, rewards) for ss in particles]
        #a_iter = get_particle_paths(g_binary, particles_a, rewards)
        #b_iter  = get_particle_paths(g_binary, particles_b, rewards)

    for i, p in enumerate(p_iters):
        for k,v in p.items():
            paths[i][k] += v
        # for k,v in b_iter.items():
        #     path_b[k] += v
    cs = [np.zeros((len(gs[0]), len(gs[0]))) for _ in gs]
    #cr = np.zeros((len(g_r), len(g_r)))
    for i, pp in enumerate(paths):
        for p, w in pp.items():
            cs[i][p[0], p[1]] = w
    # cb = np.zeros((len(g_b), len(g_b)))
    # for p, w in path_b.items():
    #     cb[p[0], p[1]] = w
    edits = {}
    for i, r in zip(range(len(gs)) , [(np.nan_to_num(normalize_graph(cr) - g_r)) for cr, g_r in zip(cs, gs)]):
        rr = r.nonzero()
        for x, y in zip(rr[0], rr[1]):
            edits[(x, y, i)] = r[x, y]
    sorted_x = sorted(edits.items(), key=lambda kv: kv[1])[::-1]
    edits = [np.zeros((len(gs[0]), len(gs[0]))) for _ in gs]
    #r_edits = np.zeros((len(gs[0]), len(gs[0])))
    #b_edits = np.zeros((len(gs[0]), len(gs[0])))
    for i, v in zip(range(budget), sorted_x):
        x,y, color = v[0]
        edits[color][x,y] = 1
        gs[color][x,y] += w_edit
    return [normalize_graph(gg) for gg in gs], edits

def get_performance(results_root):
    files = sorted(glob.glob(os.path.join(results_root, "*.gz")))
    a = {}
    for f in files:
        d = jl.load(f)
        a[f] = {"budget": d["config"]["budget"], "value_loss": d["history"]["value_loss"][-1], "exceeded": d["history"]["edit_num_edits_exceeded"][-1], "value_fair_loss": d["history"]["value_fair_loss"][-1],  "table": d["df"]}
    return a
def get_performance_files(results_root):
    files = sorted(glob.glob(os.path.join(results_root, "*.gz")))
    a = {}
    for f in files:
        d = jl.load(f)
        a[f] = {"budget": d["config"]["budget"], "value_loss": d["history"]["value_loss"][-1], "exceeded": d["history"]["edit_num_edits_exceeded"][-1], "value_fair_loss": d["history"]["value_fair_loss"][-1],  "table": d["df"]}
    return a
def get_files(results_root, budget=50):
    files = sorted(glob.glob(os.path.join(results_root, "*.gz")))
    a = {}
    for f in files:
        d = jl.load(f)
        if d["config"]["budget"]==budget:
            a[d["config"]["method"]] = d
        #a[f] = {"budget": d["config"]["budget"], "value_loss": d["history"]["value_loss"][-1], "exceeded": d["history"]["edit_num_edits_exceeded"][-1], "value_fair_loss": d["history"]["value_fair_loss"][-1],  "table": d["df"]}
    return a