import geopandas as gpd
import collections as c
import shapely
import fiona
import pandas as pd
import os
import joblib as jl
import re
import numpy as np
import itertools as it
from bs4 import BeautifulSoup
from haversine import haversine, Unit, haversine_vector
from paths_inc import *
import ujson
import requests
from geographiclib.geodesic import Geodesic

data_path = os.path.join(proj_root ,"data" ,"networks")
school_path = os.path.join(proj_root,"data" ,"schools")
demographics_path = os.path.join(proj_root ,"data" ,"demographics")
route_file = os.path.join(data_path, "CTA_BusRoutes.kml")
stop_file = route_file = os.path.join(data_path, "CTA_BusStops.kml")
lines_cache_path = os.path.join(proj_root ,"data" ,"networks", "lines")
school_file = os.path.join(school_path, "schools.csv")
demo_file = os.path.join(demographics_path, "DEC_10_SF1_QTP3", "DEC_10_SF1_QTP3.csv")
tract_file = os.path.join(demographics_path, "CensusTractsTIGER2010.csv")

out_nodes = os.path.join(data_path, "chicago_nodes_new.gz")
out_edges = os.path.join(data_path, "chicago_edges_new.gz")
out_tracts = os.path.join(data_path, "chicago_tracts_new.gz")
out_edges_kml = os.path.join(data_path, "chicago_edges_new.kml")
out_nodes_kml = os.path.join(data_path, "chicago_nodes_new.kml")

demo_keys = {"total":'Number; RACE - Total population - One race', "white": "Number; RACE - Total population - One race - White", "black": "Number; RACE - Total population - One race - Black or African American", "latino": "Number; HISPANIC OR LATINO - Total population"}

gpd.io.file.fiona.drvsupport.supported_drivers['KML'] = 'rw'
fiona.drvsupport.supported_drivers['LIBKML'] = 'rw'
df = gpd.read_file(route_file, driver='KML')

def demo_field(v):
    if "Cook County" in v:
        return float(re.search(r'\d*\.?\d+', v).group())
    else:
        return np.nan


def get_segments_from_api():
    key = "YCyFCeYT6Dy7GCRdMXG6ExRvM"
    intersections = c.defaultdict(list)

    url = "http://ctabustracker.com/bustime/api/v2/getroutes?key={key}&format=json".format(key=key)
    routes = ujson.loads(requests.get(url).text)

    for r in routes["bustime-response"]["routes"]:
        rt = r["rtdd"]
        url = "http://www.ctabustracker.com/bustime/api/v2/getpatterns?key={key}&rt={rt}&format=json".format(key=key,
                                                                                                             rt=rt)
        patterns = ujson.loads(requests.get(url).text)

        if "ptr" not in patterns["bustime-response"]:
            print("Error: {rt}".format(rt=rt))
        else:
            for pat in patterns["bustime-response"]["ptr"]:
                pat_list = []
                for pp in pat["pt"]:
                    if "stpnm" in pp:
                        pat_list.append(map_namestring(pp["stpnm"]))
                intersections[rt].append(pat_list)
    return intersections


def map_namestring(rr):
    return tuple(set([x.strip() for x in rr.split("&")]))
def get_routes_from_file():
    f = open(os.path.join(lines_cache_path, "line_list.txt")).read()
    soup = BeautifulSoup(f)
    test = [x.split(' ', 1)[0] for x in soup.get_text().splitlines()]

    routes = c.defaultdict(list)
    intersections = c.defaultdict(list)
    for t in test:

        soup = BeautifulSoup(open(os.path.join(lines_cache_path, "stoplist_" + t + ".htm")), features="lxml")
        rows = soup.find('table').find_all("tr")
        seen = set()
        for i, tr in enumerate(rows):
            if i > 0:
                td = tr.find_all('td')
                row = [i.text for i in td]
                #print(row)
                nn = map_namestring(row[1])
                if nn not in seen:
                    routes[t].append(row[0])
                    intersections[t].append(nn)
                    seen.add(nn)
    return routes, intersections

def get_field_data(v, demo_keys):
    r = {}
    for k, kk in demo_keys.items():
        r[k] = v[kk]
    r["key"] = demo_field(v["Geography"])
    return r

def join_demographics(demo_file, tract_file):
    dff_demo = pd.read_csv(demo_file, header=1)
    dff_tract = pd.read_csv(tract_file)
    dff_tract['the_geom'] = dff_tract['the_geom'].apply(shapely.wkt.loads)
    dff_tract = gpd.GeoDataFrame(dff_tract, geometry='the_geom')
    dff_tract.rename(columns={'NAME10': 'key'}, inplace=True)
    dff_tract.rename(columns={'the_geom': 'geometry'}, inplace=True)

    d = pd.DataFrame({i: get_field_data(v, demo_keys) for i, v in dff_demo.iterrows()}).transpose()

    return gpd.GeoDataFrame(pd.merge(dff_tract, d, how="left", on=["key", "key"]))


def index_nodes(dff_demo, dff_nodes):
    col = []
    for i, node in dff_nodes.iterrows():
        n = np.nonzero([node["geometry"].within(v["geometry"]) for i, v in dff_demo.iterrows()])
        if len(n[0])==1:
            vv = n[0][0]
        else:
            vv = np.nan
        col.append(vv)
    dff_nodes["tracts"] = col
    return dff_nodes

def get_points(object):
    if isinstance(object, shapely.geometry.point.Point):
        return [object]
    elif isinstance(object, shapely.geometry.MultiPoint):
        return [v for v in object]
    elif isinstance(object, shapely.geometry.linestring.LineString) or isinstance(object, shapely.geometry.multilinestring.MultiLineString):
        return []
    # elif isinstance(object, shapely.geometry.collection.GeometryCollection):
    #     r = []
    #     for v in object:
    #         r.extend(get_points(v))
    #     return r
    else:
        return []

def get_edges(dff, num_edges=2, th = 0.0005):
    edges = []
    for i, a in dff.iterrows():
        k = a["name"]

        dists = {v["name"]: a["geometry"].distance(v["geometry"]) for i, v in dff.iterrows() if v["name"][0] in set(k[0:2]) and v["name"][1] not in set(k[0:2])}
        sorted_d = [v for v in sorted(dists.items(), key=lambda kv: kv[1]) if v[1] > th]


        inds_add = set()
        for kk, vv in sorted_d[0:num_edges]:
            edges.append([k, kk, vv])
            # if kk[0:2] not in inds_add:
            #     inds_add.add(kk[0:2])
            #     edges.append([k, kk, vv])
            # if len(inds_add) == 2:
            #     break

        dists = {v["name"]: a["geometry"].distance(v["geometry"]) for i, v in dff.iterrows() if v["name"][1] in set(k[0:2]) and v["name"][0] not in set(k[0:2])}
        sorted_d = [v for v in sorted(dists.items(), key=lambda kv: kv[1]) if v[1] > th]

        inds_add = set()
        for kk, vv in sorted_d[0:num_edges]:
            edges.append([k, kk, vv])
            # if kk[0:2] not in inds_add:
            #     inds_add.add(kk[0:2])
            #     edges.append([k, kk, vv])
            # if len(inds_add) == 2:
            #     break
        print(i)
    return edges

def window(seq, n=2):
    "Returns a sliding window (of width n) over data from the iterable"
    "   s -> (s0,s1,...s[n-1]), (s1,s2,...,sn), ...                   "
    itt = iter(seq)
    result = tuple(it.islice(itt, n))
    if len(result) == n:
        yield result
    for elem in itt:
        result = result[1:] + (elem,)
        yield result



def build_nodes_new(df):
    points = c.defaultdict(list)
    for i, vi in df.iterrows():
        points[map_namestring(vi["Name"])].append(vi["geometry"])
    return {k:shapely.geometry.MultiPoint(v).centroid for k,v in points.items()}

def get_edges_new(nodes, th=np.inf):
    edges = {}

    intersections = get_segments_from_api()
    nodes_dict = {v["name"]: v["geometry"] for i, v in nodes.iterrows()}
    #nodes_vs = [(v.y, v.x) for v in nodes["geometry"]]

    for k_line, k_inter_list in intersections.items():
        for seg in k_inter_list:
            for v1, v2 in window(seg):
                if v1 in nodes_dict and v2 in nodes_dict and v1 != v2 and (v1, v2) not in edges:
                    dist = haversine((nodes_dict[v1].y, nodes_dict[v1].x), (nodes_dict[v2].y, nodes_dict[v2].x), unit=Unit.METERS)
                    if dist < th:
                        edges[(v1, v2)] = dist
    return [[k[0], k[1], v] for k,v in edges.items()]

    # for i, p in enumerate(nodes_vs):
    #     dists_iter = haversine_vector(list(it.repeat(p, len(nodes_vs))), nodes_vs, Unit.METERS)
    #     inds_iter = np.argsort(dists_iter)[1:k+1]
    #     for jj in inds_iter:
    #         if dists_iter[jj] < th:
    #             edges.append([nodes_dict[i], nodes_dict[jj], dists_iter[jj]])
    # for k_line, k_inter in intersections.items():
    #     for v1, v2 in window(k_inter):
    #         if v1 in nodes_dict and v2 in nodes_dict and v1 != v2:
    #             dist = haversine((nodes_dict[v1].y, nodes_dict[v1].x), (nodes_dict[v2].y, nodes_dict[v2].x), unit=Unit.METERS)
    #             if dist < th:
    #                 edges.append([v1, v2, dist])
    #                 edges.append([v2, v1, dist])


def filter_nodes(dff_nodes, edges):
    e_nodes = set()
    for v1,v2, w in edges:
        e_nodes.add(v1)
        e_nodes.add(v2)
    s =pd.Series([v in e_nodes for v in dff_nodes["name"]])
    dff_nodes = dff_nodes[s.values]
    return dff_nodes.reset_index(drop=True)

def build_nodes(df, line_interact_th=5):
    ret = {}
    inds = c.defaultdict(int)

    uids = {i: v["Name"] for i, v in df.iterrows()}
    uids_rev = {v: k for k, v in uids.items()}

    for i, vi in df.iterrows():
        for j, vj in df.iterrows():
            vik = uids_rev[vi["Name"]]
            vjk = uids_rev[vj["Name"]]
            if vik < vjk and not vi["Name"].startswith("X") and not vj["Name"].startswith("X") :
                test = get_points(vi["geometry"].intersection(vj["geometry"]))
                print(test)
                if len(test) <= line_interact_th:
                    for vv in test:
                        inds[(vik, vjk)] += 1
                        ret[(vik, vjk, inds[(vik, vjk)])] = vv
    return ret

def build_graph(dff_nodes, edges, out=False):
    uids = {i: v["name"] for i, v in dff_nodes.iterrows()}
    uids_rev = {v: k for k, v in uids.items()}
    keys = []
    weights = []
    geometries = []
    for e in edges:
        n1 = dff_nodes.iloc[uids_rev[e[0]]]
        n2 = dff_nodes.iloc[uids_rev[e[1]]]
        if out:
            keys.append(str(n1["name"]) + "-" + str(n2["name"]))
        else:
            keys.append((uids_rev[e[0]], uids_rev[e[1]]))
        weights.append(e[2])
        geometries.append(shapely.geometry.LineString([n1["geometry"], n2["geometry"]]))
    return gpd.GeoDataFrame({'name': keys, 'weight': weights}, geometry=geometries)


def add_nodes_new(dff_nodes, dff_edges, dff_schools):

    uid_i = np.max([v[0] for v in dff_edges["name"]] + [v[1] for v in dff_edges["name"]])+1
    for i, r in dff_schools.iterrows():
        p = shapely.geometry.point.Point([r["lat"], r["long"]])
        dff_nodes = dff_nodes.append({"name": r["name"], "geometry": p}, ignore_index=True)

        dists = {j:haversine((p.y, p.x), (p2["geometry"].y, p2["geometry"].x), unit=Unit.METERS) for j, p2 in dff_nodes.iterrows()}

        sorted_d = sorted(dists.items(), key=lambda kv: kv[1])
        inds = sorted_d[1:3]
        v1 = inds[0][0]
        v2 = inds[1][0]

        e1 = shapely.geometry.LineString((p, dff_nodes.iloc[v1]["geometry"]))
        e2 = shapely.geometry.LineString((dff_nodes.iloc[v1]["geometry"], p))
        e3 = shapely.geometry.LineString((p, dff_nodes.iloc[v2]["geometry"]))
        e4 = shapely.geometry.LineString((dff_nodes.iloc[v2]["geometry"], p))
        dff_edges = dff_edges.append({"name": (uid_i, v1), "geometry": e1, "weight": e1.length }, ignore_index=True)
        dff_edges = dff_edges.append({"name": (v1, uid_i), "geometry": e2, "weight": e2.length}, ignore_index=True)
        dff_edges = dff_edges.append({"name": (uid_i, v2), "geometry": e3, "weight": e3.length}, ignore_index=True)
        dff_edges = dff_edges.append({"name": (v2, uid_i), "geometry": e4, "weight": e4.length}, ignore_index=True)
        uid_i += 1
    return dff_nodes, dff_edges



def add_nodes(dff_nodes, dff_edges, dff_schools):

    uid_i = np.max([v[0] for v in dff_edges["name"]] + [v[1] for v in dff_edges["name"]])+1
    for i, r in dff_schools.iterrows():
        p = shapely.geometry.point.Point([r["lat"], r["long"], 0])
        dff_nodes = dff_nodes.append({"name": (uid_i, uid_i, -1), "geometry": p}, ignore_index=True)

        dists = {j:p.distance(r_node["geometry"]) for j, r_node in dff_nodes.iterrows()}
        sorted_d = sorted(dists.items(), key=lambda kv: kv[1])
        inds = sorted_d[0:2]
        v1 = inds[0][0]
        v2 = inds[1][0]

        e1 = shapely.geometry.LineString((p, dff_nodes.iloc[v1]["geometry"]))
        e2 = shapely.geometry.LineString((dff_nodes.iloc[v1]["geometry"], p))
        e3 = shapely.geometry.LineString((p, dff_nodes.iloc[v2]["geometry"]))
        e4 = shapely.geometry.LineString((dff_nodes.iloc[v2]["geometry"], p))
        dff_edges = dff_edges.append({"name": (uid_i, v1), "geometry": e1, "weight": e1.length }, ignore_index=True)
        dff_edges = dff_edges.append({"name": (v1, uid_i), "geometry": e2, "weight": e2.length}, ignore_index=True)
        dff_edges = dff_edges.append({"name": (uid_i, v2), "geometry": e3, "weight": e3.length}, ignore_index=True)
        dff_edges = dff_edges.append({"name": (v2, uid_i), "geometry": e4, "weight": e4.length}, ignore_index=True)
        uid_i += 1
    return dff_nodes, dff_edges

ret = build_nodes_new(df)
dff_nodes = gpd.GeoDataFrame({'name': [i for i in list(ret.keys())]}, geometry=list(ret.values()))
dff_nodes_out = gpd.GeoDataFrame({'name': [str(i) for i in list(ret.keys())]}, geometry=list(ret.values()))

print("Unfiltered:" + str(len(dff_nodes)))
edges = get_edges_new(dff_nodes)
dff_nodes = filter_nodes(dff_nodes, edges)
print("Filtered:"+ str(len(dff_nodes)))

dff_edges = build_graph(dff_nodes, edges, out=False)
dff_schools = pd.read_csv(school_file)
dff_nodes, dff_edges = add_nodes_new(dff_nodes, dff_edges, dff_schools) #attach school nodes
dff_tract = join_demographics(demo_file, tract_file)   #join demographics and tract
dff_nodes = index_nodes(dff_tract, dff_nodes) #find the tract that I index into
dff_edges_out = build_graph(dff_nodes, edges, out=True) #for kmz

with fiona.drivers():
    # Might throw a WARNING - CPLE_NotSupported in b'dataset sample_out.kml does not support layer creation option ENCODING'
    jl.dump(dff_edges, out_edges)
    jl.dump(dff_nodes, out_nodes)
    jl.dump(dff_tract, out_tracts)
    dff_edges_out.to_file(out_edges_kml, driver='KML')
    dff_nodes_out.to_file(out_nodes_kml, driver='KML')
