import networkx as nx
import pickle
from tqdm import tqdm
import numpy as np
import pandas as pd
import geopandas as gpd
from random import choice
from termcolor import colored
import timeit
import json
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from pathlib import Path
from typing import List, Dict

def _read_file(path: str) -> Dict:
      return json.load(Path(path).open())

def random_spanning_tree(G):
  T = G.copy()
  rand = np.random.uniform(0, 1, T.size())
  for i, edge in enumerate(T.edges()):
      T.edges[edge]["weight"] = rand[i]
  return nx.minimum_spanning_tree(T)
# setup steps to deal with messy files
def cleanup(DATA_DIR):
    block_data = pd.read_csv(f'{DATA_DIR}/small_final_attributes.csv')
    # {block group : {school: distance in seconds from block group to school}}
    dist_dict = _read_file(f'{DATA_DIR}/distance_data.txt')
    # {school: [block groups in district]}
    school_dict = _read_file(f'{DATA_DIR}/original_districts.txt')
    schoollist = school_dict.keys()

    # block_data is a table (pandas DataFrame) holding all the attributes of every block group and school (rows are block groups)
    # dealing with the messiness of pandas read_csv with list/tuple entries
    block_data['GEOID10'] = block_data['GEOID10'].astype('str')
    block_data = block_data.set_index('GEOID10')
    for j in range(len(block_data)):
        string_list = block_data['borders'].iloc[j].split(',')[1:-1]
        good_list = []
        for i in range(len(string_list)):
            if string_list[i][2:-1] in block_data.index:
                good_list.append(string_list[i][2:-1])
        block_data['borders'].iloc[j] = good_list
    string_list = list(block_data['POS'])
    for j in range(len(block_data)):
        good_list = string_list[j]
        good_list = good_list.split(',')
        for i in range(len(good_list)):
            good_list[i] = float(good_list[i][2:-2])
        block_data['POS'].iloc[j] = tuple(good_list)
    block_data['MINO'] = block_data['HS_POP'] - \
        (block_data['RRATIO']*block_data['HS_POP']).round()
    block_data['enrollment']['420454051001'] = 0
    block_data['is_school']['420454051001'] = False

    # connecting disconnected districts. GerryChains has a more elegant method
    G = nx.to_networkx_graph(block_data['borders'].to_dict())
    # TODO: automate
    G.add_edge('420293021012', '420293020002')
    G.add_edge('420293021012', '420293021011')
    G.add_edge('420293003012', '420293003022')
    G.add_edge('420454033003', '420454016001')
    G.add_edge('420454076006', '420454106024')
    G.add_edge('420454017001', '420454020001')
    return dist_dict,school_dict,block_data,G

def redistrict(DATA_DIR,big_loop=1, steps=50, tree_loop=150, mino_coef=.5, buffer=0, random=False, enrollment_err=10/11, update_rate=5, animation='map'):
    final_scores = []
    dist_dict,school_dict,block_data,G = cleanup(DATA_DIR)
    schoollist = school_dict.keys()
    tot_mino = block_data["MINO"].sum()
    tot_pop = block_data['HS_POP'].sum()
    target_ratio = tot_mino/tot_pop
    def _calc_score(sch, district):
        pop = block_data['HS_POP'].loc[district].sum()
        mino = block_data['MINO'].loc[district].sum()
        # dem_score -> squared difference between minority ratio of district and overall minority ratio * district population
        dem_score = (abs((mino/pop) - target_ratio))**2*pop*100
        # trav_score -> total drive time for all students (max ensures that when time < buffer, the value is not negative)
        trav_score = sum([max([dist_dict[x][sch]-buffer, 0]) *
                      block_data['HS_POP'].loc[x] for x in district])
        return (dem_score, trav_score)
    district_scores = {dist: _calc_score(dist, school_dict[dist]) for dist in schoollist}
    current_dem_score = np.round(sum([district_scores[u][0] for u in schoollist])/tot_pop,3)
    current_trav_score = np.round(sum([district_scores[u][1] for u in schoollist])/tot_pop,3)
    final_scores=[]
    frames = 0
    for k in tqdm(range(big_loop)):
        trees = 0
        valid = False
        while trees < steps:
            pair = list(zip(*choice(list(nx.quotient_graph(G,
                                                        list(map(set, school_dict.values()))).edges()))))[0]
            school_dict_r = {v: k for k, val in school_dict.items() for v in val}
            schl1,schl2 = school_dict_r[pair[0]], school_dict_r[pair[1]]
            district_pair = school_dict[schl1] + school_dict[schl2]
            # initialize districts and scores to the current partitioning
            best_district_1 = school_dict[schl1]
            best_district_2 = school_dict[schl2]
            subgraph_copy = nx.subgraph(G, district_pair).copy()
            # counts the number of times we pick a tree without a good breaking (mostly for debugging)
            # assigns random weights to edges then picks minimum tree this is a random spanning tree (hopefully)
            T = random_spanning_tree(subgraph_copy)
            
            # get the shortest path between schools, we will cut one of these edges
            school_path = list(nx.all_simple_paths(
            T, source=schl1, target=schl2))[0]

            # trash all the school_path edges and collect the connected components
            # then we can go from one side of the path to the other and see which split gives balanced population
            
            T.remove_edges_from(nx.utils.pairwise(school_path))
            comps = [list(nx.node_connected_component(T, i))
                    for i in school_path]
            split_pop = block_data['HS_POP'].loc[comps[0]].sum()
            first_enroll = block_data['enrollment'].loc[schl1]
            second_enroll = block_data['enrollment'].loc[schl2]
            pair_pop = block_data['HS_POP'].loc[district_pair].sum()
            # check enrollment capacities
            for j in range(len(school_path)):
                if split_pop*enrollment_err > first_enroll:
                    split = 0
                    break
                if (pair_pop - split_pop)*enrollment_err < second_enroll:
                    split = j+1
                    valid = True
                    trees +=1
                    break
                split_pop += block_data['HS_POP'].loc[comps[j+1]].sum()
            if not valid:
                continue
            district_1 = [y for x in [comps[i]
                                    for i in range(split)] for y in x]
            district_2 = [x for x in district_pair if x not in district_1]
            dem_score1, trav_score1 = _calc_score(schl1, district_1)
            dem_score2, trav_score2 = _calc_score(schl2, district_2)
            all_scores = ((dem_score1, trav_score1),(dem_score2, trav_score2))
            district_scores[schl1] = all_scores[0]
            district_scores[schl2] = all_scores[1]
            school_dict[schl1] = district_1
            school_dict[schl2] = district_2


        current_dem_score = np.round(sum([district_scores[u][0] for u in schoollist])/tot_pop,6)
        current_trav_score = np.round(sum([district_scores[u][1] for u in schoollist])/tot_pop,6)
        final_scores.append((district_scores, current_dem_score, current_trav_score))
    return final_scores
data_dir = '/Users/zoeshleifer/promys2020/lab/Gerrymandering2-promys2020/gerry_tree'

def pickle_save(obj, path):
    with open(path, "wb") as f:
        return pickle.dump(obj, f)


new_results = (redistrict(data_dir,big_loop=1000,steps=30,mino_coef=1,tree_loop=1,random=True))
pickle_save(new_results,'redistrict_random_results30.pkl')
