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


def read_file(path: str) -> Dict:
    return json.load(Path(path).open())


def random_spanning_tree(G):
    T = G.copy()
    rand = np.random.uniform(0, 1, T.size())
    for i, edge in enumerate(T.edges()):
        T.edges[edge]["weight"] = rand[i]
    return nx.minimum_spanning_tree(T)
# setup steps to deal with messy files


def cleanup(DATA_DIR):
    block_data = pd.read_csv(f'{DATA_DIR}/small_final_attributes')
    # {block group : {school: distance in seconds from block group to school}}
    dist_dict = read_file(f'{DATA_DIR}/distance_data.txt')
    # {school: [block groups in district]}
    school_dict = read_file(f'{DATA_DIR}/original_districts.txt')
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
    return dist_dict, school_dict, block_data, G


def redistrict_random(DATA_DIR, samples=100, steps=50, buffer=0, enrollment_err=10/11):
    dist_dict, original_school_dict, block_data, original_G = cleanup(DATA_DIR)
    school_dict = original_school_dict
    schoollist = school_dict.keys()
    tot_mino = block_data["MINO"].sum()
    tot_pop = block_data['HS_POP'].sum()
    target_ratio = tot_mino/tot_pop

    # score calculations (compactness is not implemented yet)
    def _calc_score(sch, district):
        pop = block_data['HS_POP'].loc[district].sum()
        mino = block_data['MINO'].loc[district].sum()
        # dem_score -> squared difference between minority ratio of district and overall minority ratio * district population
        dem_score = (abs((mino/pop) - target_ratio))**2*pop*100
        # trav_score -> total drive time for all students (max ensures that when time < buffer, the value is not negative)
        trav_score = sum([max([dist_dict[x][sch]-buffer, 0]) *
                          block_data['HS_POP'].loc[x] for x in district])
        return (dem_score, trav_score)
    # improvements is equivilent to steps, it counts the number of times we beat the current score (when we don't beat the score, the map doesn't change (assuming random == False))
    district_scores, dem_scores, trav_scores = [], [], []
    for i in tqdm(range(samples)):
        school_dict = original_school_dict
        G = original_G
        improvements = 0
        while improvements < steps:
            schl1 = choice(list(school_dict))
            neighbour_blocks = []
            for i in school_dict[schl1]:
                for block in G.neighbors(i):
                    if block not in school_dict[schl1]:
                        neighbour_blocks.append(block)
            try:
                block_choice = choice(neighbour_blocks)
            except:
                continue
            for i in school_dict:
                if block_choice in school_dict[i]:
                    schl2 = i
                    break
            district_pair = school_dict[schl1] + school_dict[schl2]
            # initialize districts and scores to the current partitionin
            subgraph_copy = nx.subgraph(G, district_pair).copy()
            # Have we gotten a better score?
            improved = False
            # assigns random weights to edges then picks minimum tree this is a random spanning tree (hopefully)
            for i in range(50):
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
                for i in range(len(school_path)):
                    # if we get past max. enrollment without a valid split, the tree cannot be broken
                    if split_pop*enrollment_err > first_enroll:
                        split = 0
                        break
                    if (pair_pop - split_pop)*enrollment_err < second_enroll and split_pop*enrollment_err < first_enroll:
                        split = i+1
                        break
                    # add the population of the next component in the path
                    split_pop += block_data['HS_POP'].loc[comps[i+1]].sum()
                if split == 0:
                    continue
                # we have a split, now we make districts and check scores
                district_1 = [y for x in [comps[i]
                                          for i in range(split)] for y in x]
                district_2 = [x for x in district_pair if x not in district_1]
                school_dict[schl1] = district_1
                school_dict[schl2] = district_2
                improvements += 1
                break
        scoring = {dist: _calc_score(
            dist, school_dict[dist]) for dist in schoollist}
        district_scores.append(scoring)
        dem_scores.append(
            np.round(sum([scoring[u][0] for u in schoollist])/tot_pop, 3))
        trav_scores.append(
            np.round(sum([scoring[u][1] for u in schoollist])/tot_pop, 3))
        return district_scores, dem_scores, trav_scores

    def read_file(path: str) -> Dict:
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
    dist_dict = read_file(f'{DATA_DIR}/distance_data.txt')
    # {school: [block groups in district]}
    school_dict = read_file(f'{DATA_DIR}/original_districts.txt')
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
    return dist_dict, school_dict, block_data, G


def redistrict_random(DATA_DIR, samples=100, steps=50, buffer=0, enrollment_err=10/11):
    dist_dict, original_school_dict, block_data, original_G = cleanup(DATA_DIR)
    school_dict = original_school_dict
    schoollist = school_dict.keys()
    tot_mino = block_data["MINO"].sum()
    tot_pop = block_data['HS_POP'].sum()
    target_ratio = tot_mino/tot_pop

    # score calculations (compactness is not implemented yet)
    def _calc_score(sch, district):
        pop = block_data['HS_POP'].loc[district].sum()
        mino = block_data['MINO'].loc[district].sum()
        # dem_score -> squared difference between minority ratio of district and overall minority ratio * district population
        dem_score = (abs((mino/pop) - target_ratio))**2*pop*100
        # trav_score -> total drive time for all students (max ensures that when time < buffer, the value is not negative)
        trav_score = sum([max([dist_dict[x][sch]-buffer, 0]) *
                          block_data['HS_POP'].loc[x] for x in district])
        return (dem_score, trav_score)
    # improvements is equivilent to steps, it counts the number of times we beat the current score (when we don't beat the score, the map doesn't change (assuming random == False))
    district_scores, dem_scores, trav_scores = [], [], []
    for i in tqdm(range(samples)):
        import ipdb; ipdb.set_trace()
        school_dict = original_school_dict
        G = original_G
        improvements = 0
        while improvements < steps:
            schl1 = choice(list(school_dict))
            neighbour_blocks = []
            for i in school_dict[schl1]:
                for block in G.neighbors(i):
                    if block not in school_dict[schl1]:
                        neighbour_blocks.append(block)
            try:
                block_choice = choice(neighbour_blocks)
            except:
                continue
            for i in school_dict:
                if block_choice in school_dict[i]:
                    schl2 = i
                    break
            district_pair = school_dict[schl1] + school_dict[schl2]
            # initialize districts and scores to the current partitionin
            subgraph_copy = nx.subgraph(G, district_pair).copy()
            # Have we gotten a better score?
            improved = False
            # assigns random weights to edges then picks minimum tree this is a random spanning tree (hopefully)
            first_enroll = block_data['enrollment'].loc[schl1]
            second_enroll = block_data['enrollment'].loc[schl2]
            pair_pop = block_data['HS_POP'].loc[district_pair].sum()
            for i in range(50):
                T = random_spanning_tree(subgraph_copy)

                # get the shortest path between schools, we will cut one of these edges
                school_path = list(nx.all_simple_paths(
                    T, source=schl1, target=schl2))[0]

                # trash all the school_path edges and collect the connected components
                # then we can go from one side of the path to the other and see which split gives balanced population

                T.remove_edges_from(nx.utils.pairwise(school_path))
                comps = [list(nx.node_connected_component(T, i)) for i in school_path]
                split_pop = block_data['HS_POP'].loc[comps[0]].sum()
                # check enrollment capacities
                for i in range(len(school_path)):
                    # if we get past max. enrollment without a valid split, the tree cannot be broken
                    if split_pop*enrollment_err > first_enroll:
                        split = 0
                        break
                    if (pair_pop - split_pop)*enrollment_err < second_enroll and split_pop*enrollment_err < first_enroll:
                        split = i+1
                        break
                    # add the population of the next component in the path
                    split_pop += block_data['HS_POP'].loc[comps[i+1]].sum()
                if split == 0:
                    continue
                district_1 =  [y for x in [comps[i] for i in range(split)] for y in x]
                school_dict[schl1] = district_1
                school_dict[schl2] = [x for x in district_pair if x not in district_1]
                improvements += 1
                break
        scoring = {dist: _calc_score(
            dist, school_dict[dist]) for dist in schoollist}
        district_scores.append(scoring)
        dem_scores.append(
            np.round(sum([scoring[u][0] for u in schoollist])/tot_pop, 3))
        trav_scores.append(
            np.round(sum([scoring[u][1] for u in schoollist])/tot_pop, 3))
    return district_scores, dem_scores, trav_scores
data_dir = '/Users/zoeshleifer/promys2020/lab/Gerrymandering2-promys2020/gerry_tree'
results1k = redistrict_random(data_dir,steps=30,samples=1000)
def pickle_save(obj, path):
    with open(path, "wb") as f:
        return pickle.dump(obj, f)

pickle_save(results1k,'random_results1k.pkl')