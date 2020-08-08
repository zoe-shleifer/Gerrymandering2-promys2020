from tqdm import tqdm
import networkx as nx
import numpy as np
import pandas as pd
import geopandas as gpd
import time
from random import choice
import timeit
import json
import matplotlib.pyplot as plt
from pathlib import Path
from typing import List, Dict
from matplotlib import colors as mcolors

NOT_A_SCHOOL = '420454051001'
def _read_file(path: str) -> Dict:
    return json.load(Path(path).open())


def get_val_list(G, attr_name):
    return list(nx.get_node_attributes(G, attr_name).values())

def redistrict(DATA_DIR, steps=50, tree_loop=150, mino_coef=.5, buffer=0, random=False, enrollment_err=5/6, update_rate=5, pictures='map'):
    """Document kwargs"""
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
    
    block_data.loc[NOT_A_SCHOOL, 'enrollment'] = 0
    block_data.loc[NOT_A_SCHOOL, 'is_school'] = False
    # connecting disconnected districts. GerryChains has a more elegant method
    G = nx.to_networkx_graph(block_data['borders'].to_dict())
    # TODO: automate
    G.add_edge('420293021012', '420293020002')
    G.add_edge('420293021012', '420293021011')
    G.add_edge('420293003012', '420293003022')
    G.add_edge('420454033003', '420454016001')
    G.add_edge('420454076006', '420454106024')
    G.add_edge('420454017001', '420454020001')

    # prettiness, colors, more colors, etc.
    if pictures == 'map':
        colors1 = plt.cm.tab20_r(np.linspace(0., 1, 128))
        colors2 = plt.cm.nipy_spectral_r(np.linspace(0, 1, 128))
        colors = np.vstack((colors1, colors2))
        mymap = mcolors.LinearSegmentedColormap.from_list(
            'my_colormap', colors)

        def _make_map(new_dist):
            bg = gpd.read_file(
                f'{DATA_DIR}/pa_groups /tl_2010_42_bg10.shp').set_index('GEOID10')
            bg['new_dist'] = 0
            for k, val in new_dist.items():
                for v in val:
                    bg['new_dist'].loc[v] = int(k)
            bg['new_dist'] = bg['new_dist'].astype('category')
            return bg.loc[list(G.nodes())].plot(column='new_dist', categorical=True, cmap=mymap)

    elif pictures == 'graph':
        def _make_graph(new_dist):
            # thanks steven for color and size setup
            color_choice = ["#C0C0C0", '#808080', '#FF0000', '#800000', '#FFFF00', '#808000',
                            '#00FF00', '#008000', '#00FFFF', '#008080', '#0000FF', '#000080', '#800080']
            nx.set_node_attributes(G, 100, "size")
            nx.set_node_attributes(G, '#9400D3', "color")
            for i, school in enumerate(new_dist):
                for u in new_dist[school]:
                    G.nodes[u]["color"] = color_choice[i % 13]
                G.nodes[school]["color"] = "#FF00FF"
                G.nodes[school]["size"] = 300
            display(nx.get_node_attributes(G, "color").values())
            return nx.draw(G, node_color=get_val_list(G, 'color') ,
            pos=block_data['POS'].loc[list(G.nodes())], 
            node_size=get_val_list(G, 'size'))


# sum columns of block_data to get total population
    tot_mino = block_data["MINO"].sum()
    tot_pop = block_data['HS_POP'].sum()
    average_racial = tot_mino/tot_pop

    # score calculations (compactness is not implemented yet)
    def _calc_score(sch, district):
        pop = block_data['HS_POP'].loc[district].sum()
        mino = block_data['MINO'].loc[district].sum()
        # mscore -> squared difference between minority ratio of district and overall minority ratio * district population
        mscore = (abs((mino/pop) - average_racial))**2*pop*100
        # tscore -> total drive time for all students (max ensures that when time < buffer, the value is not negative)
        tscore = sum([max([dist_dict[x][sch]-buffer, 0]) *
                      block_data['HS_POP'].loc[x] for x in district])
        score = mino_coef*(mscore) + (1-mino_coef)*tscore
        return (mscore, tscore, score)
    # {district: current scores}
    district_scores = {dist: _calc_score(
        dist, school_dict[dist]) for dist in schoollist}
    current_mscore = np.round(sum([district_scores[u][0] for u in schoollist])/tot_pop,3)
    current_tscore = np.round(sum([district_scores[u][1] for u in schoollist])/tot_pop,3)
    print(
        f"initial scores: Diversity Score: {current_mscore} Mean Travel Time (sec): {current_tscore}")

    start = timeit.default_timer()
    stop = timeit.default_timer()
    # if this isn't empty, it will be printed and there is a bug, this happens when there are no paths in a tree
    bad_schools = []
    score_list = []
    frames = 0
    # wins is equivilent to steps, it counts the number of times we beat the current score
    wins = 0
    while wins < steps:
        # picking a random pair of neighboring districts
        # we pick a random edge of the quotient graph which has a node for each district and an edge if they are adj.
        # this is likely inefficient :)
        pair = list(zip(*choice(list(nx.quotient_graph(G,
                                                       list(map(set, school_dict.values()))).edges()))))[0]
        school_dict_r = {v: k for k, val in school_dict.items() for v in val}
        # schs has length 2 it holds the two schools in the district
        schs = [school_dict_r[pair[0]], school_dict_r[pair[1]]]
        district_pair = school_dict[schs[0]] + school_dict[schs[1]]
        # initialize districts and scores to the current partitioning
        best_district_1 = school_dict[schs[0]]
        best_district_2 = school_dict[schs[1]]
        best_score = district_scores[schs[0]][2] + district_scores[schs[1]][2]
        subgraph_copy = nx.subgraph(G, district_pair).copy()
        # counts the number of times we pick a tree without a good breaking (mostly for debugging)
        no_break = 0
        # Have we won?
        improved = False
        # assigns random weights to edges then picks minimum tree this is a random spanning tree (hopefully)
        for tree_iters in range(tree_loop):
            T = subgraph_copy.copy()
            rand = np.random.uniform(0, 1, T.size())
            for i, edge in enumerate(T.edges()):
                T.edges[edge]["weight"] = rand[i]
            T = nx.minimum_spanning_tree(T)
            # get the shortest path between schools, we will cut one of these edges
            try:
                school_path = list(nx.all_simple_paths(
                    T, source=schs[0], target=schs[1]))[0]
            # big sad (this really shouldn't happen)
            except:
                bad_schools.append(schs)
                print(f'bad! {bad_schools}')
                break
            # trash all the school_path edges and collect the connected components
            # then we can go from one side of the path to the other and see which split gives balanced population
            T.remove_edges_from(nx.utils.pairwise(school_path))
            comps = [list(nx.node_connected_component(T, i))
                     for i in school_path]
            split_pop = block_data['HS_POP'].loc[comps[0]].sum()
            first_enroll = block_data['enrollment'].loc[schs[0]]
            second_enroll = block_data['enrollment'].loc[schs[1]]
            pair_pop = block_data['HS_POP'].loc[district_pair].sum()
            # check enrollment capacities
            for i in range(len(school_path)):
                # if we get past max. enrollment without a valid split, the tree cannot be broken
                if split_pop*enrollment_err > first_enroll:
                    split = 0
                    break
                if (pair_pop - split_pop)*enrollment_err < second_enroll:
                    split = i+1
                    break
                # add the population of the next component in the path
                split_pop += block_data['HS_POP'].loc[comps[i+1]].sum()
            if split == 0:
                no_break += 1
                continue
            # we have a split, now we make districts and check scores
            district_1 = [y for x in [comps[i]
                                      for i in range(split)] for y in x]
            district_2 = [x for x in district_pair if x not in district_1]
            mscore1, tscore1, score1 = _calc_score(schs[0], district_1)
            mscore2, tscore2, score2 = _calc_score(schs[1], district_2)
            score = score1 + score2
        # when random argument is True this step will be skipped
            if score < best_score or random:
                improved = True
                all_scores = ((mscore1, tscore1, score1),
                              (mscore2, tscore2, score2))
                best_district_1 = district_1
                best_district_2 = district_2
                if random:
                    score_list.append(all_scores)

        frames += 1
        # if we won, we edit the current districting
        if improved:
            wins += 1
            district_scores[schs[0]] = all_scores[0]
            district_scores[schs[1]] = all_scores[1]
            school_dict[schs[0]] = best_district_1
            school_dict[schs[1]] = best_district_2

        if frames % update_rate == 0 and wins > 0:
            stop = timeit.default_timer()
            past_mscore = current_mscore
            past_tscore = current_tscore
            current_mscore = np.round(sum([district_scores[u][0] for u in schoollist])/tot_pop,3)
            current_tscore = np.round(sum([district_scores[u][1] for u in schoollist])/tot_pop,3)

            print(
                f"{wins} --  Iteration Time: {np.round((frames/(stop-start)),2)} -- Diversity Score: {current_mscore - past_mscore} -- Mean Travel Time (sec): {current_mscore - past_mscore}")
    if pictures == 'map':
        fig = plt.figure(figsize=[20, 15])
        ax = _make_map(school_dict)
    elif pictures == 'graph':
        fig = plt.figure(figsize=[20, 15])
        ax = fig.add_subplot(1, 1, 1)
        ax = _make_graph(school_dict)
    if not random:
        final_diversity = round(
            sum([district_scores[u][0] for u in schoollist])/tot_pop, 2)
        final_travel = round(sum([district_scores[u][1]
                                  for u in schoollist])/tot_pop, 2)
        return {'districting': school_dict, 'diversity': final_diversity, 'travel': final_travel}
    else:
        return {'last_step': school_dict,
                'all_scores': score_list}

import fire
if __name__ == '__main__':
    fire.Fire(redistrict)