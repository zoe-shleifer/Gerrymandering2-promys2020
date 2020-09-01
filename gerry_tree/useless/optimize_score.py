from helpers import make_graph, make_map,cleanup,signed_color
import networkx as nx
from tqdm.notebook import tqdm
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

def random_spanning_tree(G):
  T = G.copy()
  rand = np.random.uniform(0, 1, T.size())
  for i, edge in enumerate(T.edges()):
      T.edges[edge]["weight"] = rand[i]
  return nx.minimum_spanning_tree(T)

def redistrict(DATA_DIR, steps=50, tree_loop=150, coefs = [1,1,1], buffer=0, random=False, enrollment_err=10/11, update_rate=5, multipliers=[126.5,1,300],pictures='map'):
    # dist_dict    -- {block group id: {school:distance from block to school}}
    # school_dict  -- {school's block group id: [ids of block groups assigned to school]}
    # target_ratio -- overall ratio of number of minority students to total population
    dist_dict,school_dict,block_data,G = cleanup(DATA_DIR)
    school_dict_r = {v:k for k,val in school_dict.items() for v in val}
    schoollist = school_dict.keys()
    tot_mino = block_data["MINO"].sum()
    tot_pop = block_data['HS_POP'].sum()
    target_ratio = tot_mino/tot_pop
    coef_sum = sum(coefs)
    mino_coef,trav_coef,comp_coef = (multipliers[i]*coefs[i]/coef_sum for i in coefs)
    # score calculations (compactness is not implemented yet)
    def _calc_score(sch, district):
        pop = block_data['HS_POP'].loc[district].sum()
        mino = block_data['MINO'].loc[district].sum()
        # dem_score -> squared difference between minority ratio of district and overall minority ratio * district population
        dem_score = (abs((mino/pop) - target_ratio))**2*pop*100
        # trav_score -> total drive time for all students (max ensures that when time < buffer, the value is not negative)
        trav_score = sum([max([dist_dict[x][sch]-buffer, 0]) *
                      block_data['HS_POP'].loc[x] for x in district])
        comp_score=0
        for i in G.edges():
            if school_dict_r[i[0]]!=school_dict_r[i[1]]:
                    comp_score += 20
        comp_score=comp_score/len(schoollist)
        score = mino_coef*(dem_score) + (trav_coef)*trav_score + (comp_coef)*comp_score
        return (dem_score, trav_score, comp_score, score)
    def make_graph_dict(new_dist):
      make_graph(new_dist,G,block_data)
    def make_map_dict(new_dist):
      make_map(new_dist,DATA_DIR)
    # {district: current scores}
    district_scores = {dist: _calc_score(dist, school_dict[dist]) for dist in schoollist}
    current_dem_score = np.round(sum([district_scores[u][0] for u in schoollist])/tot_pop,3)
    current_trav_score = np.round(sum([district_scores[u][1] for u in schoollist])/tot_pop,3)
    current_comp_score = np.round(sum([district_scores[u][2] for u in schoollist])/tot_pop,3)

    print('*'*91)
    print(
        f"|   ||TIME {'{:>19}'.format('||')} Diversity Score:  {'{:6.3f}'.format(current_dem_score)} || Mean Travel Time (sec): {'{:6.3f}'.format(current_trav_score)}|| Compactness Score: {'{:6.3f}'.format(current_comp_score)}|")
    print('|'+'*'*3+'||'+'*'*22+'||'+'*'*26+'||'+'*'*32+'|')

    start = timeit.default_timer()
    stop = timeit.default_timer()

    score_list,animation_list = [],[]
    frames = 0
    # improvements is equivilent to steps, it counts the number of times we beat the current score (when we don't beat the score, the map doesn't change (assuming random == False))
    def improvement_condition(scores,best_score):
        return scores[3]<best_score
    improvements = 0


    #while improvements < steps:
    from tqdm import tqdm
    for step in tqdm(range(10000)):
        if improvements >= steps: break
        # picking a random pair of neighboring districts
        # we pick a random edge of the quotient graph which has a node for each district and an edge if they are adj.
        # this is likely inefficient :)
        schl1 = choice(list(school_dict))
        neighbour_blocks = []
        for i in school_dict[schl1]:
            for block in G.neighbors(i):
                if block not in school_dict[schl1]:
                    neighbour_blocks.append(block)
        # check neighbors
        try:
            block_choice = choice(neighbour_blocks)
        except:
            continue
        schl2 = school_dict_r[block_choice]
        district_pair = school_dict[schl1] + school_dict[schl2]
        # initialize districts and scores to the current partitioning
        best_district_1 = school_dict[schl1]
        best_district_2 = school_dict[schl2]
        # idx_that_you_need_to_change_if_you_add_a_score_like_zoe_did_on_august_10 = 3
        best_score = district_scores[schl1][3] + district_scores[schl2][3]
        subgraph_copy = nx.subgraph(G, district_pair).copy()
        # counts the number of times we pick a tree without a good breaking (mostly for debugging)
        no_break = 0
        # Have we gotten a better score?
        improved = False
        # assigns random weights to edges then picks minimum tree this is a random spanning tree (hopefully)
        for tree_iters in range(tree_loop):
            T = random_spanning_tree(subgraph_copy)
            # get the shortest path between schools, we will cut one of these edges
            school_path = list(nx.all_simple_paths(T, source=schl1, target=schl2))[0]
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
            scores1 = _calc_score(schl1, district_1)
            scores2 = _calc_score(schl2, district_2)
            score_sums = np.sum(list(map(np.array,list(zip(scores1,scores2)))),1)

            # when random argument is True we don'toptimize score

            if improvement_condition(score_sums,best_score) or random:
                improved = True
                all_scores = (scores1,scores2)
                best_district_1 = district_1
                best_district_2 = district_2
                if random:
                    score_list.append(all_scores)

        frames += 1
        # if score decreased, we edit the current districting
        if improved:
            improvements += 1
            district_scores[schl1] = all_scores[0]
            district_scores[schl2] = all_scores[1]
            school_dict[schl1] = best_district_1
            school_dict[schl2] = best_district_2
            for block in best_district_1:
                school_dict_r[block] = schl1
            for block in best_district_2:
                school_dict_r[block] = schl2
            if frames % update_rate == 0:
                stop = timeit.default_timer()
                last_dem_score = current_dem_score
                last_trav_score = current_trav_score
                last_comp_score = current_comp_score
                current_dem_score = np.round(sum([district_scores[u][0] for u in schoollist])/tot_pop,3)
                current_trav_score = np.round(sum([district_scores[u][1] for u in schoollist])/tot_pop,3)
                current_comp_score = np.round(sum([district_scores[u][2] for u in schoollist])/tot_pop,3)
                dem_dif,trav_dif,comp_dif = (current_dem_score - last_dem_score)/current_dem_score, \
                (current_trav_score - last_trav_score)/current_trav_score, \
                (current_comp_score - last_comp_score)/current_comp_score
                print(
                    f"|{colored('{:03d}'.format(improvements),'cyan')}|| Iteration Time: {'{:03.2f}'.format(frames/(stop-start))} || Diversity Score: {signed_color(dem_dif)} || Mean Travel Time (sec): {signed_color(trav_dif)}||Compactness Score: {signed_color(comp_dif)}|")

    print('|'+'*'*3+'||'+'*'*22+'||'+'*'*26+'||'+'*'*32+'|')
    print(
    f"|   || Total Time: {'{:>8.1f}'.format(stop-start)} {'{:>2}'.format('||')}Diversity Score:   {'{:6.3f}'.format(current_dem_score)} || Mean Travel Time (sec): {'{:5.3f}'.format(current_trav_score)}|| Compactness Score: {'{:5.3f}'.format(current_comp_score)}|")
    print('*'*91)
    if pictures == 'map':
        fig = plt.figure(figsize=[20, 15])
        ax = make_map(school_dict,DATA_DIR)
    elif pictures == 'graph':
        fig, ax = plt.subplots(figsize=(10, 6))
        animator = animation.FuncAnimation(fig, make_graph_dict, frames=animation_list)
        rc('animation', html='html5')
        animator.save(f'{DATA_DIR}/graph_anim.gif', writer='imagemagick', fps=5)
    if not random:
        final_diversity = round(
            sum([district_scores[u][0] for u in schoollist])/tot_pop, 2)
        final_travel = round(sum([district_scores[u][1]
                                  for u in schoollist])/tot_pop, 2)
        return {'districting': school_dict, 'diversity': final_diversity, 'travel': final_travel}
    else:
        return school_dict,current_dem_score,current_trav_score,current_comp_score

data_dir = '/Users/zoeshleifer/promys2020/lab/Gerrymandering2-promys2020/gerry_tree'
redistrict(data_dir,steps=100,tree_loop=10,random=False)
