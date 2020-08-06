from tqdm import tqdm
import networkx as nx
import numpy as np
import pandas as pd
import time
from random import choice
import timeit
import json
import matplotlib.pyplot as plt
from pathlib import Path
from typing import List, Dict


def read_file(path:str) -> Dict:
    return json.load(Path(path).open())
def redistrict(DATA_DIR,steps=50,tree_loop=150,mino_coef=.5,buffer = 0,random=False):
    attrs = pd.read_csv('/Users/zoeshleifer/promys2020/lab/small_final_attributes')
    attrs['GEOID10'] = attrs['GEOID10'].astype('str')
    attrs = attrs.set_index('GEOID10')
    for j in range(len(attrs)):
        list1 = attrs['borders'].iloc[j].split(',')[1:-1]
        list2 = []
        for i in range(len(list1)):
                if list1[i][2:-1] in attrs.index:
                    list2.append(list1[i][2:-1])
        attrs['borders'].iloc[j] = list2
    attrs['MINO'] = attrs['HS_POP'] - (attrs['RRATIO']*attrs['HS_POP']).round()
    attrs['enrollment']['420454051001'] = 0
    attrs['is_school']['420454051001'] = False

    R = nx.to_networkx_graph(attrs['borders'].to_dict())
    R.add_edge('420293021012','420293020002')
    R.add_edge('420293021012','420293021011')
    R.add_edge('420293003012', '420293003022') 
    R.add_edge('420454033003', '420454016001')
    R.add_edge('420454076006', '420454106024')
    R.add_edge('420454017001', '420454020001')
    dist_dict = read_file(f'{DATA_DIR}/distance_data.txt')
    school_dict = read_file(f'{DATA_DIR}/original_districts.txt')

    
    enrollment_err = 5/6
    update_rate = 5
    tot_mino = attrs["MINO"].sum()
    tot_pop  = attrs['HS_POP'].sum()
    schoollist = school_dict.keys()
    average_racial=tot_mino/tot_pop
    nx.set_node_attributes(R,100,"size")
    nx.set_node_attributes(R,'#9400D3',"color")
    for u in schoollist:
        R.nodes[u]["size"]=300

    def calc_score(sch,district):
        pop = attrs['HS_POP'].loc[district].sum()
        mino = attrs['MINO'].loc[district].sum()
        mscore = (abs((mino/pop) - average_racial)*100)**2*pop
        tscore = sum([max([dist_dict[x][sch]-buffer,0])*attrs['HS_POP'].loc[x] for x in district])
        score = mino_coef*(mscore) + (1-mino_coef)*tscore
        return (mscore,tscore,score)
    district_scores = {dist : calc_score(dist,school_dict[dist]) for dist in schoollist}
    print(f"initial scores: M.score: {sum([district_scores[u][0] for u in schoollist])/tot_pop} T.score: {sum([district_scores[u][1] for u in schoollist])/tot_pop}")


    start = timeit.default_timer() 
    stop = timeit.default_timer()
    bad_schools = []
    frames = 0
    wins = 0
    while wins < steps:
        pair = list(zip(*choice(list(nx.quotient_graph(R, list(map(set,school_dict.values()))).edges()))))[0]
        school_dict_r = {v:k for k,val in school_dict.items() for v in val}
        schs = [school_dict_r[pair[0]],school_dict_r[pair[1]]]
        district_pair = school_dict[schs[0]] + school_dict[schs[1]]
        best_district_1 = school_dict[schs[0]]
        best_district_2 = school_dict[schs[1]]
        best_score = calc_score(schs[0],school_dict[schs[0]])[2] + calc_score(schs[1],school_dict[schs[1]])[2]
        subgraph_copy = nx.subgraph(R,district_pair).copy()
        no_break = 0
        win = 0
        for tree_iters in range(tree_loop):
            T = subgraph_copy.copy()
            rand = np.random.uniform(0, 1, T.size())
            for i,edge in enumerate(T.edges()):
                T.edges[edge]["weight"] = rand[i]
            T = nx.minimum_spanning_tree(T)
            try:
                school_path = list(nx.all_simple_paths(T, source=schs[0], target=schs[1]))[0]
            except:
                bad_schools.append(schs)
                print(f'bad! {bad_schools}')
                break
            T.remove_edges_from(nx.utils.pairwise(school_path))
            comps = [list(nx.node_connected_component(T,i)) for i in school_path]
            split_pop = attrs['HS_POP'].loc[comps[0]].sum()
            first_enroll = attrs['enrollment'].loc[schs[0]]
            second_enroll = attrs['enrollment'].loc[schs[1]]
            pair_pop = attrs['HS_POP'].loc[district_pair].sum()
            for i in range(len(school_path)):
                if split_pop*enrollment_err > first_enroll:
                    split = 0
                    break
                if (pair_pop - split_pop)*enrollment_err < second_enroll:
                    split = i+1
                    break
                split_pop += attrs['HS_POP'].loc[comps[i+1]].sum()
            if split == 0: no_break+=1 ; continue
            district_1 = [y for x in [comps[i] for i in range(split)] for y in x]
            district_2 = [x for x in district_pair if x not in district_1]
            mscore1,tscore1,score1 = calc_score(schs[0],district_1)
            mscore2,tscore2,score2 = calc_score(schs[1],district_2)
            score = score1 + score2
        if not random:
            if score < best_score:
                win = 1
                all_scores = ((mscore1,tscore1,score1),(mscore2,tscore2,score2))
                best_district_1 = district_1
                best_district_2 = district_2
        else: all_scores.append()        
            
        frames += 1
        wins += win
        if win:
            district_scores[schs[0]] = all_scores[0]
            district_scores[schs[1]] = all_scores[1]
            school_dict[schs[0]] = best_district_1
            school_dict[schs[1]] = best_district_2
            # region timer end
        if frames%update_rate == 0 and wins>0:
            stop = timeit.default_timer()
            print(f"{wins} --  Speed: {np.round((frames/(stop-start)),2)} M: {sum([district_scores[u][0] for u in schoollist])/tot_pop} T: {sum([district_scores[u][1] for u in schoollist])/tot_pop}")

redistrict("/Users/zoeshleifer/promys2020/lab",steps=500,mino_coef=1,tree_loop=50)

# kwarg: individual upper-bound vs. overall upperbound
# 