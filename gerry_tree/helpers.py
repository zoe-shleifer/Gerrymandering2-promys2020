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

from typing import Dict
def read_file(path: str) -> Dict:
      return json.load(Path(path).open())


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
    return dist_dict,school_dict,block_data,G

def signed_color(score):
  if score >= 0: color = 'red'
  else: color = 'green'
  return colored('{:+7.2%}'.format(score),color)

def make_map(new_dist,DATA_DIR):
    bgs = []
    colors1 = plt.cm.tab20_r(np.linspace(0., 1, 128))
    colors2 = plt.cm.nipy_spectral_r(np.linspace(0, 1, 128))
    colors = np.vstack((colors1, colors2))
    mymap = mcolors.LinearSegmentedColormap.from_list(
        'my_colormap', colors)
    bg = gpd.read_file(
        f'{DATA_DIR}/pa_groups /tl_2010_42_bg10.shp').set_index('GEOID10')
    bg['new_dist'] = 0
    for k, val in new_dist.items():
        for v in val:
            bg['new_dist'].loc[v] = int(k)
            bgs.append(v)
    bg['new_dist'] = bg['new_dist'].astype('category')
    bg.loc[bgs].plot(column='new_dist', categorical=True, cmap=mymap)


def make_graph(new_dist,G,block_data):
    display(block_data['POS'])
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
    nx.draw(G, node_color=list(nx.get_node_attributes(G, "color").values()), pos=block_data['POS'].loc[list(G.nodes())], node_size=list(nx.get_node_attributes(G, 'size').values()))
