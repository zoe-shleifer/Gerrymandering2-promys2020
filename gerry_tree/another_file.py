def read_file(path: str) -> Dict:
      return json.load(Path(path).open())


def signed_color(score):
  if score >= 0: color = 'red'
  else: color = 'green'
  return colored('{:+7.2%}'.format(score), color)


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


def redistrict(DATA_DIR, steps=50, tree_loop=150, mino_coef=.5, buffer=0, random=False, enrollment_err=5/6, update_rate=5, pictures='map'):
    # dist_dict    -- {block group id: {school:distance from block to school}}
    # school_dict  -- {school's block group id: [ids of block groups assigned to school]}
    # target_ratio -- overall ratio of number of minority students to total population

    dist_dict, school_dict, block_data, G = cleanup(DATA_DIR)
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
        score = mino_coef*(dem_score) + (1-mino_coef)*trav_score
        return (dem_score, trav_score, score)
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
            return nx.draw(G, node_color=list(nx.get_node_attributes(G, "color").values()), pos=block_data['POS'].loc[list(G.nodes())], node_size=list(nx.get_node_attributes(G, 'size').values()))

    # {district: current scores}
    district_scores = {dist: _calc_score(
        dist, school_dict[dist]) for dist in schoollist}
    current_dem_score = np.round(
        sum([district_scores[u][0] for u in schoollist])/tot_pop, 3)
    current_trav_score = np.round(
        sum([district_scores[u][1] for u in schoollist])/tot_pop, 3)

    print('*'*91)
    print(
        f"|   ||TIME {'{:>19}'.format('||')} Diversity Score:  {'{:6.3f}'.format(current_dem_score)} || Mean Travel Time (sec): {'{:6.3f}'.format(current_trav_score)}|")
    print('|'+'*'*3+'||'+'*'*22+'||'+'*'*26+'||'+'*'*32+'|')

    start = timeit.default_timer()
    stop = timeit.default_timer()

    score_list, animation_list = [], []
    frames = 0
    # improvements is equivilent to steps, it counts the number of times we beat the current score (when we don't beat the score, the map doesn't change (assuming random == False))
    final_scores = []
    for i in range(1000):
      improvements = 0
      while improvements < steps:
          # picking a random pair of neighboring districts
          # we pick a random edge of the quotient graph which has a node for each district and an edge if they are adj.
          # this is likely inefficient :)
          pair = list(zip(*choice(list(nx.quotient_graph(G,
                                                        list(map(set, school_dict.values()))).edges()))))[0]
          school_dict_r = {v: k for k, val in school_dict.items() for v in val}
          schl1, schl2 = school_dict_r[pair[0]], school_dict_r[pair[1]]
          district_pair = school_dict[schl1] + school_dict[schl2]
          # initialize districts and scores to the current partitioning
          best_district_1 = school_dict[schl1]
          best_district_2 = school_dict[schl2]
          best_score = district_scores[schl1][2] + district_scores[schl2][2]
          subgraph_copy = nx.subgraph(G, district_pair).copy()
          # counts the number of times we pick a tree without a good breaking (mostly for debugging)
          no_break = 0
          # Have we gotten a better score?
          improved = False
          # assigns random weights to edges then picks minimum tree this is a random spanning tree (hopefully)
          for tree_iters in range(tree_loop):
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
              dem_score1, trav_score1, score1 = _calc_score(schl1, district_1)
              dem_score2, trav_score2, score2 = _calc_score(schl2, district_2)
              score = score1 + score2

              # when random argument is True we don't optimize score

              if score < best_score or random:
                  improved = True
                  all_scores = ((dem_score1, trav_score1, score1),
                                (dem_score2, trav_score2, score2))
                  best_district_1 = district_1
                  best_district_2 = district_2
                  improvements += 1
                  district_scores[schl1] = all_scores[0]
                  district_scores[schl2] = all_scores[1]
                  school_dict[schl1] = best_district_1
                  school_dict[schl2] = best_district_2
                  if random:
                      score_list.append(all_scores)

      final_scores.append(all_scores)
      frames += 1
      # if score decreased, we edit the current districting

      if frames % update_rate == 0 and improvements > 0:
          stop = timeit.default_timer()
          last_dem_score = current_dem_score
          last_trav_score = current_trav_score
          current_dem_score = np.round(
              sum([district_scores[u][0] for u in schoollist])/tot_pop, 3)
          current_trav_score = np.round(
              sum([district_scores[u][1] for u in schoollist])/tot_pop, 3)
          dem_dif, trav_dif = (current_dem_score - last_dem_score) / \
                               current_dem_score, (current_trav_score -
                                                   last_trav_score)/current_trav_score
          print(
              f"|{colored('{:03d}'.format(improvements),'cyan')}|| Iteration Time: {'{:03.2f}'.format(frames/(stop-start))} || Diversity Score: {signed_color(dem_dif)} || Mean Travel Time (sec): {signed_color(trav_dif)}|     {current_dem_score} {current_trav_score}")
          print({'diversity': current_dem_score, 'travel': current_trav_score})
    print('|'+'*'*3+'||'+'*'*22+'||'+'*'*26+'||'+'*'*32+'|')
    print(
    f"|   ||Total Time: {'{:<6.2f}'.format(stop-start)} {'{:>5}'.format('||')} Diversity Score:  {'{:6.3f}'.format(current_dem_score)} || Mean Travel Time (sec): {'{:6.3f}'.format(current_trav_score)}|")
    print('*'*91)
    return final_scores

