using Graphs, DataFrames, JSON, ProgressBars, CSV
const DATA_DIR = "/Users/zoeshleifer/Gerrymandering2-promys2020/gerry_tree"
block_data = CSV.read("$DATA_DIR/small_final_attributes.csv")
block_data.GEOID10 = map(string,block_data.GEOID10)
stringy_borders = block_data.borders
select!(block_data, Not(borders))
print(names(block_data))



function make_data(data_dir ::String)
    g = json_to_graph("$data_dir/small_final_graph.txt")
    init_districts = open(JSON.parse, "$data_dir/original_districts.txt")
    schools = init_districts.keys()
    distances = open(JSON.parse, "$data_dir/distance_data.txt")
    block_csv = read_csv("$  data_dir/small_final_graph.txt")
end

function random_spanning(g :: SimpleGraph):: SimpleGraph
    rand(G.vertices())
end
