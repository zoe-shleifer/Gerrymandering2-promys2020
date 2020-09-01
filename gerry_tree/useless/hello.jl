using Graphs, Pandas, JSON, ProgressBars

@printf "hello world"
function make_data(DATA_DIR :: string)
    const g = json_to_graph("$DATA_DIR/small_final_graph.txt")
    const init_districts = open(JSON.parse, "$DATA_DIR/original_districts.txt")
    const distances = open(JSON.parse, "$DATA_DIR/distance_data.txt")
    block_csv = read_csv("$DATA_DIR/small_final_graph.txt")
function json_to_graph(GRAPH_PATH :: string) :: SimpleGraph
    g_dict = open(JSON.parse, GRAPH_PATH)

    #transfer all nodes to ExVertex types
    new_vs = Array(ExVertex, length(g_dict["nodes"]))
    i = 1
    for vertex in g_dict["nodes"]
        new_vs[i] = ExVertex(i, vertex["id"])
        for (key, value) in vertex
            if !(key == "id")
                new_vs[i].attributes[key] = value
            end
        end
        i += 1
    end

    #transfer all edges to ExEdge types
    #source/target indices in the "links" dicts are 0-based, while ExEdge indices are 1-based
    new_es = Array(ExEdge{ExVertex}, length(g_dict["links"]))
    i = 1
    for edge in g_dict["links"]
        new_es[i] = ExEdge(i, new_vs[edge["source"]+1], new_vs[edge["target"]+1])
        for (key, value) in edge
            if !(key in ("source", "target"))
                new_es[i].attributes[key] = value
            end
        end
        i += 1
    end

    #create graph
    g = graph(new_vs, new_es, is_directed=g_dict["directed"])
    return g

function random_spanning(g: SimpleGraph):: SimpleGraph
    rand(G.vertices())