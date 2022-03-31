import mmot.graph_utilities as gu 
import igraph as ig 

def test_ConvertToTree():

    # Number of marginals
    num_marginals = 5 

    # The set A that defines the pairwise costs
    edge_list = [[0,1],
                [0,2],
                [1,3],
                [2,3],
                [3,4]]

    cost_graph = ig.Graph()
    cost_graph.add_vertices(num_marginals)
    cost_graph.add_edges(edge_list)

    root_node = 0
    tree, vertmap, layers = gu.ConvertToTree(cost_graph, root_node)

    assert (len(layers)==4)

