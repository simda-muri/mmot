import cairocffi as cairo
import igraph as ig 

def ConvertToTree(undirected_graph, root_node):
    """
    Starting from an undirected graph, this function constructs a tree with the 
    specified vertex as the root node.  Vertices are duplicated if necessary to 
    define the tree.

    ARGUMENTS:
        undirected_graph (igraph.Graph) : Undirected graph defining pairwise costs
        root_node (int) : Index of the vertex that in the undirected graph that will 
                        serve as the root node of the tree.

    RETURNS:
        igraph.Graph : A directed graph containing a tree with the specified root node.  This
                    tree might have multiple vertices corresponding to the same 
                    marginal distribution.  This duplication is necessary to handle 
                    general pairwise cost structures.  The mapping of the tree vertices
                    to the vertices in the undirected graph is provided by the second
                    output of this function.
        list(int) :  A mapping from vertices in the tree to the vertices (i.e., marginals)
                    in the original undirected graph.   The i-th component of this 
                    list contains the index of the undirected vertex that matches the 
                    i-th vertex in the tree.
        list(list(int)) : A list of lists containing the vertices in each layer of the tree.  
                        layers[i] contains the indices of vertices in layer i of the tree.  
                        layers[0] contains the vertices that are farthest from the root node.
                        layers[-1] contains just the root node.
            
    """
    tree, vertmap = undirected_graph.unfold_tree(roots=[root_node])

    bfs_verts, bfs_layers,_ = tree.bfs(vertmap.index(root_node)) 

    parallel_layers = [bfs_verts[bfs_layers[i]:bfs_layers[i+1]] for i in range(len(bfs_layers)-1)]

    layer_ids = [None]*tree.vcount()  # will contain mapping from vertex id to which layer the vertex is in
    for layer_ind, layer in enumerate(parallel_layers):
        for vert_ind in layer:
            layer_ids[vert_ind] = layer_ind
        
    # Construct a directed tree from the undirected tree
    directed_tree = ig.Graph(directed=True)
    directed_tree.add_vertices(tree.vcount())

    directed_edge_list = []
    for edge in tree.get_edgelist():
        if(layer_ids[edge[0]]>layer_ids[edge[1]]):
            directed_edge_list.append(edge)
        else:
            directed_edge_list.append([edge[1],edge[0]])

    directed_tree.add_edges(directed_edge_list)

    # Reverse the layer definition because the bfs starts at the root node  
    parallel_layers.reverse()

    return directed_tree, vertmap, parallel_layers
