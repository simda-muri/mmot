import numpy as np
from w2 import BFM

import os 

import matplotlib.pyplot as plt 
import matplotlib as mpl

import IPython.display as idisp

from .graph_utilities import * 
from .bfm_utilities import *

def interpolate_function(func, xs, ys):
  """ Bilinear interpolation of a function defined at cell centers.

  Mimics the BFM "interpolate_function" function, which assumes the values of "func" live at the center of 
  each pixel and uses bilinear interpolation to interpolate between points.  The values around the edge are 
  duplicated at "ghost" points to estimate the function in the half-pixel boundary around the edge.  The gradient 
  in this boundary region is therefore zero.

  See for example, https://github.com/Math-Jacobs/bfm/blob/c93de454c49c6958e0a8e18b285c2e005b55507f/python/src/main.cpp#L263

  ARGUMENTS:
    func (np.array) :  Nx by Ny matrix of function values at the cell centers 
    xs (np.array) : A length M vector containing the x component of the points where the function is to be evaluated
    ys (np.array) : A length M vector containing the y component of the points where the function is to be evaluated

  RETURNS:
    np.array : The values of the function at the points (xs[i],ys[i])

  """
  n1, n2 = func.shape 

  xindex = np.round(np.maximum(np.minimum(xs*n1 - 0.5, n1-1),0)).astype(np.int)
  yindex = np.round(np.maximum(np.minimum(ys*n2 - 0.5, n2-1),0)).astype(np.int)

  xfrac = xs*n1 - 0.5 - xindex 
  yfrac = ys*n2 - 0.5 - yindex 

  xother = np.round(np.maximum(np.minimum(xindex + np.sign(xfrac), n1-1), 0)).astype(np.int)
  yother = np.round(np.maximum(np.minimum(yindex + np.sign(yfrac), n2-1), 0)).astype(np.int)

  v1 = (1.0-np.abs(xfrac))*(1.0-np.abs(yfrac))*func[xindex,yindex]
  v2 = np.abs(xfrac)*(1-np.abs(yfrac))*func[xother,yindex]
  v3 = (1.0-np.abs(xfrac))*np.abs(yfrac)*func[xindex,yother]
  v4 = np.abs(xfrac)*np.abs(yfrac)*func[xother,yother]
        
  return v1+v2+v3+v4



def evaluate_gradient(func, xs, ys):
  """ Computes an approximation of the gradient of a 2d scalar function defined at cell centers.

  Evaluates the gradient of a scalar function defined on an n1xn2 regular grid.  Assumes the values of 
  the function are at the cell centers.

  ARGUMENTS:
    func (np.array) : Nx by Ny matrix of function values at the cell centers 
    xs (np.array) : A length M vector containing the x component of the points where the function's gradient should be evaluated
    ys (np.array) : A length M vector containing the y component of the points where the function's gradient should be evaluated

  RETURNS:
    np.array : A length M vector containing the x derivative of the function at the points (xs[i],ys[i])
    np.array : A length M vector containing the y derivative of the function at the points (xs[i],ys[i])

  """
  assert len(xs)==len(ys)
  assert(len(xs.shape)==1)
  assert(len(ys.shape)==1)
  
  n1, n2 = func.shape
  dx = 0.5/n1
  dy = 0.5/n2

  derivx = (interpolate_function(func, xs+dx, ys) - interpolate_function(func, xs-dx, ys))/(2.0*dx)
  derivy = (interpolate_function(func, xs, ys+dy) - interpolate_function(func, xs, ys-dy))/(2.0*dy)

  return derivx, derivy 



class MMOTSolver:
  """ Solves the MMOT problem with weighted pairwise squared L2 cost.

      .. math::


  """
  
  def __init__(self, measures, edges, x, y, unroll_node=0, weights=None):

      self._measures = measures 
      num_marginals = len(measures)
      
      self.f_tmp = None 
      self.save_root_node = -1

      self._x = x 
      self._y = y

      # Check to make sure the edge definitions are consistent with the number of measures
      for edge in edges:
          assert(len(edge)==2)
          assert(edge[0]<len(measures))
          assert(edge[1]<len(measures))
          assert(edge[0]!=edge[1])

      self._edges = edges 
      self._edge_weights = dict() 
      self._bary_weights = None 

      if(weights is not None):
          assert(np.min(weights)>=0)
          
          # If the weights are defined for each measure
          if(len(weights)==len(measures)):
              self._bary_weights = weights 
              self._bary_weights /= np.sum(self._bary_weights)

              # The edge weights are the products of the barycenter weights 
              for i in range(len(weights)):
                  for j in range(i+1,len(weights)):
                      self._edge_weights[i,j] = weights[i]*weights[j] 
                      self._edge_weights[j,i] = weights[i]*weights[j]
        
          # More general setting where the weights are defined on the edges 
          elif(len(weights)==len(edges)):          
              for i, edge in enumerate(edges):
                  self._edge_weights[edge[0], edge[1]] = weights[i]
                  self._edge_weights[edge[1], edge[0]] = weights[i]
          else:
              raise RuntimeError("Weights passed to MMOTSolver do not have the correct shape.  The number of weights must match either the number of edges or the number of marginals.")

      else:
 
          for i in range(len(measures)):
              for j in range(i+1,len(measures)):
                self._edge_weights[i,j] = 1.0
                self._edge_weights[j,i] = 1.0

      self._unrolled_tree, self._measure_map = self.CreateUndirected(self._measures, self._edges, unroll_node)

      self._n1, self._n2 = measures[0].shape

      self._bf = BFM(self._n1, self._n2, measures[0])

      self._kernel = initialize_kernel(self._n1, self._n2)

  def CreateUndirected(self, measures, edges, unroll_node):
      """
      Creates the unrolled undirected graph (needed to construct tree)

       RETURNS:
        igraph.Graph : An undirected tree constructed by "unrolling" the original 
                       pairwise cost graph.  Note that some of the measures may 
                       be duplicated during the "unrolling" process, causing this 
                       tree to have more nodes than the original number of measures.
        list(int) :  A mapping from vertices in the unrolled tree to vertices (i.e., marginals)
                    in the original undirected graph.   The i-th component of this 
                    list contains the index of the measure that matches the i-th vertex in the tree.
      """
      self._orig_graph = ig.Graph()
      self._orig_graph.add_vertices(len(measures))
      self._orig_graph.add_edges(edges)

      print(unroll_node)
      graph, vertmap = self._orig_graph.unfold_tree(roots=[unroll_node])

      return graph, vertmap

  def Barycenter(self, unrolled_dual_vars):
    """ Returns the barycenter of the original measures given dual variables on the unrolled tree.  The dual variables are the same 
        as those that would be passed to the ComputeCost or Step functions.
        
        ARGUMENTS:
            unrolled_dual_vars (list of np.array) : List of vectors containing the dual variables for each node in the unrolled graph.
            weights (np.array) : Vector containing weights on each measure in the original problem.  Must sum to one.

        RETURNS:
            np.array : A 2d numpy array containing the barycenter.
    """
    assert(self._bary_weights is not None)

    # First, combine the unrolled dual variables to estimate the dual variables in the original problem
    measure_shape = self._measures[0].shape 
    dual_vars = [np.zeros(measure_shape) for i in range(len(self._measures))]
    for i,f in enumerate(unrolled_dual_vars):
        dual_vars[self._measure_map[i]] += f 

    bary = np.zeros(measure_shape) 
    for i,f in enumerate(dual_vars):
        bary += push_forward(self._bf, f/self._bary_weights[i], self._measures[i], self._x, self._y)

    bary *= np.prod(measure_shape)/np.sum(bary)

    #bary = push_forward(self._bf, weights[0]*dual_vars[0], self._measures[0], self._x, self._y)
        
    return bary 

  def NumDual(self):
      """ Returns the number of vector-valued dual variables, which is equivalent 
          to the number of vertices in the unrolled tree.
      """
      return self._unrolled_tree.vcount()

  def CreateDirected(self, root_node):
      """
      Uses a breadth-first search to convert the undirected tree into a directed 
      tree where all edges "flow" towards the given root_node.  This function 
      starts with the undirected tree created by a call to :code:`CreateUndirected` 
      in the :code:`__init__` function and stored as a member variable in
      :code:`self._unrolled_tree`

      RETURNS:
        igraph.Graph : A directed graph containing a tree with the specified root node. 
        
        list(list(int)) : A list of lists containing the vertices in each layer of the tree.  
                          layers[i] contains the indices of vertices in layer i of the tree.  
                          layers[0] contains the vertices that are farthest from the root node.
                          layers[-1] contains just the root node.
      """
      bfs_verts, bfs_layers, _ = self._unrolled_tree.bfs(root_node) 

      parallel_layers = [bfs_verts[bfs_layers[i]:bfs_layers[i+1]] for i in range(len(bfs_layers)-1)]
      
      layer_ids = [None]*self._unrolled_tree.vcount()  # will contain mapping from vertex id to which layer the vertex is in
      for layer_ind, layer in enumerate(parallel_layers):
        for vert_ind in layer:
          layer_ids[vert_ind] = layer_ind
          
      # Construct a directed tree from the undirected tree
      directed_tree = ig.Graph(directed=True)
      directed_tree.add_vertices(self._unrolled_tree.vcount())

      directed_edge_list = []
      for edge in self._unrolled_tree.get_edgelist():
        if(layer_ids[edge[0]]>layer_ids[edge[1]]):
          directed_edge_list.append(edge)
        else:
          directed_edge_list.append([edge[1],edge[0]])

      directed_tree.add_edges(directed_edge_list)

      # Reverse the layer definition because the bfs starts at the root node  
      parallel_layers.reverse()

      return directed_tree, parallel_layers

  def Visualize(self, root_node=None, **kwargs):
    """
    Visualizes the undirected (or optionally directed) tree representation of the 
    cost graph.  The nodes in the tree will be labeled with pairs in the measure_map.
    The first integer in the label corresponds to the index in the tree and the 
    second integer corresponds to the original measure.   Nodes in a directed tree 
    will be colored based on their layer, which corresponds to the number of 
    edges between the node and the root node.
    
    ARGUMENTs:
      root_node (None, int, or string) : If an integer, a directed tree flowing into this node 
                                will be constructed and visualized.  If root_node=='original',
                                the original unrolled graph will be visualized.  Otherwise the 
                                unrolled, but undirected, tree will be visualized.
    """ 
    
    vert_labels = ['f{},m{}'.format(i,v) for i,v in enumerate(self._measure_map)]

    if(root_node is None):
      num_verts = self._unrolled_tree.vcount()
      tree = self._unrolled_tree
      layers = [list(range(num_verts))]
      
    elif(isinstance(root_node, str)):
      assert(root_node=='original')
      num_verts = self._orig_graph.vcount()
      layers = [list(range(num_verts))]
      tree = self._orig_graph
      vert_labels = ['m{}'.format(v) for v in range(len(self._measures))]

    else:
      num_verts = self._unrolled_tree.vcount()
      tree, layers = self.CreateDirected(root_node)

    num_layers = len(layers)

    node_to_layer = dict()
    for layer_num, layer_nodes in enumerate(layers):
      for ind in layer_nodes:
        node_to_layer[ind] = layer_num 
      
    cm = plt.get_cmap('Blues') 
    cNorm  = mpl.colors.Normalize(vmin=0, vmax=len(layers)+1)
    scalarMap = mpl.cm.ScalarMappable(norm=cNorm, cmap=cm)
    
    # Plot the directed tree
    files_to_create = ['temp.png']
    if('filename' in kwargs):
      files_to_create.append(kwargs['filename'])

    for file in files_to_create:
      layout = tree.layout(layout='reingold_tilford') # <- TODO: Make this an option
      ig.plot(tree, layout=layout, bbox=(400, 400), edge_width=4, margin=50,
              vertex_color=[list(scalarMap.to_rgba(node_to_layer[i])[0:3]) for i in range(num_verts)],
              vertex_label=vert_labels,
              vertex_size=40,
              target=file)
    
    idisp.display(idisp.Image(filename=files_to_create[0]))
    os.remove(files_to_create[0])
    


  def ComputeCost(self,dual_vars):
        """ Computes the multi-marginal dual objective function.

            The dual objective function is given by
            Σ_i ∫ f_i ν_i(x)dx 
        """
        n1, n2 = self._measures[0].shape

        cost = 0
        for dual_ind, meas_ind in enumerate(self._measure_map):
          cost += np.sum(dual_vars[dual_ind]*self._measures[meas_ind])/(n1*n2)
        return cost

  def StepSizeUpdate(self, sigma, value, oldValue, gradSq):
  
    # Parameters for Armijo-Goldstein
    scaleDown = 0.75
    scaleUp   = 1/scaleDown
    upper = 0.9
    lower = 0.1
    
    # Armijo-Goldstein
    diff = value - oldValue

    if diff > gradSq * sigma * upper:
        return sigma * scaleUp
    elif diff < gradSq * sigma * lower:
        return sigma * scaleDown
    return sigma
    
  def Step(self, root_node, dual_vars, step_size):

    if(root_node != self.save_root_node):
      self.save_root_node = root_node 
      self.tree, self.layers = self.CreateDirected(root_node)
      self.f_tmp = None 

    num_verts = self.tree.vcount()

    error = 0.0    


    # Check to see if we've already taken a step with this root node and can reuse the net fluxes
    if(self.f_tmp is None):

        self.f_tmp = [None]*num_verts

        # Do the final c-transform to update the root note
        for layer in self.layers[:-1]:
            for vert_ind in layer:
          
                # Get the index of the downstream node 
                next_vert_ind = self.tree.vs[vert_ind].out_edges()[0].target
                w = self._edge_weights[self._measure_map[vert_ind],self._measure_map[next_vert_ind]]
                
                if(self.tree.degree(vert_ind, mode="in")==0):
                    self.f_tmp[vert_ind] = c_transform(self._bf, dual_vars[vert_ind], self._x, self._y, w)
            
                else:
                    # Compute the net flux (f_i - \sum_j`` f_tmp[j]) at this vertex
                    f_net = np.copy(dual_vars[vert_ind])
                    for edge in self.tree.vs[vert_ind].in_edges():
                        assert(self.f_tmp[edge.source] is not None)
                        f_net -= self.f_tmp[edge.source]

                    self.f_tmp[vert_ind] = c_transform(self._bf, f_net, self._x, self._y, w)
    
    # Gradient updates for all but root node
    for layer in self.layers[:-1]:
        for vert_ind in layer:

            # Update the dual variable at this node 
            next_vert_ind = self.tree.vs[vert_ind].out_edges()[0].target
            w = self._edge_weights[self._measure_map[vert_ind],self._measure_map[next_vert_ind]]
            smu = push_forward(self._bf, self.f_tmp[vert_ind], self._measures[self._measure_map[next_vert_ind]], self._x, self._y, w)
            error += update_potential(dual_vars[vert_ind], smu, self._measures[self._measure_map[vert_ind]], self._kernel, -step_size)
       
    # Do the final c-transform to update the root note
    for layer in self.layers[:-1]:
      for vert_ind in layer:

        next_vert_ind = self.tree.vs[vert_ind].out_edges()[0].target
        w = self._edge_weights[self._measure_map[vert_ind],self._measure_map[next_vert_ind]]

        if(self.tree.degree(vert_ind, mode="in")==0):
          self.f_tmp[vert_ind] = c_transform(self._bf, dual_vars[vert_ind], self._x, self._y, w)# C-tranform
          
        else:
          # Compute the net flux (f_i - \sum_j f_tmp[j]) at this vertex
          f_net = np.copy(dual_vars[vert_ind])
          for edge in self.tree.vs[vert_ind].in_edges():
            assert(self.f_tmp[edge.source] is not None)
            f_net -= self.f_tmp[edge.source]

          self.f_tmp[vert_ind] = c_transform(self._bf, f_net, self._x, self._y, w)
      
    fsum = np.zeros(dual_vars[0].shape)
    for edge in self.tree.vs[self.layers[-1][0]].in_edges():
      fsum += self.f_tmp[edge.source]

    dual_vars[self.layers[-1][0]] = fsum

    return error