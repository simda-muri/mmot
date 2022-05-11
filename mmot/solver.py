import numpy as np
from w2 import BFM

import os 

import matplotlib.pyplot as plt 
import matplotlib as mpl

import itertools

import IPython.display as idisp

from .graph_utilities import * 
from .bfm_utilities import *


class MMOTResult:
    """ Class storing results of MMOTSolver. 

    Attributes
    ----------
    dual_vars : list of np.array
        The optimal dual variables computed by MMOTSolver.Solve.
    conv_code : int
        The convergence reason.  Positive indicates success.  Negative indicates failure.  Details can be found in the MMOTSolver.Solve function.
    line_its : list of int
        Number of backtracking line search iterations required at each iteration of the optimization.
    costs : list of float
        Value of the dual objective function at each iteration of the optimizer.
    grad_sq_norms : list of float 
        Squared norm of the gradient at each iterations of the optimizer.
    step_sizes : list of float 
        Step size at the beginning of each iteration (e.g., at the beginning of the line search).
        
    """

    def __init__(self, dual_vars, conv_code, line_its, costs, grad_sq_norms, step_sizes):
        self.dual_vars = dual_vars 
        self.conv_code = conv_code 
        self.line_its = line_its 
        self.costs = costs 
        self.grad_sq_norms = grad_sq_norms
        self.step_sizes = step_sizes

class MMOTSolver:
  """ Solves the MMOT problem with weighted pairwise squared L2 cost.


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

      self._edges = []
      for edge in edges:
          if(self._edge_weights[edge[0],edge[1]]>1e-8):
              self._edges.append(edge)

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

    # fig, axs = plt.subplots(ncols=len(dual_vars))
    # for i, f in enumerate(dual_vars):
    #     axs[i].imshow(f)
    # fig.suptitle('Dual Variables for Barycenter')

    # bary = np.zeros(measure_shape) 
    # for i,f in enumerate(dual_vars):
    #     if(self._bary_weights[i]>1e-8):
    #         bary += push_forward(self._bf, f/self._bary_weights[i], self._measures[i], self._x, self._y)
    #         #bary += push_forward3(f/self._bary_weights[i], self._measures[i], self._x, self._y)

    # bary *= np.prod(measure_shape)/np.sum(bary)

    #bary = push_forward3(dual_vars[0]/self._bary_weights[0], self._measures[0], self._x, self._y)
    bary = push_forward(self._bf, dual_vars[0]/self._bary_weights[0], self._measures[0], self._x, self._y)
        
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
    
    Parameters
    ----------
      root_node : None, int, or string
        If an integer, a directed tree flowing into this node will be constructed and visualized.  
        If root_node=='original', the original unrolled graph will be visualized.  Otherwise the 
        unrolled, but undirected, tree will be visualized.

    Other Parameters
    ----------------
    filename : string 
        The name of an image file to create with the visualization.  
    layout : string 
        The name of the layout used by igraph to visualize the graph.  Defaults to 'reingold_tilford',
        which is good for trees.   See the [igraph documentation](https://igraph.org/python/tutorial/latest/visualisation.html#graph-layouts)
        for other options.
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

    if('layout' in kwargs):
        layout_name = kwargs['layout']
    else:
        layout_name = 'reingold_tilford'

    for file in files_to_create:
      layout = tree.layout(layout=layout_name)
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

    
  def Step(self, root_node, dual_vars, step_size):
      
    if(root_node != self.save_root_node):
      self.save_root_node = root_node 
      self.tree, self.layers = self.CreateDirected(root_node)
      self.f_tmp = None 

    num_verts = self.tree.vcount()

    grad_squared_norm = 0.0 # Squared norm of the gradient 


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
            #smu = push_forward2(self.f_tmp[vert_ind], self._measures[self._measure_map[next_vert_ind]], self._x, self._y, w)
            #smu = push_forward3(self.f_tmp[vert_ind], self._measures[self._measure_map[next_vert_ind]], self._x, self._y, w)
            
            grad_squared_norm += update_potential(dual_vars[vert_ind], smu, self._measures[self._measure_map[vert_ind]], self._kernel, -step_size)
       
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

    return grad_squared_norm

  def Solve(self, dual_vars=None, **kwargs):
    """ Finds the dual variables that solve the MMOT problem.

        | Convergence code | Description |
         ----------------------------------
        | -3    |  Failed line search. |
        | -2    | Reach maximum iterations. |
        | -1    | Generic failure. |
        |  1    | Converged because of sufficient small gradient norm. | 
        |  2    | Converged because of small change in objective. |
        
        Parameters
        ----------
        dual_vars : None or list of np.array
            Initial guesses for the dual variables.  The length of the list should be the same
            as `NumDual()` and each dual variable should have the same shape as the locations `x`
            and `y` given to the constructor. If `dual_vars` is `None`, then the dual variables
            will be initialized to zero.

        Other Parameters
        ----------------
        max_its : int
            The maximum number of iterations the optimizer is allowed to take.  Defaults to 1000.
        step_size : float
            The step size used in the gradient step of the optimizer.  Defaults to 1
        ftol_abs : float 
            The absolute tolerance on the change in objective function value.  If the change is smaller than this value, the optimizer will terminate.
        ftol_rel : float 
            The relative tolerance on the change in objective function value.  If the change is smaller than this value, the optimizer will terminate.
        gtol_abs : float 
            The absolute tolerance on the norm of the gradient. 
        gtol_rel : float 
            The relative tolerance on the norm of the gradient. 
        max_line_ts : int 
            Maximum number of backtracking steps in the line search.
        root_nodes : string or list of int 
            Nodes to cycle over when defining the root node in the directed graph.

        Returns
        --------
        result : MMOTResult
            An instance of the SolverResult class, which contains the optimal dual variables as well as convergence information.

    """ 


    if(dual_vars is None):
        dual_vars = [np.zeros(self._x.shape) for i in range(self.NumDual())]
    else:
        assert(len(dual_vars)==self.NumDual())
        for f in dual_vars:
            assert(dual_vars.shape[0]==self._x.shape[0])
            assert(dual_vars.shape[1]==self._y.shape[0])

    max_line_its = 20

    # Extract options from the kwargs
    if('max_its' in kwargs):
        max_its = kwargs['max_its']
    else:
        max_its = 1000

    if('step_size' in kwargs):
        step_size = kwargs['step_size']
    else:
        step_size = 1.0

    if('ftol_abs' in kwargs):
        ftol_abs = kwargs['ftol_abs']
    else:
        ftol_abs = 1e-11
    
    if('gtol_abs' in kwargs):
        gtol_abs = kwargs['gtol_abs']
    else:
        gtol_abs = 1e-7

    if('ftol_rel' in kwargs):
        ftol_rel = kwargs['ftol_rel']
    else:
        ftol_rel = 1e-11
    
    if('gtol_rel' in kwargs):
        gtol_rel = kwargs['gtol_rel']
    else:
        gtol_rel = 1e-7

    if('max_line_its' in kwargs):
        max_line_its = kwargs['max_line_its']
    else:
        max_line_its = 20

    if('root_nodes' in kwargs):
        root_nodes = kwargs['root_nodes']
    else:
        root_nodes = 'all'

    # Get ready to hold the result and convergence history
    res = MMOTResult(None, -1, [], [], [], [])

    if(root_nodes=='all'):
        root_nodes = np.arange(self.NumDual())
    root_cycler = itertools.cycle(root_nodes)
    old_cost = self.ComputeCost(dual_vars)
    res.costs.append(old_cost)

    alpha = 1.0
    print('Iteration, StepSize,        Cost,        Error,  Line Its')
    for it in range(max_its):
        next_root = next(root_cycler)
        
        #alpha = np.maximum(np.minimum(1.2*alpha,1.0),1e-4) 
        #alpha = 1.0
        res.step_sizes.append(alpha*step_size)

        line_it = 0
        while(line_it<max_line_its):
            new_duals = np.copy(dual_vars)
            newSqNorm = self.Step(next_root, new_duals, alpha*step_size)
            if(line_it==0):
                gradSqNorm = np.copy(newSqNorm)

            new_cost = self.ComputeCost(new_duals)

            if(new_cost>=old_cost+1e-8*step_size*alpha*gradSqNorm):
                res.costs.append(new_cost) 
                old_cost = np.copy(new_cost)
                dual_vars = np.copy(new_duals)
                gradSqNorm = np.copy(newSqNorm)

                if(line_it==0):
                    alpha = np.minimum(1.5*alpha,1.0) 
                break
            else:
                alpha *= 0.5
                
            line_it += 1

        if((line_it>=max_line_its)&(len(root_nodes)==1)):
            print('{:9d},   {:0.4f},  {:0.4e},   {:0.4e},  {:8d}'.format(it,alpha*step_size, res.costs[-1], gradSqNorm, line_it))
            print('Terminating due to failed line search.')
            res.conv_code = -3
            break 

        res.grad_sq_norms.append(gradSqNorm)
        res.line_its.append(line_it)

        
        if((it%10)==0):
            print('{:9d},   {:0.4f},  {:0.4e},   {:0.4e},  {:8d}'.format(it,alpha*step_size, res.costs[-1], gradSqNorm, line_it))

        # Check for convergence in cost
        if(np.abs(res.costs[-1]-res.costs[-2])<ftol_abs):
            print('{:9d},   {:0.4f},  {:0.4e},   {:0.4e},  {:8d}'.format(it,alpha*step_size, res.costs[-1], gradSqNorm, line_it))
            print('Terminating due to small change in objective.')
            res.conv_code = 2
            break

        # Check for convergence via gradient 
        if(gradSqNorm<gtol_abs):
            print('{:9d},   {:0.4f},  {:0.4e},   {:0.4e},  {:8d}'.format(it,alpha*step_size, res.costs[-1], gradSqNorm, line_it))
            print('Terminating due to small gradient norm.')
            res.conv_code = 1
            break

    if(it==max_its):
        res.conv_code = -2

    res.dual_vars = dual_vars
    return res