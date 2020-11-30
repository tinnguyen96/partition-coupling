# Generate new graph as an Icosahedral
if False:
    graph_name = "Icosahedral"
    g = igraph.Graph.Famous(graph_name)
else:
    n, p =  20, 0.15
    g = igraph.Graph.Erdos_Renyi(n, p)

## number of vertices in the graph
nvertices = g.vcount()

def rinit(g):
    """greedy initialization of graph coloring.  Adds a new color whenever needed.
    """
    nvertices = g.vcount()
    vertex_colors = -np.ones([nvertices], dtype=int)
    color_ids = set() 
    for ivertex in range(nvertices):
        n_i = igraph.Graph.neighbors(g, ivertex)
        legal_colors = color_ids.difference(vertex_colors[n_i])
        if len(legal_colors) == 0:
            new_color_id = len(color_ids)
            color_ids.add(new_color_id)
            legal_colors.add(new_color_id)
        vertex_colors[ivertex] = min(legal_colors)
    return vertex_colors

vertex_colors_init = rinit(g)
print(vertex_colors_init)

## all possible colours
ncolors = len(set(vertex_colors_init))+1
all_colours = np.array(sns.color_palette("Paired", n_colors=ncolors))
    
def color_probs(g, ncolors, n, vertex_colors):
    """color_probs returns uniform probability of new color assigments 
    of vertex  across all the legal colors, i.e. those not shared 
    by neighbors.
    
    Args:
        g: igraph Graph object
        ncolors: number of different colors
        n: index of node to re-color
        vertex_colors: array of indices of current colors
    """
    legal = np.ones(ncolors)
    neighbors = igraph.Graph.neighbors(g, n)
    legal[list(set(vertex_colors[neighbors]))] = 0.
    probs = legal / sum(legal)
    return probs

## Markov chain,
def single_kernel(g, ncolors, vertex_colors, n=None):
    """single_kernel makes a single markov step by reassigning the color of a randomly chosen vertex.
    
    Args:
        g: graph object
        ncolors: total number of colors that may be used.
        vertex_colors: color assignment of each vertex.  An np.array 
            of ints with values between 0 and ncolors-1. 
    
    Returns:
        New assignments of vertex colors
    """
    if n is None: n = np.random.choice(g.vcount())
    v_probs = color_probs(g, ncolors, n, vertex_colors)
    vertex_colors[n] = np.random.choice(ncolors, p=v_probs)
    return vertex_colors

def gibbs_sweep_single(g, ncolors, vertex_colors):
    for n in range(g.vcount()): vertex_colors = single_kernel(g, ncolors, vertex_colors.copy(), n)
    return vertex_colors

# utilities color relabling step
def color_ordering(ncolors, vertex_colors):
    """color_ordering returns the order of occurrence of each color in vertex_colors.
    Unused colors are assigned an order greater than the number of unique colors in 
    vertex_colors.
    """
    complete_list_of_colors = np.array(list(vertex_colors) +  list(range(ncolors)))
    idx_of_first_occurrence = [np.where(complete_list_of_colors==c)[0][0] for c in range(ncolors)]
    return np.argsort(idx_of_first_occurrence)
    
def relabel_colors(ncolors, vertex_colors, new_order):
    old_ordering = color_ordering(ncolors, vertex_colors)
    vertex_colors_new = vertex_colors.copy()
    for c in range(ncolors):
        vertex_colors_new[np.where(vertex_colors==old_ordering[c])] = new_order[c]
    return vertex_colors_new

def max_coupling(v1_probs, v2_probs):
    """max_coupling as described in Jacob's chapter 3 notes.
    """
    ncolors = len(v1_probs)
    
    # compute overlap pmf
    overlap = np.min([v1_probs, v2_probs], axis=0)
    overlap_size = np.sum(overlap)
    overlap_size = np.min([1.0, overlap_size]) # protect from rounding error 
    if np.random.choice(2, p=[1-overlap_size, overlap_size]) == 1:
        newz = np.random.choice(ncolors, p=overlap/overlap_size)
        return newz, newz
    
    # sample from complements independently
    v1_probs -= overlap
    v1_probs /= (1-overlap_size)
    
    v2_probs -= overlap
    v2_probs /= (1-overlap_size)
    
    newz1 = np.random.choice(ncolors, p=v1_probs)
    newz2 = np.random.choice(ncolors, p=v2_probs)
    return newz1, newz2

def opt_coupling(v1_probs, v2_probs, clusts1, clusts2, intersection_sizes):
    """opt_coupling returns a sample from the optimal coupling of v1_probs and v2_probs.
    
    Args:
        v1_probs, v2_probs: marginals for chains 1 and 2
        clusts1, clusts2: color group assignments chains 1 and 2
    """
    assert len(v1_probs) == len(v2_probs)
    ncolors = len(v1_probs)
    pairwise_dists = utils.pairwise_dists(clusts1, clusts2, intersection_sizes, allow_new_clust=False)
    _, (v1_color, v2_color), _ = utils.optimal_coupling(
        v1_probs, v2_probs, pairwise_dists, normalize=True,
        change_size=100)
    return v1_color, v2_color


def double_kernel(g, ncolors, vertex_colors1, vertex_colors2, n, clusts1, clusts2,
                  intersection_sizes, coupling="Maximal"):
    """double_kernel simulates one step for a pair of coupled Markov chains over colorings.
    
    A vertex, n_i, is selected uniformly at random from the set of all vertices and has its 
    color reassigned.  Marginally this assigment is uniformly random over the set of 
    allowable colors.  The joint distribution of their coupling is set by the coupling argument.
    
    
    Args:
        g: graph object
        ncolors: total number of possible colors
        vertex_colors1, vertex_colors2: current color assignments of all vertices in both chains
        n: index of vertex to recolor
        coupling: method of coupling Gibs proposal "Maximal", "Optimal" or "Random"
        
    Returns:
        vertex_colors1, vertex_colors2 : new assignments of vertex colors.
    """
    # remove node n from clusts and intersection sizes
    clusts1[vertex_colors1[n]].remove(n)
    clusts2[vertex_colors2[n]].remove(n)
    intersection_sizes[vertex_colors1[n], vertex_colors2[n]] -= 1
    
    # compute marginal probabilities
    v1_probs = color_probs(g, ncolors, n, vertex_colors1)
    v2_probs = color_probs(g, ncolors, n, vertex_colors2)
    
    # Sample new color assignments from coupling
    if coupling == "Maximal":
        v1_color, v2_color = max_coupling(v1_probs, v2_probs)
    elif coupling == "Common_RNG":
        v1_color, v2_color = utils.naive_coupling(v1_probs, v2_probs)
    elif coupling == "Random":
        # This is an independent coupling
        v1_color = np.random.choice(ncolors, p=v1_probs)
        v2_color = np.random.choice(ncolors, p=v2_probs)
    else:
        # This defines the coupling by solving an optimal transport problem.
        assert coupling == "Optimal"
        v1_color, v2_color = opt_coupling(v1_probs, v2_probs, clusts1, clusts2, intersection_sizes)
        
    # update group assignments and intersection sizes
    clusts1[v1_color].add(n); clusts2[v2_color].add(n)
    intersection_sizes[v1_color, v2_color] += 1
    vertex_colors1[n], vertex_colors2[n] = v1_color, v2_color
    
    return vertex_colors1, vertex_colors2

def gibbs_sweep_couple(g, ncolors, vertex_colors1, vertex_colors2, coupling="Maximal"):
    """gibbs_sweep_couple performs Gibbs updates for every node in the graph, coupling
    each update across the two chains.
    
    We compute intersection sizes once at the start and then update it for better time complexity.
    """
    # Compute clusters and intersection sizes from scratch once
    clusts1 = utils.z_to_clusts(vertex_colors1, total_clusts=ncolors)
    clusts2 = utils.z_to_clusts(vertex_colors2, total_clusts=ncolors)
    intersection_sizes = np.array([[len(c1.intersection(c2)) for c2 in clusts2] for c1 in clusts1])
    
    # sample from conditional for each vertex
    for n in range(g.vcount()):
        vertex_colors1, vertex_colors2 = double_kernel(
            g, ncolors, vertex_colors1.copy(), vertex_colors2.copy(), n,
            clusts1, clusts2, intersection_sizes, coupling=coupling)
    return vertex_colors1, vertex_colors2