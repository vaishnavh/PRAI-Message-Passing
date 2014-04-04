"""
Author: Santhosh Kumar M (CS09B042)
File: Utils.py
"""

import Factor
import Message
import Clique
import Junction_Tree


def moralize_graph(input_file):
    """
    Performs moralization of the given graph if directed and returns the list of Factor objects (factors).
    """
    f = open(input_file)  # Opens the input file
    lines = f.readlines()  # Reads lines of the file and stores it in a list
    lines = filter(None, [line.strip(' \n\t') for line in lines])  # Strips the lines
    # of whitespaces and compresses the list
    lines = lines[4:]  # Removes first four dummy lines

    # Replacing \t in the string
    for i in xrange(len(lines)):
        new_string = str()
        for c in lines[i]:
            if c == '\t':
                new_string += ' '
            else:
                new_string += c
        # new_string has only space separated terms
        lines[i] = new_string
    # All string in lines have only space separated terms

    factors = dict()
    i = 0
    while i < len(lines):
        # Go through the table of factors
        # The first line must be of the form P(A,B,C...) which we shall
        # overlook
        child_node = lines[i].split()[0][2:]
        child_true = lines[i].split()[2][0]
        i += 1  # Skips the header line (? TODO)
        # How many variables are there!? Read the next line! :)
        if len(lines[i].split()) == 1:
            # We've read something like P(B = t)
            pos = child_true == 'f'
            factor_values = [0.0] * 2
            factor_values[pos] = float(lines[i])
            factor_values[1 - pos] = 1 - float(lines[i])
            factors[child_node] = Factor.Factor(child_node, factor_values)
            i += 1
        else:
            n_vars = len(lines[i + 1].split()) - 1
            vars_in_factor = lines[i].split()[0:n_vars]
            factor_name = ''.join(vars_in_factor)  # ['A','B','C'] -> 'ABC'
            factor_name = child_node + factor_name
            j = i + 1
            i += pow(2, n_vars) + 1  # Number of entries expected in table + header
            factor_values = [0.0] * pow(2, n_vars + 1)
            while j < i:
                row = lines[j].split()
                row_pos = row[:n_vars]  # A list of t t f etc,
                row_val = float(row[n_vars])  # Value of the factor
                pos = (child_true == 'f')
                n_pos = 1 - pos
                for k in xrange(len(row_pos)):
                    pos = (pos * 2) + (row_pos[k] == 'f')
                    n_pos = (n_pos * 2) + (row_pos[k] == 'f')
                factor_values[pos] = row_val
                factor_values[n_pos] = 1.0 - row_val
                j += 1
            factors[factor_name] = Factor.Factor(factor_name, factor_values)
    return factors.values()



def read_file(input_file):
    """
    Reads the given input file.
    Performs moralization to convert to undirected graph if given graph is directed.
    Assigns values to factors which are stored as a list of objects (factors) of class Factor.
    Returns the list of Factor() objects (factors) and the list of random variables, say, variables = ['A', 'C', 'B'], in the given graph.
    """
    f = open(input_file)  # Opens the input file
    lines = f.readlines()  # Reads lines of the file and stores it in a list
    lines = filter(None, [line.strip(' \n\t') for line in lines])  # Strips the lines
    # of whitespaces and compresses the list
    is_directed = 0
    if lines[0] == 'DIRECTED':
        is_directed = 1
    lines = lines[4:]  # Removes first four dummy lines
    nodes = set()
    # Replacing \t in the string
    for i in xrange(len(lines)):
        new_string = str()
        for c in lines[i]:
            if c == '\t':
                new_string += ' '
            else:
                new_string += c
        # new_string has only space separated terms
        lines[i] = new_string
    # All string in lines have only space separated terms

    #Reading nodes
    i = 0
    while i < len(lines):
        # Go through the table of factors
        # The first line must be of the form P(A,B,C...) which we shall
        # overlook
        # If the graph is directed, we've to consider the node J in
        # P(J = t| ... )
        if is_directed == 1:
            child_node = lines[i].split()[0][2:]
            nodes = nodes.union([child_node])
        i += 1  # Skips the header line (? TODO)
        if len(lines[i].split()) > 1:  # To skip cases like P(B = t)
            # How many variables are there!? Read the next line! :)
            n_vars = len(lines[i + 1].split()) - 1
            vars_in_factor = lines[i].split()[0:n_vars]
            for var_in_factor in vars_in_factor:
                # If the factor has not been recorded yet...
                nodes = nodes.union([var_in_factor])
            j = i + 1
            i += pow(2, n_vars)  # Number of entries expected in table + header
            #nodes now contains all the nodes as a dictionary
        i += 1 # Regardless of the case being P(B = t) or P(B = t | A, C) or
        # P(A, B, C) ...
    if is_directed == 1:
        return [moralize_graph(input_file), list(nodes)]
    else:
        #Undirected
        factors = dict()
        i = 0
        while i < len(lines):
            # Go through the table of factors
            # The first line must be of the form P(A,B,C...) which we shall
            # overlook
            i += 1  # Skips the header line (? TODO)
            # How many variables are there!? Read the next line! :)
            n_vars = len(lines[i + 1].split()) - 1
            vars_in_factor = lines[i].split()[0:n_vars]
            factor_name = ''.join(vars_in_factor)  # ['A','B','C'] -> 'ABC'
            j = i + 1
            i += pow(2, n_vars) + 1  # Number of entries expected in table + header
            factor_values = [0.0] * pow(2, n_vars)
            while j < i:
                row = lines[j].split()
                row_pos = row[:n_vars]  # A list of t t f etc,
                row_val = float(row[n_vars])  # Value of the factor
                pos = 0
                for k in xrange(len(row_pos)):
                    pos = (pos * 2) + (row_pos[k] == 'f')
                factor_values[pos] = row_val
                j += 1
            factors[factor_name] = Factor.Factor(factor_name, factor_values)
        return [factors.values(), list(nodes)]




def factors_to_cliques(factors):
    """
    Simply creates a Clique() object for every given Factor() object and returns the list of Clique() objects.
    """
    cliques = []
    for factor in factors:
        # Add a clique object for every factor object
        cliques = cliques + [Clique.Clique(factor.name, factor.values)]
    return cliques

def is_tree(factors):
    """
    Checks whether the given graph is a tree and returns true (or false) respectively.
    """

    # Basically have to ensure that there are only N - 1 edges
    # And that all nodes are reachable
    # Also all factors must be of size 2 at most
    nodes = set()
    no_of_edges = 0
    for factor in factors:
        if len(factor.name) > 2:
            return False # Any factor of greater size => cycle
        elif len(factor.name) == 2:
            no_of_edges += 1  # Encountered an edge
        nodes = nodes.union(list(factor.name)) # Add nodes

    if no_of_edges != len(nodes) - 1:
        # Not a tree
        return False

    return True

def compute_neighbours(cliques):
    """
    Computes the neighbours of each clique and returns the neighbours dictionary.
    For example,
    cliques = [clique_1, clique_2, clique_3] wherein clique_1.name = 'AB', clique_2.name = 'BDE', clique_3.name = 'AC'
    neighbours = {clique_1 : [clique_2, clique_3], clique_2 : [clique_1], clique_3 : [clique_1]}
    """
    # Called for a normal tree
    neighbours = dict()
    for i in xrange(len(cliques)):
        for j in xrange(len(cliques)):
            if i != j:
                # See if they have any variable in common
                if set(cliques[i].name).intersection(set(cliques[j].name)) != set():
                    # Some variable in common exists
                    if cliques[i] in neighbours.keys():
                        if cliques[j] not in neighbours[cliques[i]]:
                            neighbours[cliques[i]] += [cliques[j]]  # Add to list if not there already
                    else:
                        # Hasn't been keyed in the dictionary yet
                        neighbours[cliques[i]] = [cliques[j]]
    return neighbours

def get_elimination_ordering(variables, query_variables):
    """
    Returns an elimination ordering of the random variables after removing the query variables as a list.
    For example, if random variables are 'B', 'C', 'A' and query variable is 'B', then one elimination order is ['C', 'A'].
    """
    return list(set(variables) - set(query_variables))

def multiply_factors(factor_1, factor_2):
        """
        Multiplies the given two factors and returns the resultant factor.
        Names of factors needn't correlate to length
        """

        if factor_1 == None:
            return factor_2
        elif factor_2 == None:
            return factor_1
        size_1 = 0
        size_2 = 0

        #Set the actual scopes
        while (pow(2, size_1) < len(factor_1.values)):
            size_1 += 1
        while (pow(2, size_2) < len(factor_2.values)):
            size_2 += 1


        nodes_1 = list(factor_1.name)[:size_1]  # Nodes participating in factor 1
        nodes_2 = list(factor_2.name)[:size_2]  # Nodes participating in factor 2
        nodes = list(set(nodes_1 + nodes_2))  # Gets only the unique nodes participating
        # in both factors

        remaining_nodes = list(set(factor_1.name + factor_2.name) - set(nodes))

        # Resetting the factor names
        factor_1 = Factor.Factor(factor_1.name[:size_1], factor_1.values)
        factor_2 = Factor.Factor(factor_2.name[:size_2], factor_2.values)
        product_name = ''.join(nodes)
        product_values = [0] * pow(2, len(product_name))



        # What if the factor after trimming is actually of zero size?
        if size_1 == 0:
            return Factor.Factor(factor_2.name + ''.join(remaining_nodes), factor_2.values)
        elif size_2 == 0:
            return Factor.Factor(factor_1.name + ''.join(remaining_nodes), factor_1.values)

        for pos in xrange(len(product_values)):  # For each row in the factor table
            pos_1 = 0  # What row in factor_1 table?
            pos_2 = 0  # What row in factor_2 table?
            for node_pos in xrange(len(nodes)):  # For each node in the product factor
                if pos & pow(2, len(nodes) - 1 - node_pos) != 0:
                    # The current node in the product, is set as 'f'
                    if nodes[node_pos] in nodes_1:
                        node_1_pos = nodes_1.index(nodes[node_pos])
                        pos_1 += pow(2, len(nodes_1) - 1 - node_1_pos)
                        # Narrow down to the correct row
                    if nodes[node_pos] in nodes_2:
                        node_2_pos = nodes_2.index(nodes[node_pos])
                        pos_2 += pow(2, len(nodes_2) - 1 - node_2_pos)
                        # Narrow down to the correct row
            product_values[pos] = factor_1.values[pos_1] * factor_2.values[pos_2]
        return Factor.Factor(product_name + ''.join(remaining_nodes), product_values)



def get_max_cliques(factors, elimination_order):
    """
    Given the elimination order, use node elimination to return a list of maximal elimination cliques.
    """

    factors = set(factors)
    cliques = []
    for variable in elimination_order:
        assigned_factors = set()
        for factor in factors:
            if variable in factor.name:
                assigned_factors = assigned_factors.union([factor])
        factors = factors - assigned_factors # Remove assigned factors
        product = None
        for factor in assigned_factors:
            product = multiply_factors(product, factor)
        cliques = cliques + [product] # Add new clique
        factors = factors.union([Factor.Factor(''.join(set(product.name) - set([variable])), [])]) # Adds a dummy message factor
    return factors_to_cliques(cliques)




def get_junction_tree(max_cliques):
    """
    From the given maximal elimination cliques (nodes), compute maximum weight spanning tree.
    Return a Junction_Tree() object with nodes as cliques and edges as neighbours of each clique corresponding to the computed maximum weight spanning tree.
    """
    if len(max_cliques) == 1:
        neighbours = dict()
        neighbours[max_cliques[0]] = None
        return Junction_Trees.Junction_Trees(max_edge, neighbours)
    # TODO
    neighbours = dict()
    new_vertex_1 = -1
    new_vertex_2 = -1
    j = 0
    max_edge = 0
    while j < len(max_cliques):
        k = j + 1
        while k < len(max_cliques):
            i = len(set(max_cliques[j].name).intersection(max_cliques[k].name))
            if i > max_edge:
                new_vertex_1 = j
                new_vertex_2 = k
                max_edge = i
            k += 1
        j += 1
    neighbours[max_cliques[new_vertex_1]] = [max_cliques[new_vertex_2]]
    neighbours[max_cliques[new_vertex_2]] = [max_cliques[new_vertex_1]]
    # Chose largest edge

    # Choose subsequent edges
    while len(neighbours) < len(max_cliques) :
        max_edge = 0
        new_vertex = None
        pivot = None
        # Iterate over all possible edges onto the tree
        # and choose the heaviest edge
        for tree_vertex in neighbours.keys():
            k = 0
            for vertex in max_cliques:
                if vertex not in neighbours.keys():
                    # tree_vertex is in tree, vertex is not in the tree
                    i = len(set(tree_vertex.name).intersection(vertex.name))
                    if i > max_edge:
                        new_vertex = vertex
                        pivot = tree_vertex
                        max_edge = i

        # Add two new edges
        neighbours[pivot] += [new_vertex]
        neighbours[new_vertex] = [pivot]
    return Junction_Tree.Junction_Tree(max_cliques, neighbours);










