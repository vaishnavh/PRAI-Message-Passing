"""
Author: Santhosh Kumar M (CS09B042)
File: Utils.py
"""

import Factor
import Node
import Message

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
            factor_values = [0] * 2
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
            factor_values = [0] * pow(2, n_vars + 1)
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
                factor_values[n_pos] = 1 - row_val
                j += 1
            factors[factor_name] = Factor.Factor(factor_name, factor_values)
    return factors.values()




def read_file(input_file):
    """
    Reads the given input file.
    Performs moralization to convert to undirected graph if given graph is directed.
    Assigns values to factors which are stored as a list of objects (factors) of class Factor.
    Creates a list of objects (nodes) of class Node corresponding to the random variables in the graph.
    Returns the list of Factor() objects (factors) and the list of Node() objects.
    """
    f = open(input_file)  # Opens the input file
    lines = f.readlines()  # Reads lines of the file and stores it in a list
    lines = filter(None, [line.strip(' \n\t') for line in lines])  # Strips the lines
    # of whitespaces and compresses the list
    is_directed = 0
    if lines[0] == 'DIRECTED':
        is_directed = 1
    lines = lines[4:]  # Removes first four dummy lines
    nodes = dict()
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
            if not nodes.has_key(child_node):
                nodes[child_node] = Node.Node(child_node, Message.Message([1.0, 1.0]), Message.Message([1.0, 1.0]))  # TODO
        i += 1  # Skips the header line (? TODO)
        if len(lines[i].split()) > 1:  # To skip cases like P(B = t)
            # How many variables are there!? Read the next line! :)
            n_vars = len(lines[i + 1].split()) - 1
            vars_in_factor = lines[i].split()[0:n_vars]
            for var_in_factor in vars_in_factor:
                # If the factor has not been recorded yet...
                if not nodes.has_key(var_in_factor):
                    nodes[var_in_factor] = Node.Node(var_in_factor, Message.Message([1.0, 1.0]), Message.Message([1.0, 1.0]))  # TODO
            j = i + 1
            i += pow(2, n_vars)  # Number of entries expected in table + header
            #nodes now contains all the nodes as a dictionary
        i += 1 # Regardless of the case being P(B = t) or P(B = t | A, C) or
        # P(A, B, C) ...
    if is_directed == 1:
        return [moralize_graph(input_file), nodes.values()]
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
            factor_values = [0] * pow(2, n_vars)
            while j < i:
                row = lines[j].split()
                row_pos = row[:n_vars]  # A list of t t f etc,
                row_val = row[n_vars]  # Value of the factor
                pos = 0
                for k in xrange(len(row_pos)):
                    pos = (pos * 2) + (row_pos[k] == 'f')
                factor_values[pos] = row_val
                j += 1
            factors[factor_name] = Factor.Factor(factor_name, factor_values)
        return [factors.values(), nodes.values()]























