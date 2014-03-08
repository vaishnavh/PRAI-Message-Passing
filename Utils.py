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


def read_file(input_file):
    """
    Reads the given input file.
    Performs moralization to convert to undirected graph if given graph is directed.
    Assigns values to factors which are stored as a list of objects (factors) of class Factor.
    Creates a list of objects (nodes) of class Node corresponding to the random variables in the graph.
    Returns the list of Factor() objects (factors) and the list of Node() objects.
    """
    f = open(input_file) #Opens the input file
    lines = f.readlines() #Reads lines of the file and stores it in a list
    lines = filter(None,[line.strip(' \n\t') for line in lines]) #Strips the lines of whitespaces and compresse
    #s the list
    if lines[1] == 'DIRECTED':








