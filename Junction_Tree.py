"""
Author: Santhosh Kumar M (CS09B042)
File: Clique_Tree.py
"""

import Message
import Clique

class Junction_Tree():
    """
    Class to represent a junction tree and perform message passing in it.
    """

    def __init__(self, cliques, neighbours):
        """
        Initializes the cliques and neighbours of each clique in the junction tree with the given list of Clique() objects and neighbours dictionary.
        Defines messages between all pairs of cliques which are adjacent in the clique tree.
        For example,
        cliques = [clique_1, clique_2, clique_3] wherein clique_1.name = 'AB', clique_2.name = 'BDE', clique_3.name = 'AC'
        neighbours = {clique_1 : [clique_2, clique_3], clique_2 : [clique_1], clique_3 : [clique_1]}
        """
        self.cliques = cliques
        self.neighbours = neighbours

        # Initialize messages using objects of Message() class - code goes here.
        self.messages = []
        for clique in self.cliques:
            for nbr in self.neighbours[clique]:
                self.messages += Message.Message(''.join(set(clique.name) - set(nbr.name)), []);

    def normalize(self, clique):
        """
        Normalizes the values corresponding to the given clique and returns the normalized clique.
        """
        return Clique.Clique(clique.name, clique.values/sum(clique.values))



    def multiply_messages_clique(self, messages, clique):
        """
        Multiplies the given list of incoming messages and the given clique and returns the resultant clique.
        """
        product = Clique.Clique(clique.name, clique.values)
        for message in messages:
            if to_clique == clique: # Must be true by default
                size_1 = 0
                #Set the actual scopes
                while (pow(2, size_1) < len(factor_1.values)):
                    size_1 += 1
                nodes_1 = list(product.name)[:size_1]  # Nodes participating in factor 1
                nodes_2 = list(message.name)  # Nodes participating in factor 2
                nodes = list(set(nodes_1 + nodes_2))  # Gets only the unique nodes participating
                # in both factors

                remaining_nodes = list(set(product.name + message.name) - nodes)

                # Resetting the factor names
                factor_1 = Clique.Clique(product.name[:size_1], product.values)
                factor_2 = Clique.Clique(message.name, message.values)
                product_name = ''.join(nodes)
                product_values = [0] * pow(2, len(product_name))


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
                product = Clique.Clique(product_name + ''.join(remaining_nodes), product_values)
        return product







    def marginalize(self, clique, variable):
        """
        Marginalizes the given clique over the given variable and returns the marginalized clique.
        """
        scope = clique.name
        # Trimming absent variables
        size = 0
        while pow(2, size) < len(clique.values):
            size += 1
        clique = Clique(clique.name[:size], clique.values)
        var_pos =  clique.name.index(variable) # TODO: Node or variable!? Resolved. Variable.
        marg_nodes = [c for c in clique.name]
        marg_nodes = marg_nodes[:var_pos] + marg_nodes[(var_pos+1):] # Remove variable from the list
        marg_values = [0] * pow(2, len(marg_nodes))
        for pos in xrange(len(clique.values)):
            marg_pos = 0  # Where this corresponds to in the marginal table
            for node_pos in xrange(len(clique.name)):
                if (node_pos != var_pos) and (pos & pow(2, len(clique.name) - 1 - node_pos) != 0):
                    marg_node_pos =  marg_nodes.index(clique.name[node_pos])
                    marg_pos += pow(2, len(marg_nodes) - 1 - marg_node_pos)
            marg_values[marg_pos] += clique.values[pos]  #Sums over variable
        return Clique.Clique(''.join(marg_nodes) + scope[size:], marg_values)





    def message_passing(self):
        """
        Performs message passing in the junction tree.
        """
        upward = []
        #Upward phase
        while len(upward) < len(cliques):
            for clique in self.cliques:
                if clique not in upward:
                    # Not done already
                    degree = len(neighbours[clique])
                    # Is ready?
                    nbrs_done = len(set(upward).intersection(neighbours[clique]))
                    if nbrs_done == degree - 1:
                        # Ready to pass upward message





    def marginal_inference(self, variable):
        """
        Performs marginal inference, P(variable), in the junction tree and returns the marginal probability distribution as a list.
        """


    def joint_inference(self, variables):
        """
        Performs joint inference, P(variables), in the junction tree and returns the joint probability distribution as a list.
        """


    def conditional_inference(self, variable_1, variable_2):
        """
        Performs conditional inference, P(variable_1 | variable_2), in the junction tree and returns the conditional probability distribution as a list.
        """


