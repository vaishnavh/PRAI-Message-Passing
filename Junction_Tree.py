"""
Author: Santhosh Kumar M (CS09B042)
File: Clique_Tree.py
"""

import Message
import Clique
import Factor
import Utils

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
        self.cliques = neighbours.keys()
        self.neighbours = neighbours

        # Initialize messages using objects of Message() class - code goes here.
        self.messages = []
        for clique in self.cliques:
            for nbr in self.neighbours[clique]:
                self.messages += [Message.Message(''.join(set(clique.name) - set(nbr.name)), clique, nbr, [])]

    def normalize(self, clique):
        """
        Normalizes the values corresponding to the given clique and returns the normalized clique.
        """
        return Clique.Clique(clique.name, [v/sum(clique.values) for v in clique.values])



    def multiply_messages_clique(self, messages, clique):
        """
        Multiplies the given list of incoming messages and the given clique and returns the resultant clique.
        """
        product = Clique.Clique(clique.name, clique.values)
        for message in messages:
            if message.to_clique == clique: # Must be true by default
                size_1 = 0
                #Set the actual scopes
                while (pow(2, size_1) < len(product.values)):
                    size_1 += 1

                size_2 = 0
                #Set the actual scopes
                while (pow(2, size_2) < len(message.message_vector)):
                    size_2 += 1

                nodes_1 = list(product.name)[:size_1]  # Nodes participating in factor 1
                nodes_2 = list(message.name)[:size_2]  # Nodes participating in factor 2
                nodes = list(set(nodes_1 + nodes_2))  # Gets only the unique nodes participating
                # in both factors

                remaining_nodes = list(set(product.name + message.name) - set(nodes))

                # Resetting the factor names
                factor_1 = Clique.Clique(product.name[:size_1], product.values)
                factor_2 = Clique.Clique(message.name[:size_2], message.message_vector)
                product_name = ''.join(nodes)
                product_values = [0] * pow(2, len(product_name))


                if size_1 == 0:
                    product = Clique.Clique(message.name[:size_2] + ''.join(remaining_nodes), factor_2.values)
                elif size_2 == 0:
                    product = Clique.Clique(product.name[:size_1] + ''.join(remaining_nodes), factor_1.values)
                else:

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
        """
        size_1 = 0
        # Set the actual scopes
        while (pow(2, size_1) < len(product.values)):
            size_1 += 1
        product.name = product.name[:size_1]
        """
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
        clique = Clique.Clique(clique.name[:size], clique.values)
        if variable not in clique.name[:size]:
            return clique
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
        # We'll maintain a list of nodes which have passed
        # the upward and downward messages
        upward = []
        downward = []

        #Finale signalled by passing of all downward messages
        while len(downward) < len(self.cliques):
            for clique in self.cliques:
                # Not done already
                degree = len(self.neighbours[clique])
                # Is ready?
                nbrs_done = len(set(upward).intersection(self.neighbours[clique]))
                send_downward = False
                if clique not in upward and nbrs_done == degree - 1:
                    # Ready to set upward message
                    incoming_messages = [m for m in self.messages if m.to_clique == clique and m.from_clique in upward]
                    outgoing_message = [m for m in self.messages if m.from_clique == clique and m.to_clique not in upward][0]
                    upward += [clique]  # Add to nodes that have sent upward message
                    product = self.multiply_messages_clique(incoming_messages, clique)
                    marg = list(set(product.name) - set(outgoing_message.to_clique.name))
                    for variable in marg:
                        product = self.marginalize(product, variable)
                    # Multiplied and marginalized over non sep set variables
                    outgoing_message.name = product.name
                    outgoing_message.message_vector = product.values
                elif clique not in upward and nbrs_done == degree:
                    # Root
                    upward += [clique]
                    send_downward = True
                elif clique in upward and clique not in downward and nbrs_done == degree:
                    # If upward pass is over everywhere around
                    # now check if parent is done with downward pass
                    send_downward = (len([nbr for nbr in self.neighbours[clique] if nbr in downward]) == 1)
                if send_downward:
                    # Can send downward message
                    in_messages = [m for m in self.messages if m.to_clique == clique]
                    # Pass every downward message
                    for to_clique in self.neighbours[clique]:
                        if to_clique not in downward:
                            outgoing_message = [m for m in self.messages if m.to_clique == to_clique and m.from_clique == clique][0]
                            incoming_messages = [m for m in in_messages if m.from_clique != to_clique]

                            product = self.multiply_messages_clique(incoming_messages, clique)

                            #Marginalize over non-sep-set variables
                            marg = list(set(product.name) - set(outgoing_message.to_clique.name))
                            for variable in marg:
                                product = self.marginalize(product, variable)
                            outgoing_message.name = product.name
                            outgoing_message.message_vector = product.values
                    downward += [clique]

    def marginal_inference(self, variable):
        """
        Performs marginal inference, P(variable), in the junction tree and returns the marginal probability distribution as a list.
        """
        # Where all does this variable occur?
        relevant_cliques = [c for c in self.cliques if variable in c.name]
        min_size = min([len(c.name) for c in relevant_cliques])
        # Choose the smallest clique that contains this variable
        minimum_clique = [c for c in relevant_cliques if len(c.name) == min_size][0]
        # Multiply all incoming messages
        incoming_messages = [m for m in self.messages if m.to_clique == minimum_clique]
        minimum_clique = self.multiply_messages_clique(incoming_messages, minimum_clique)
        eliminate = minimum_clique.name
        result = minimum_clique
        # Marginalize all other variables in the local joint probability
        for marg in eliminate:
            if marg != variable:
                result = self.marginalize(result, marg)
        return self.normalize(result).values

    def joint_inference(self, variables):
        """
        Performs joint inference, P(variables), in the junction tree and returns the joint probability distribution as a list.
        """
        # TODO: take care of re-ordering
        best_clique = None
        for clique in self.cliques:
            if len(set(variables).intersection(clique.name)) == len(variables):
                # We will choose the best clique - smallest clique containing
                # all of them
                if best_clique == None:
                    best_clique = clique
                elif len(best_clique.name) > len(clique.name):
                    best_clique = clique
        if best_clique != None:
            # Some clique was chosen
            product = best_clique
            incoming_messages = [m for m in self.messages if m.to_clique == best_clique]
            # Multiply all incoming messages
            product = self.multiply_messages_clique(incoming_messages, best_clique)
            eliminate = [variable for variable in product.name if variable not in variables]
            for variable in eliminate:
                product = self.marginalize(product, variable)
            # product contains required values
            Z = len(variables)
            product_values = [0]*len(product.values)
            for pos in xrange(len(product.values)):  # For each row in the final table
                pos_c = 0  # What row in clique?
                nodes = variables
                for node_pos in xrange(Z):  # For each node in the final table
                    if pos & pow(2, Z - 1 - node_pos) != 0:
                        # The current node in the product, is set as 'f'
                        node_c_pos = nodes.index(product.name[node_pos])
                        pos_c += pow(2, Z - 1 - node_c_pos)
                        # Narrow down to the correct row
                product_values[pos] = product.values[pos_c]
            product.name = variables
            product.values = product_values
            return self.normalize(product).values
        else:
            # We have no node that contains these variables :(
            # We find a minimal subtree here.
            visited = []
            leaves  = []# Tells me what all are NOT there in the final tree
            to_visit  = [clique for clique in self.cliques if (len(self.neighbours[clique]) == 1)] # We will reduce this bit-by-bit
            # Used to find parent
            # TODO: check for corner cases
            partial_order = [l for l in self.cliques if len(self.neighbours[l]) == 1] #The leaves
            parent_dict = dict()
            #Find partial order
            work_set = list(partial_order)
            while len(partial_order) < len(self.cliques):
                new_work_set = []
                new_partial_order = set()
                for node in work_set:
                    parents = [p for p in self.neighbours[node] if p not in partial_order]
                    if len(parents) > 0:
                        parent = parents[0]
                        parent_dict[node] = parent
                        # Add parent to list ONLY IF every other child of it
                        # is there in the partial order
                        children = [c for c in self.neighbours[parent] if c not in partial_order]
                        if len(children) <= 1:
                            if parent not in partial_order:
                                new_partial_order = new_partial_order.union([parent])
                            if parent not in new_work_set:
                                new_work_set += [parent]
                    else:
                        parent_dict[node] = None
                partial_order += list(new_partial_order)
                work_set = list(new_work_set)

            to_visit = list(partial_order)
            for clique in to_visit:
                # when there's something to visit
                # Find the parent
                # print "Visiting ", clique.name
                rel = len(set(clique.name).intersection(variables))
                parents = [p for p in self.neighbours[clique] if partial_order.index(p) > partial_order.index(clique)]
                unleaved_children = [c for c in self.neighbours[clique] if partial_order.index(c) < partial_order.index(clique) and c not in leaves]
                # For a node, we check if all its neighbours have been visited
                # if yes, then if all of them are leaved out, this note could
                # be leaved out.
                # If more than one neighbour is not visited, visit this node
                # later. Maybe some descendant is unleaved
                # If exactly one neighbour is unvisited, we check whether all
                # the children are leaved. If yes, do something sane.
                # If some children are not leaved, don't leave this and forget
                # going up
                if len(unleaved_children) == 0:
                    # All children in leaves
                    parent = parents[0]
                    if rel == 0:
                        # Useless clique. Visit it and add parent to visit list
                        leaves += [clique] # Which means this wont be included  in the tree
                    else:
                        sepset = set(clique.name).intersection(parent.name)
                        rel_sepset = len(set(variables).intersection(sepset))
                        if rel_sepset == rel:
                            #Useless clique. Go to parent
                            leaves += [clique] # Wont be in final tree

            # Create new tree
            unleaved_cliques = [c for c in self.cliques if c not in leaves]
            # For ever new factor we retain itself * incoming messages /
            # parent-ward message
            factors = []
            nodes = []
            clique_factor = dict()
            for clique in unleaved_cliques:
                # leaf_messages = [m for m in self.messages if m.to_clique = clique and m.from_clique in leaves]
                product_values = self.joint_inference(clique.name) #self.multiply_messages_clique(leaf_messages, clique)
                new_factor = Factor.Factor(clique.name, product_values)
                clique_factor[clique] = new_factor
                factors += [new_factor]
                nodes += list(new_factor.name)
            for message in self.messages:
                if message.to_clique in unleaved_cliques and message.from_clique in unleaved_cliques:
                    #Divide clique_factor by this message
                    divisor = Factor.Factor(message.name, [1/v for v in message.message_vector])
                    factor = clique_factor[message.from_clique]
                    quotient = Utils.multiply_factors(factor, divisor)
                    factor.name = quotient.name
                    factor.values = quotient.values
            elimination_order = Utils.get_elimination_ordering(list(set(nodes)), variables, factors)
            new_cliques = Utils.get_max_cliques(factors, elimination_order)
            new_junction_tree = Utils.get_junction_tree(new_cliques)
            new_junction_tree.message_passing()
            return new_junction_tree.joint_inference(variables)









    def conditional_inference(self, variable_1, variable_2):
        """
        Performs conditional inference, P(variable_1 | variable_2), in the junction tree and returns the conditional probability distribution as a list.
        """
        joint = self.joint_inference([variable_1, variable_2])
        marginal = self.marginalize(Clique.Clique(variable_1 + variable_2, joint), variable_1).values
        joint[0] /= marginal[0]
        joint[1] /= marginal[1]
        joint[2] /= marginal[0]
        joint[3] /= marginal[1]
        return joint






