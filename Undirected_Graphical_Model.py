"""
Author: Santhosh Kumar M (CS09B042)
File: Undirected_Graphical_Model.py
"""

import Factor
import Node

class Undirected_Graphical_Model():
    """
    Class to construct an undirected graphical model (UGM) and perform exact inference (variable elimination) on it.
    """

    def __init__(self, factors, nodes):
        """
        Initializes the UGM with the given factors list and nodes list.
        """
        self.factors = factors
        self.nodes = nodes


    def is_chain(self):
        """
        Checks whether the given graph is a chain and returns true (or False) respectively.
        """
        node_names = [node.name  for node in self.nodes]
        factor_names = [factor.name for factor in self.factors]
        node_count = dict()  # A count of how many times a
        #node has occured in a two-sized factor
        for factor_name in factor_names:
            if len(factor_name) > 2:
                # A chain cannot have a factor of more than three
                # variables
                return False
            elif len(factor_name) == 2:
                # Don't look at single variable factors!
                for i in xrange(2):
                    if factor_name[i] in node_count.keys():
                        if node_count[factor_name[i]] == 2:
                            # Break if a variable has occured
                            # twice already!
                            return False
                        node_count[factor_name[i]] += 1
                    else:
                        node_count[factor_name[i]] = 1
        leaves = 0
        for node_name in node_names:
            if node_name not in node_count.keys():
                # The node doesn't occur in any factor.
                # Must not occur ideally.
                return False
            elif node_count[node_name] > 2:
                # Node occurs in more than two two-sized
                # factors.
                return False
            elif node_count[node_name] == 1:
                # Node occurs in exactly one two=sized
                # factor. Could be a leaf node.
                leaves += 1
        return (leaves == 2)  # Exactly two leaves must exist!


    def normalize(self, factor):
        """
        Normalizes the values corresponding to the given factor and returns the normalized factor.
        """
        sum_of_factors = float(sum(factor.values))
        return Factor.Factor(factor.name, [float(x)/float(sum_of_factors) for x in factor.values])


    def multiply_factors(self, factor_1, factor_2):
        """
        Multiplies the given two factors and returns the resultant factor.
        """
        nodes_1 = [c for c in factor_1.name]  # Nodes participating in factor 1
        nodes_2 = [c for c in factor_2.name]  # Nodes participating in factor 2
        nodes = list(set(nodes_1 + nodes_2))  # Gets only the unique nodes participating
        # in both factors
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
        return Factor.Factor(product_name, product_values)


    def marginalize(self, factor, variable):
        """
        Marginalizes the given factor over the given variable and returns the marginalized factor.
        """
        var_pos =  factor.name.index(variable) # TODO: Node or variable!? Resolved. Variable.
        marg_nodes = [c for c in factor.name]
        marg_nodes = marg_nodes[:var_pos] + marg_nodes[(var_pos+1):] # Remove variable from the list
        marg_values = [0] * pow(2, len(marg_nodes))
        for pos in xrange(len(factor.values)):
            marg_pos = 0  # Where this corresponds to in the marginal table
            for node_pos in xrange(len(factor.name)):
                if (node_pos != var_pos) and (pos & pow(2, len(factor.name) - 1 - node_pos) != 0):
                    marg_node_pos =  marg_nodes.index(factor.name[node_pos])
                    marg_pos += pow(2, len(marg_nodes) - 1 - marg_node_pos)
            marg_values[marg_pos] += factor.values[pos]  #Sums over variable
        return Factor.Factor(''.join(marg_nodes), marg_values)


    def get_elimination_ordering(self, query_variables):
        """
        Returns an elimination ordering of the random variables after removing the query variables as a list.
        For example, if random variables are 'B', 'C', 'A' and query variable is 'B', then one elimination order is ['C', 'A'].
        """
        adj = dict()  # An adjacency list of edjes
        for factor in self.factors:
            for node_1 in factor.name:
                for node_2 in factor.name:
                    if node_1 != node_2:
                        # If they are not the same nodes,
                        # draw edge between them
                        if node_1 in adj.keys():
                            adj[node_1] = adj[node_1].union(node_2)
                        else:
                            adj[node_1] = set(node_2)
                        if node_2 in adj.keys():
                            adj[node_2] = adj[node_2].union(node_1)
                        else:
                            adj[node_2] = set(node_1)
        to_elim = set(adj.keys()) - set(query_variables)  # Everything but the query variables have to be eliminated
        elim_order = []  # Our final elimination order
        while len(to_elim) > 0:
            # While there are variables to eliminate
            # we will consider each of them and choose the best
            # based on number of fill in edges
            elim_node = None  # We've not made a decision about what to eliminate
            elim_cost = -1  # The number of fillin edges introduced for the one we are to eliminate
            for node in to_elim:
                # Count the number of pairs of neighbours who are not
                # neighbours
                pair_count = 0
                for nbr_1 in adj[node]:
                    for nbr_2 in adj[node]:
                        #For every pair
                        if (nbr_1 != nbr_2) and (nbr_1 not in adj[nbr_2]):
                            pair_count += 1
                pair_count /= 2
                if elim_node == None:
                    # First in the iteration...
                    elim_node = node
                    elim_cost = pair_count
                elif elim_cost > pair_count:
                    # We found a better node!
                    elim_cost = pair_count
                    elim_node = node
            #  We are done choosing, remove node and add to elim
            to_elim = to_elim - set([elim_node])
            elim_order = elim_order + [elim_node]
            for nbr_1 in adj[elim_node]:
                adj[nbr_1] = adj[nbr_1] - set([elim_node])
                for nbr_2 in adj[elim_node]:
                    if nbr_2 != nbr_1:
                        # Construct all fill in edges
                        adj[nbr_1] = adj[nbr_1].union([nbr_2])
                        adj[nbr_2] = adj[nbr_2].union([nbr_1])
            adj.pop(elim_node, None)  # Delete the node
        return elim_order





    def marginal_inference(self, variable):
        """
        Performs marginal inference, P(variable), on the UGM and returns the marginal probability distribution as a list.
        """
        elimination_order = self.get_elimination_ordering(variable)
        temp_factors = self.factors
        for var_pos in xrange(len(elimination_order)):
            # Eliminate this variable
            factors_involved = [factor for factor in temp_factors if (elimination_order[var_pos] in factor.name)]
            # The above line finds factors that involve the to-be-eliminated
            # variable
            temp_factors = list(set(temp_factors) - set(factors_involved))
            # temp_factors is updated with the remaining factors
            if len(factors_involved) > 0:
                # There is atleast one factor involved
                # Multiply all these factors
                product_factor = factors_involved[len(factors_involved)-1] # Get the last factor
                for factor_pos in xrange(len(factors_involved)-1):
                    product_factor = self.multiply_factors(product_factor, factors_involved[factor_pos])
                # Marginalize over this factor on the  to-be-eliminated variable
                product_factor = self.marginalize(product_factor, elimination_order[var_pos])
                temp_factors = temp_factors + [product_factor] # Add back the product factor
        # Finally on eliminating all the other variables, we might be left with a
        # set of factors dependent only on 'variable'. We need to multiply
        # them.
        marginal = temp_factors[len(temp_factors)-1]
        for factor_pos in xrange(len(temp_factors)-1):
            marginal = self.multiply_factors(marginal, temp_factors[factor_pos])
        return self.normalize(marginal).values


    def joint_inference(self, variable_1, variable_2):
        """
        Performs joint inference, P(variable_1, variable_2), on the UGM and returns the joint probability distribution as a list.
        """
        elimination_order = self.get_elimination_ordering([variable_1, variable_2])
        temp_factors = self.factors
        for var_pos in xrange(len(elimination_order)):
            # Eliminate this variable
            factors_involved = [factor for factor in temp_factors if (elimination_order[var_pos] in factor.name)]
            # The above line finds factors that involve the to-be-eliminated
            # variable
            temp_factors = list(set(temp_factors) - set(factors_involved))
            # temp_factors is updated with the remaining factors
            if len(factors_involved) > 0:
                # There is atleast one factor involved
                # Multiply all these factors
                product_factor = factors_involved[len(factors_involved)-1] # Get the last factor
                for factor_pos in xrange(len(factors_involved)-1):
                    product_factor = self.multiply_factors(product_factor, factors_involved[factor_pos])
                # Marginalize over this factor on the  to-be-eliminated variable
                product_factor = self.marginalize(product_factor, elimination_order[var_pos])
                temp_factors = temp_factors + [product_factor] # Add back the product factor
        # Finally on eliminating all the other variables, we might be left with a
        # set of factors dependent only on 'variable_1' and 'variable_2'. We need to multiply
        # them.
        marginal = temp_factors[len(temp_factors)-1]
        for factor_pos in xrange(len(temp_factors)-1):
            marginal = self.multiply_factors(marginal, temp_factors[factor_pos])

        marginal = self.normalize(marginal)  # Normalize
        #Reorder the variables if not ordered
        if marginal.name[0] != variable_1:
            t = marginal.values[1]
            marginal.values[1] = marginal.values[2]
            marginal.values[2] = t
        return marginal.values


    def conditional_inference(self, variable_1, variable_2):
        """
        Performs conditional inference, P(variable_1 | variable_2), on the UGM and returns the conditional probability distribution as a list.
        """
        elimination_order = self.get_elimination_ordering([variable_1, variable_2])
        temp_factors = self.factors
        for var_pos in xrange(len(elimination_order)):
            # Eliminate this variable
            factors_involved = [factor for factor in temp_factors if (elimination_order[var_pos] in factor.name)]
            # The above line finds factors that involve the to-be-eliminated
            # variable
            temp_factors = list(set(temp_factors) - set(factors_involved))
            # temp_factors is updated with the remaining factors
            if len(factors_involved) > 0:
                # There is atleast one factor involved
                # Multiply all these factors
                product_factor = factors_involved[len(factors_involved)-1] # Get the last factor
                for factor_pos in xrange(len(factors_involved)-1):
                    product_factor = self.multiply_factors(product_factor, factors_involved[factor_pos])
                # Marginalize over this factor on the  to-be-eliminated variable
                product_factor = self.marginalize(product_factor, elimination_order[var_pos])
                temp_factors = temp_factors + [product_factor] # Add back the product factor
        # Finally on eliminating all the other variables, we might be left with a
        # set of factors dependent only on 'variable_1' and 'variable_2'. We need to multiply
        # them.
        marginal = temp_factors[len(temp_factors)-1]
        for factor_pos in xrange(len(temp_factors)-1):
            marginal = self.multiply_factors(marginal, temp_factors[factor_pos])

        marginal = self.normalize(marginal)  # Normalize
        # Marginalize over variable_1
        marginal_1 = self.marginalize(marginal, variable_1)  # This marginal contains only marginal_2 now

        # Reorder the variables if not ordered as variable_1, variable_2
        if marginal.name[0] != variable_1:
            t = marginal.values[1]
            marginal.values[1] = marginal.values[2]
            marginal.values[2] = t
        marginal.values[0] /= marginal_1.values[0]
        marginal.values[1] /= marginal_1.values[1]
        marginal.values[2] /= marginal_1.values[0]
        marginal.values[3] /= marginal_1.values[1]

        return marginal.values


    def forward_message_pass(self):
        """
        Performs forward message passing and fills the alpha message vector of all nodes.
        """
        leaves = str() #A string of two leaves
        for node in self.nodes:
             is_present = [1 for factor in self.factors if (node.name in factor.name) and (len(factor.name) == 2)]
             if sum(is_present) == 1:
                 leaves = leaves + node.name
        # Choosing a leaf to begin with, by making an alphabetical ordering
        begin_leaf = 0
        if leaves[1] < leaves[0]:
            begin_leaf = 1

        # Develop a dictionary to address nodes. Saves time.
        node_dict = dict()
        for node in self.nodes:
            node_dict[node.name] = node

        # Develop a dictionary to address factors a node is associated with.
        factor_dict = dict()
        for factor in self.factors:
            for node_name in factor.name:
                if node_name in factor_dict.keys():
                    factor_dict[node_name] += [factor]
                else:
                    factor_dict[node_name] = [factor]

        # Begin message passing
        curr_node =  leaves[begin_leaf]
        node_dict[curr_node].alpha_message.message_vector = [1.0, 1.0]
        prev_node =  None
        is_reached = False
        while not is_reached: # The condition is put within the loop
            # Find the node that follows
            if prev_node != None:
                next_factors = [factor for factor in factor_dict[curr_node] if  not (prev_node in factor.name)]
            else:
                next_factors = [factor for factor in factor_dict[curr_node]]
            # Next factors are the set of factors that don't contain the previous node
            # but contain the current node!
            prev_node = curr_node

            product_factor = Factor.Factor(prev_node, node_dict[prev_node].alpha_message.message_vector)
            # To begin with \mu_(x_{n-1})


            for factor in factor_dict[curr_node]:
               #Choose the factor that has only the current node x_{n-1} as its variable
               if factor.name == curr_node:
                   product_factor = self.multiply_factors(product_factor, factor)
                   #Mutiply message with \psi{x_n-1}


            #Search for the next node
            for factor in next_factors:
                if len(factor.name) == 2:
                    if factor.name[0] == prev_node:
                        curr_node = factor.name[1]
                    else:
                        curr_node = factor.name[0]
                    product_factor = self.multiply_factors(product_factor, factor)
                    #Multiply message with psi(x_n, x_{n-1})

            #Current and previous nodes have been updated
            node_dict[curr_node].alpha_message.message_vector = self.marginalize(product_factor, prev_node).values
            #The product is marginalized over the previous node


            if curr_node == leaves[1 - begin_leaf]:
                is_reached = 1 #We've reached the other end!


    def backward_message_pass(self):
        """
        Performs backward message passing and fills the beta message vector of all nodes.
        """
        leaves = str() #A string of two leaves
        for node in self.nodes:
             is_present = [1 for factor in self.factors if (node.name in factor.name) and (len(factor.name) == 2)]
             if sum(is_present) == 1:
                 leaves = leaves + node.name
        # Choosing a leaf to begin with, by making an alphabetical ordering
        begin_leaf = 0
        if leaves[1] > leaves[0]:
            begin_leaf = 1

        # Develop a dictionary to address nodes. Saves time.
        node_dict = dict()
        for node in self.nodes:
            node_dict[node.name] = node

        # Develop a dictionary to address factors a node is associated with.
        factor_dict = dict()
        for factor in self.factors:
            for node_name in factor.name:
                if node_name in factor_dict.keys():
                    factor_dict[node_name] += [factor]
                else:
                    factor_dict[node_name] = [factor]

        # Begin message passing
        curr_node =  leaves[begin_leaf]
        node_dict[curr_node].beta_message.message_vector = [1.0, 1.0]
        prev_node =  None
        is_reached = False
        while not is_reached: # The condition is put within the loop
            # Find the node that follows
            if prev_node != None:
                next_factors = [factor for factor in factor_dict[curr_node] if  not (prev_node in factor.name)]
            else:
                next_factors = [factor for factor in factor_dict[curr_node]]
            # Next factors are the set of factors that don't contain the previous node
            # but contain the current node!
            prev_node = curr_node

            product_factor = Factor.Factor(prev_node, node_dict[prev_node].beta_message.message_vector)
            # To begin with \mu_(x_{n-1})


            for factor in factor_dict[curr_node]:
               #Choose the factor that has only the current node x_{n-1} as its variable
               if factor.name == curr_node:
                   product_factor = self.multiply_factors(product_factor, factor)
                   #Mutiply message with \psi{x_n-1}


            #Search for the next node
            for factor in next_factors:
                if len(factor.name) == 2:
                    if factor.name[0] == prev_node:
                        curr_node = factor.name[1]
                    else:
                        curr_node = factor.name[0]
                    product_factor = self.multiply_factors(product_factor, factor)
                    #Multiply message with psi(x_n, x_{n-1})

            #Current and previous nodes have been updated
            node_dict[curr_node].beta_message.message_vector = self.marginalize(product_factor, prev_node).values
            #The product is marginalized over the previous node

            if curr_node == leaves[1 - begin_leaf]:
                is_reached = 1 #We've reached the other end!


    def chain_marginal_inference(self, variable):
        """
        Performs marginal inference, P(variable), on the chain by message passing and returns the marginal probability distribution as a list.
        """
        # run two passes
        self.forward_message_pass()
        self.backward_message_pass()
        product_factor = None
        for node in self.nodes:
            # search for the node
            if node.name == variable:
                # product of alpha and beta messages at the node
                product_factor = self.multiply_factors(Factor.Factor(variable,node.alpha_message.message_vector), Factor.Factor(variable, node.beta_message.message_vector))
                break
        for factor in self.factors:
            # look for factor corresponding to the node alone
            if factor.name == variable:
                product_factor = self.multiply_factors(product_factor, factor)
        return self.normalize(product_factor).values


    def chain_consecutive_joint_inference(self, variable_1, variable_2):
        """
        Performs joint inference on the given consecutive nodes, P(variable_1, variable_2), on the chain by message passing and returns the joint probability distribution as a list.
        """
        # Run both passs
        self.forward_message_pass()
        self.backward_message_pass()
        product_factor = None

        # Develop a dictionary to address nodes. Saves time.
        node_dict = dict()
        for node in self.nodes:
            node_dict[node.name] = node

        # Develop a dictionary to address factors a node is associated with.
        factor_dict = dict()
        for factor in self.factors:
            for node_name in factor.name:
                if node_name in factor_dict.keys():
                    factor_dict[node_name] += [factor]
                else:
                    factor_dict[node_name] = [factor]


        # Find the leaves!
        leaves = str() #A string of two leaves
        for node in self.nodes:
             is_present = [1 for factor in self.factors if (node.name in factor.name) and (len(factor.name) == 2)]
             if sum(is_present) == 1:
                 leaves = leaves + node.name

        # Choosing a leaf to begin with, by making an alphabetical ordering
        begin_leaf = 0
        if leaves[1] < leaves[0]:
            begin_leaf = 1

        # To find which of variable_1 or 2 occurs at the alpha end
        prev_node =  None
        curr_node = leaves[begin_leaf]
        is_reached = False
        while not is_reached: # The condition is put within the loop

            if curr_node == variable_1:
                prev_node = variable_1
                curr_node = variable_2
                break
            elif curr_node == variable_2:
                prev_node = variable_2
                curr_node = variable_1
                break

            # Find the node that follows
            if prev_node != None:
                next_factors = [factor for factor in factor_dict[curr_node] if  not (prev_node in factor.name)]
            else:
                next_factors = [factor for factor in factor_dict[curr_node]]
            # Next factors are the set of factors that don't contain the previous node
            # but contain the current node!
            prev_node = curr_node

            #Search for the next node
            for factor in next_factors:
                if len(factor.name) == 2:
                    if factor.name[0] == prev_node:
                        curr_node = factor.name[1]
                    else:
                        curr_node = factor.name[0]

            #Check if we've reached
            if curr_node == leaves[1 - begin_leaf]:
                is_reached = 1 #We've reached the other end!


        #Now  that we have identified the alpha and beta ends...
        product_factor = self.multiply_factors(Factor.Factor(prev_node, node_dict[prev_node].alpha_message.message_vector), Factor.Factor(curr_node, node_dict[curr_node].beta_message.message_vector))

        for factor in factor_dict[prev_node]:
            if len(factor.name) == 1:
                # \psi_{v_1}
                product_factor = self.multiply_factors(product_factor, factor)
            elif factor in factor_dict[curr_node]:
                # \psi_{v_1, v_2}
                product_factor = self.multiply_factors(product_factor, factor)

        for factor in factor_dict[curr_node]:
            if len(factor.name) == 1:
                # \psi_{v_2}:
                product_factor = self.multiply_factors(product_factor, factor)

        # Make sure that the output is formatted as v_1, v_2
        if product_factor.name[0] != variable_1:
            # Swap 1th and 2th entries.
            t = product_factor.values[1]
            product_factor.values[1] = product_factor.values[2]
            product_factor.values[2] = t

        return self.normalize(product_factor).values


    def chain_non_consecutive_joint_inference(self, variable_1, variable_2):
        """
        Performs joint inference on the given non-consecutive nodes, P(variable_1, variable_2), on the chain by message passing and returns the joint probability distribution as a list.
        """
        self.forward_message_pass()
        self.backward_message_pass()

        # Develop a dictionary to address nodes. Saves time.
        node_dict = dict()
        for node in self.nodes:
            node_dict[node.name] = node

        # Develop a dictionary to address factors a node is associated with.
        factor_dict = dict()
        for factor in self.factors:
            for node_name in factor.name:
                if node_name in factor_dict.keys():
                    factor_dict[node_name] += [factor]
                else:
                    factor_dict[node_name] = [factor]


        # Find the leaves!
        leaves = str() #A string of two leaves
        for node in self.nodes:
             is_present = [1 for factor in self.factors if (node.name in factor.name) and (len(factor.name) == 2)]
             if sum(is_present) == 1:
                 leaves = leaves + node.name

        # Choosing a leaf to begin with, by making an alphabetical ordering
        begin_leaf = 0
        if leaves[1] < leaves[0]:
            begin_leaf = 1

        # To find which of variable_1 or 2 occurs at the alpha end
        prev_node =  None
        curr_node = leaves[begin_leaf]
        is_reached = False
        do_computation = False
        product_factor = None  # Final result
        while not is_reached: # The condition is put within the loop

            # Change state. We've to start recomputing messages now.
            if not do_computation:
                if prev_node == variable_1 or prev_node == variable_2:
                    do_computation = 1
                    product_factor = Factor.Factor(prev_node, node_dict[prev_node].alpha_message.message_vector)
                    # res = \mu_{v_1}
                    for factor in factor_dict[prev_node]:
                        if factor.name == prev_node:
                            product_factor = self.multiply_factors(product_factor, factor)
                            # res = res * \psi_{v_1}

                    for factor in factor_dict[curr_node]:
                        if factor.name == curr_node or factor in factor_dict[prev_node] :
                            # res = res * \psi_{x_1} * \psi_{v_1, x_1}
                            product_factor = self.multiply_factors(product_factor, factor)

            # Invariant here: if we are in do_computation state:
            # product_factor contains only one of variable_1/2 AND some other
            # variable x_i which stands for the curr_node and its \psi_{x_i}
            # and all information between

            if do_computation:
                if curr_node == variable_1 or curr_node == variable_2:
                    #  We've reached the required end
                    product_factor = self.multiply_factors(product_factor, Factor.Factor(curr_node, node_dict[curr_node].beta_message.message_vector))
                    break




            # Find the node that follows
            if prev_node != None:
                next_factors = [factor for factor in factor_dict[curr_node] if  not (prev_node in factor.name)]
            else:
                next_factors = [factor for factor in factor_dict[curr_node]]
            # Next factors are the set of factors that don't contain the previous node
            # but contain the current node!
            prev_node = curr_node

            # Search for the next node
            for factor in next_factors:
                if len(factor.name) == 2:
                    if factor.name[0] == prev_node:
                        curr_node = factor.name[1]
                    else:
                        curr_node = factor.name[0]
            if do_computation:
                for factor in factor_dict[curr_node]:
                    if factor.name == curr_node or factor in next_factors:
                        #  Currently the result contains v_1 and message
                        #  reaching prev_node and all information about it.
                        # Now capture information about the next node
                        product_factor  =  self.multiply_factors(product_factor, factor)
                #We have information about v_1, x_i and x_{i+1}
                #Marginalize over x_i
                product_factor = self.marginalize(product_factor, prev_node)

            #should break out of this loop sometime


        # Make sure that the output is formatted as v_1, v_2
        if product_factor.name[0] != variable_1:
            # Swap 1th and 2th entries.
            t = product_factor.values[1]
            product_factor.values[1] = product_factor.values[2]
            product_factor.values[2] = t


        return self.normalize(product_factor).values




    def chain_conditional_inference(self, variable_1, variable_2):
        """
        Performs conditional inference, P(variable_1 | variable_2), on the chain by message passing and returns the conditional probability distribution as a list.
        """
        joint = self.chain_non_consecutive_joint_inference(variable_1, variable_2)
        marginal = self.marginalize(Factor.Factor(variable_1 + variable_2, joint), variable_1).values
        joint[0] /= marginal[0]
        joint[1] /= marginal[1]
        joint[2] /= marginal[0]
        joint[3] /= marginal[1]
        return joint



