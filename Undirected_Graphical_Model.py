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
        #node has occured in the factor
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
        sum_of_factors = sum(factor.values)
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

        var_pos =  factor.name.index(variable) # TODO: Node or variable!?
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
        node_names = [node.name for node in self.nodes]
        return list(set(node_names) - set(query_variables))


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
        return marginal



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
        return marginal




    def conditional_inference(self, variable_1, variable_2):
        """
        Performs conditional inference, P(variable_1 | variable_2), on the UGM and returns the conditional probability distribution as a list.
        """
        elimination_order = self.get_elimination_ordering([variable_1, variable_2])


    def forward_message_pass(self):
        """
        Performs forward message passing and fills the alpha message vector of all nodes.
        """


    def backward_message_pass(self):
        """
        Performs backward message passing and fills the beta message vector of all nodes.
        """


    def chain_marginal_inference(self, variable):
        """
        Performs marginal inference, P(variable), on the chain by message passing and returns the marginal probability distribution as a list.
        """


    def chain_consecutive_joint_inference(self, variable_1, variable_2):
        """
        Performs joint inference on the given consecutive nodes, P(variable_1, variable_2), on the chain by message passing and returns the joint probability distribution as a list.
        """


    def chain_non_consecutive_joint_inference(self, variable_1, variable_2):
        """
        Performs joint inference on the given non-consecutive nodes, P(variable_1, variable_2), on the chain by message passing and returns the joint probability distribution as a list.
        """


    def chain_conditional_inference(self, variable_1, variable_2):
        """
        Performs conditional inference, P(variable_1 | variable_2), on the chain by message passing and returns the conditional probability distribution as a list.
        """


