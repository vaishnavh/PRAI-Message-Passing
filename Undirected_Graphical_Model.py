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
        Checks whether the given graph is a chain and returns true (or false) respectively.
        """


    def normalize(self, factor):
        """
        Normalizes the values corresponding to the given factor and returns the normalized factor.
        """
	sum_of_factors = sum(factor.values)
	return [x/sum_of_factors for x in factor.values]



    def multiply_factors(self, factor_1, factor_2):
        """
        Multiplies the given two factors and returns the resultant factor.
        """



    def marginalize(self, factor, variable):
        """
        Marginalizes the given factor over the given variable and returns the marginalized factor.
        """


    def get_elimination_ordering(self, query_variables):
        """
        Returns an elimination ordering of the random variables after removing the query variables as a list.
        For example, if random variables are 'B', 'C', 'A' and query variable is 'B', then one elimination order is ['C', 'A'].
        """


    def marginal_inference(self, variable):
        """
        Performs marginal inference, P(variable), on the UGM and returns the marginal probability distribution as a list.
        """
        elimination_order = self.get_elimination_ordering(variable)


    def joint_inference(self, variable_1, variable_2):
        """
        Performs joint inference, P(variable_1, variable_2), on the UGM and returns the joint probability distribution as a list.
        """
        elimination_order = self.get_elimination_ordering([variable_1, variable_2])


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


