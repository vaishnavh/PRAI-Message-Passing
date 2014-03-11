"""
Author: Santhosh Kumar M (CS09B042)
File: main.py
"""

import Undirected_Graphical_Model
import Utils
import Factor
import Node
import Message
import sys

def main():
    input_file = sys.argv[1]

    [factors, nodes] = Utils.read_file(input_file)

    ugm = Undirected_Graphical_Model.Undirected_Graphical_Model(factors, nodes)

    if ugm.is_chain() == False:

        marginal_prob_dist = ugm.marginal_inference('A')

        joint_prob_dist = ugm.joint_inference('A', 'B')

        conditional_prob_dist = ugm.conditional_inference('A', 'B')

        print marginal_prob_dist, joint_prob_dist, conditional_prob_dist
    else:

        chain_marginal_prob_dist = ugm.chain_marginal_inference('A')

        chain_consecutive_prob_dist = ugm.chain_consecutive_joint_inference('A', 'B')

        chain_non_consecutive_prob_dist = ugm.chain_non_consecutive_joint_inference('A', 'D')
        print chain_marginal_prob_dist, chain_consecutive_prob_dist, chain_non_consecutive_prob_dist



if __name__ == "__main__":
    main()
