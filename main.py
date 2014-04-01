"""
Author: Santhosh Kumar M (CS09B042)
File: main.py 
"""

import Utils
import Factor
import Message
import sys
import Clique
import Junction_Tree

def main():
    input_file = sys.argv[1]

    [factors, variables] = Utils.read_file(input_file)

    if Utils.is_tree(factors) == True:
        # Convert factors to cliques.
        cliques = Utils.factors_to_cliques(factors)

        # Compute neighbours of each clique in the junction tree.
        neighbours = Utils.compute_neighbours(cliques)

        # Create a junction tree
        junction_tree = Junction_Tree.Junction_Tree(cliques, neighbours)

    # Junction tree algorithm
    else:
        elimination_order = Utils.get_elimination_ordering(variables, [])

        # Form a set of maximal elimination cliques
        max_cliques = Utils.get_max_cliques(factors, elimination_order)

        # Create a clique tree over maximal elimination cliques.
        junction_tree = Utils.get_junction_tree(max_cliques)

    # Message passing
    junction_tree.message_passing()

    # Marginal inference
    marginal_prob_dist = junction_tree.marginal_inference('A')

    # Joint inference
    joint_prob_dist = junction_tree.joint_inference('A', 'B')

    # Conditional inference
    conditional_prob_dist = junction_tree.conditional_inference('A', 'B')

    


if __name__ == "__main__":
    main()
