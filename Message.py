"""
Author: Santhosh Kumar M (CS09B042)
File: Message.py
"""

class Message():
    """
    Class to represent messages for message passing in a tree.
    """

    def __init__(self, name, from_clique, to_clique, message_vector):
        """
        Define a message as going from 'from_clique' clique to 'to_clique' clique.
        Initializes the messages corresponding to each of the values that the variables in the sepset can take.
        """
        self.name = name
        self.from_clique = from_clique
        self.to_clique = to_clique
        self.message_vector = message_vector






