import numpy as np

#################################################################
# Implements helper functions
#################################################################

def categorical(p, rng):
    """
    Draws multinomial samples from distribution p
    """
    return np.argmax(rng.multinomial(1,p))

def softmax(z):
    """
    Computes softmax values for each Q-value in x
    """
    # TODO: extend to multi-dimensional input? 
    ex = np.exp(z - np.max(z)) 
    return ex / np.sum(ex)
