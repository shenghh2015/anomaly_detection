import numpy as np


def ssim(x, y):
    """ mesuare the similarity between inputs of an autoencoder and 
    	outputs of the antoencoder

    Args:
        x: dimension: n x M x K x 1
        y: dimension: n x M x K x 1

    Returns:
        sim_vect: dimension: n x1

    """
	