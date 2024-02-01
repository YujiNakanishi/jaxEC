import jax.numpy as jnp
import copy

from jaxEC import selection

class Solution:
    """
    class for individuals management
    att:
        individuals -> <jnp:num:(N, D)> lift of individual
                        N -> <int> number of individual
                        D -> <int> dim of representation
        generation -> <int> num of generation
    """
    def __init__(self, individuals, generation = 1):
        self.individuals = copy.deepcopy(individuals)
        self.generation = generation
    
    def __call__(self):
        return self.individuals
    
    def __len__(self):
        return len(self.individuals)
    
    def dim(self):
        return len(self.individuals[0])
    
    def shape(self):
        return self.individuals.shape
    
    def __getitem__(self, idx):
        return self.individuals[idx]