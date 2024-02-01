import jax.numpy as jnp
import copy

class Solution:
    """
    class for individuals management
    att:
        individuals -> <list:jnp> lift of individual
                        <jnp:num:(D, )> representation (D is dim)
        generation -> <int> num of generation
    """
    def __init__(self, individuals, generation = 1):
        self.individuals = copy.deepcopy(individuals)
        self.generation = generation
    
    def __len__(self):
        return len(self.individuals)
    
    def dim(self):
        return len(self.individuals[0])
    
    def shape(self):
        return (len(self), self.dim())
    
    def __getitem__(self, idx):
        return self.individuals[idx]