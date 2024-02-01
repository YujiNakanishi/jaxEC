import jax.numpy as jnp
from jax import jit
from functools import partial

def Greatest(solution, evaluation, selection_num):
    """
    output solution class which individuals are num of greatest one with respect to evaluation.
    
    input:
        solution -> <solution class>
        evaluation -> <function> vmap function for calculating fitness value
            input:
                x -> <jnp:num:(D, )>
            output:
                scalar
        selection_num -> <int> number of individuals for outputed soluation
    
    output:
        <solution class>
    """
    fitness = evaluation(solution()) #<jnp:num:(N, )>
    index = jnp.argsort(fitness)
    survivor = solution()[index[:selection_num]]

    return type(solution)(survivor, solution.generation)