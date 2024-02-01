import jax.numpy as jnp
from jax import jit
import jax
from functools import partial

@partial(jit, static_argnames = ["selection_num"])
def greatest(solution, fitness, selection_num):
    """
    output solution which individuals are selection_num of greatest one with respect to fitness

    input:
        solution -> <jnp:num:(N, D)> list of individual
                    N -> <int> number of individual
                    D -> <int> dim of individual
        fitness -> <jnp:num:(N, )> fitness value of each individuals
        selection_num -> <int> number of selected individuals
        key -> a PRNG key
    output:
        survivor -> <jnp:num:(selection_num, D)> greatest solution
    """
    index = jnp.argsort(fitness)
    survivor = solution[index[:selection_num]]

    return survivor

@partial(jit, static_argnames = ["selection_num"])
def FPS(key, solution, fitness, selection_num):
    """
    output solution by FPS

    input:
        solution -> <jnp:num:(N, D)> list of individual
                    N -> <int> number of individual
                    D -> <int> dim of individual
        fitness -> <jnp:num:(N, )> fitness value of each individuals
        selection_num -> <int> number of selected individuals
        key -> a PRNG key
    output:
        survivor -> <jnp:num:(selection_num, D)> greatest solution
    """
    probability = fitness/jnp.sum(fitness)
    survivor = jax.random.choice(key, solution, (selection_num,), p = probability, axis = 0)

    return survivor