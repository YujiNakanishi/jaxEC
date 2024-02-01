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
    probability = 1./(fitness*jnp.sum(1./fitness))
    survivor = jax.random.choice(key, solution, (selection_num,), p = probability, axis = 0)

    return survivor


@partial(jit, static_argnames = ["selection_num", "k"])
def tournament(key, solution, fitness, selection_num, k):
    """
    output solution by tournament

    input:
        solution -> <jnp:num:(N, D)> list of individual
                    N -> <int> number of individual
                    D -> <int> dim of individual
        fitness -> <jnp:num:(N, )> fitness value of each individuals
        selection_num -> <int> number of selected individuals
        key -> a PRNG key
        k -> <int> number of matching individual
    output:
        survivor -> <jnp:num:(selection_num, D)> greatest solution
    """

    selected_index = jnp.zeros(selection_num).astype(int)
    for i in range(selection_num):
        player_index = jax.random.choice(key, len(solution), (k,))
        selected_index = selected_index.at[i].set(player_index[jnp.argmin(fitness[player_index])])

        key, subkey = jax.random.split(key)

    return solution[selected_index]