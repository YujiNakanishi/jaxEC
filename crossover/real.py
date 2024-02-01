import jax
from jax import jit
import jax.numpy as jnp
import sys

from jaxEC import util

@jit
def getOffspring_linear(father, mother):
    """
    function for linear. create offspring.
    """
    offs1 = 0.5*(father + mother)
    offs2 = 1.5*father-0.5*mother
    offs3 = 1.5*mother-0.5*father

    return jnp.stack((offs1, offs2, offs3), axis = 1)


@jit
def getTwo_linear(fitness, offspring):
    """
    function for linear. select two offsprings from three
    """
    index = jnp.argsort(fitness, axis = 1) #(N/2, 3)
    offs1 = offspring[jnp.arange(len(offspring)),index[:,0]]
    offs2 = offspring[jnp.arange(len(offspring)),index[:,1]]

    return jnp.concatenate((offs1, offs2))


def linear(key, solution, fitness_function):
    """
    linear crossover.
    create offsprings which are [0.5father + 0.5father, 1.5father - 0.5father, -0.5father + 1.5father]
    select two offsprings from three

    input:
        key -> PRNG key
        solution -> <jnp:num:(N, D)>
        fitness_function -> <vmaped function>
    
    output:
        <jnp:num:(N, D)> new solution
    """
    father, mother = util.split(key, solution, 2)

    offspring = getOffspring_linear(father, mother) #(N/2, 3, D)

    fit1 = fitness_function(offspring[:,0]) #(N/2, )
    fit2 = fitness_function(offspring[:,1])
    fit3 = fitness_function(offspring[:,2])
    fitness = jnp.stack((fit1, fit2, fit3), axis = 1) #(N/2, 3)
    
    return getTwo_linear(fitness, offspring)

@jit
def _BLX_alpha(key, father, mother, alpha):
    """
    sub function for BLX_alpha
    """
    center = 0.5*(father + mother) #(N/2, D)
    delta = jnp.abs(father - mother)

    u1 = jax.random.uniform(key, center.shape)
    key, subkey = jax.random.split(key)
    u2 = jax.random.uniform(key, center.shape)

    offspring1 = center+(2.*u1-1.)*(0.5+alpha)*delta #(N/2, D)
    offspring2 = center+(2.*u2-1.)*(0.5+alpha)*delta

    return jnp.concatenate((offspring1, offspring2))


def BLX_alpha(key, solution, alpha = 0.5):
    """
    create offsprings by BLX_alpha

    input:
        key, solution -> as usual
        alpha -> <float> hyper parameter
    
    output:
        <jnp:num:(N, D)> new solution
    """
    father, mother = util.split(key, solution, 2)
    key, subkey = jax.random.split(key)

    return _BLX_alpha(key, father, mother, alpha)