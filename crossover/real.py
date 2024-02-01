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
