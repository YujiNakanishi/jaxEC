"""
single optimization, real vector, unconstrained

objective function : f(x) = \sum x_i^2
"""

import jax
import jax.numpy as jnp
import jaxEC
from jax import jit, vmap
import sys

def func(x):
    return jnp.sum(x**2)
vfunc = vmap(jit(func))

key = jax.random.PRNGKey(0)
solution = jax.random.uniform(key, (100, 4))
key, subkey = jax.random.split(key)

for itr in range(100):
    fitness = vfunc(solution)
    solution = jaxEC.selection.FPS(key, solution, fitness, 100)
    key, subkey = jax.random.split(key)
    solution = jaxEC.crossover.real.BLX_alpha(key, solution)
    key, subkey = jax.random.split(key)
    solution = jaxEC.mutation.real.gauss(key, solution, 0.01)
    key, subkey = jax.random.split(key)

    solution = jaxEC.sort(solution, vfunc(solution))
    print(vfunc(solution)[0])