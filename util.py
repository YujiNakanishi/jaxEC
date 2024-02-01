import jax
import jax.numpy as jnp
from jax import jit

@jit
def sort(solution, fitness):
    """
    sort solution
    """
    sorted_idx = jnp.argsort(fitness)
    return solution[sorted_idx]