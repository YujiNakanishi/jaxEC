import jax.numpy as jnp
from jax import jit
import jax

def gauss(key, sigma, solution_num, dim):
    """
    output gaussian noise for mutation

    input:
        key -> PRNG key
        sigma -> <float> standard deviation
        shape -> <tuple> (N, D)
    
    output:
        <jnp:float:shape>
    """
    return sigma*jax.random.normal(key, (solution_num, dim))