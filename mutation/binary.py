import jax.numpy as jnp
import jax
from jax import jit

@jit
def flip(key, solution, p):
    """
    flip mutation

    input:
        key, solution -> as usual
        p -> <float> probability of flipping
    """
    probability = jax.random.uniform(key, solution.shape)
    mask = (probability < p).astype(int)

    mutation = solution + mask
    mutation = (mutation == 1).astype(int)

    return mutation