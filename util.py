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


def split(key, solution, num_pare):
    """
    split solution

    input:
        key -> PRNG key
        solution -> <jnp:num:(N, D)>
        num_pare -> <int>
    
    output:
        <tuple:(num_pare, )>
            <jnp:num:(N/num_pare, D)>
    """
    n = int(len(solution)/num_pare)
    solution = jax.random.permutation(key, solution)

    return (solution[i*n:(i+1)*n] for i in range(num_pare))