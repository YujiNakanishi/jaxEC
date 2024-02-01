import jax.numpy as jnp
from jax import jit
import jax

@jit
def gauss(key, solution, sigma):
    """
    output gaussian noise for mutation

    input:
        key -> PRNG key
        solution -> <jnp:num:(N, D)>
        sigma -> <float> standard deviation
        shape -> <tuple> (N, D)
    
    output:
        <jnp:float:shape> mutated solution
    """
    return solution + sigma*jax.random.normal(key, solution.shape)

@jit
def selfAdaptive(key, solution, epsilon, generation):
    """
    output mutation noise by self adaptive method

    input:
        key -> PRNG key
        solution -> <jnp:num:(N, D)>
        epsilon -> <float> min value of standard deviation
        generation -> <int> generation number
    
    output:
        <jnp:float:(N, D)> mutated solution
    Note:
    ## generation is used for calculation tau = 1/sqrt(generation). tau's definition is recomended in the book.
    ## solution[:,-1] should mean sigma(= adaptive standard deviation)
    """
    tau = 1./jnp.sqrt(generation)

    sigma = solution[:,-1]
    sigma *= jnp.exp(tau*jax.random.normal(key, (len(solution),)))
    sigma = (sigma*(sigma >= epsilon) + epsilon*(sigma < epsilon)).reshape((-1, 1))

    representation = solution[:,:-1]
    _, subkey = jax.random.split(key)
    representation += sigma*jax.random.normal(subkey, representation.shape)

    return jnp.concatenate((representation, sigma), axis = 1)