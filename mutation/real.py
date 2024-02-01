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
    ## generation is used for calculating tau = 1/sqrt(generation). tau's definition is recomended in the book.
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


@jit
def selfAdaptive_vector(key, solution, epsilon, generation):
    """
    output mutation noise by self adaptive vector method

    input:
        key -> PRNG key
        solution -> <jnp:num:(N, D)>
        epsilon -> <jnp:num:(D, )> min value of standard deviation of each element
        generation -> <int> generation number
    
    output:
        <jnp:float:(N, D)> mutated solution
    Note:
    ## generation is used for calculating tau1 and tau2.
    ## solution[:,-int(D/2):] should mean sigma(= adaptive standard deviation)
    """
    N, D = solution.shape
    tau1 = 1./jnp.sqrt(2.*generation)
    tau2 = 1./jnp.sqrt(2.*jnp.sqrt(generation))

    sigma = solution[:,-int(D/2):]

    global_normal = jax.random.normal(key, (N, int(D/2)))
    key, subkey = jax.random.split(key)
    local_normal = jax.random.normal(key, (N, int(D/2)))

    sigma *= jnp.exp(tau1*global_normal + tau2*local_normal)
    sigma = sigma*(sigma >= epsilon) + epsilon*(sigma < epsilon)

    representation = solution[:,:int(D/2)]
    key, subkey = jax.random.split(key)
    representation += sigma*jax.random.normal(key, representation.shape)

    return jnp.concatenate((representation, sigma), axis = 1)