import jax.numpy as jnp
from jax import jit
import jax

@jit
def gauss(key, solution, sigma):
    """
    output mutated solution by gaussian

    input:
        key -> PRNG key
        solution -> <jnp:num:(N, D)>
        sigma -> <float|jnp:num:(D, )> standard deviation
        shape -> <tuple> (N, D)
    
    output:
        <jnp:float:shape> mutated solution
    """
    return solution + sigma*jax.random.normal(key, solution.shape)

@jit
def selfAdaptive(key, solution, epsilon, generation):
    """
    output mutated solution by self adaptive method

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
    output mutated solution by self adaptive vector method

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


def Rechenberg(key, solution, fitness_function, sigma, c = 0.05, ps = 0.2):
    """
    output mutated solution and updated sigma by Rechenberg

    input:
        key -> PRNG key
        solution -> <jnp:num:(N, D)>
        fitness_function -> <vmap function> function for calculating fitness
            input:<jnp:num:(D, )>
            output:<scalar>
        sigma -> same with gauss
        c, ps -> <float> hyper param (default value is what is recommended in the book)
    
    output:
        mutation -> <jnp:float:(N, D)> mutated solution
        sigma -> updated sigma
    """
    solution_fitness = fitness_function(solution)

    mutation = gauss(key, solution, sigma)
    mutation_fitness = fitness_function(mutation)
    improved_probability = jnp.sum(mutation_fitness < solution_fitness)/len(solution_fitness)

    if improved_probability > ps:
        sigma *=c
    else:
        sigma /=c
    
    return mutation, sigma