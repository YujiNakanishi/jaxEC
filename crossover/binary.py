import jax
from jax import jit, vmap
import jax.numpy as jnp
import sys
from jaxEC import util



def one_pointX(key, solution):
    """
    one pointx crossover
    """
    father, mother = util.split(key, solution, 2) #(N/2, D)
    key, subkey = jax.random.split(key)
    partition = jax.random.choice(key, solution.shape[1], (len(father),))

    offspring1 = []; offspring2 = []
    for f, m, p in zip(father, mother, partition):
        f_head = f[:p]
        f_tail = f[p:]
        m_head = m[:p]
        m_tail = m[p:]

        offspring1.append(jnp.concatenate((f_head, m_tail)))
        offspring2.append(jnp.concatenate((m_head, f_tail)))
    
    offspring = jnp.stack(offspring1 + offspring2)

    return offspring



@jit
def uniform(key, solution, pc = 0.5):
    """
    uniform crossover

    input:
        pc -> <float> cross over ratio
    """
    father, mother = util.split(key, solution, 2) #(N/2, D)
    key, subkey = jax.random.split(key)
    mask = (jax.random.uniform(key, father.shape) < pc).astype(int)

    offs1 = father*mask + mother*(mask == 0)
    offs2 = mother*mask + father*(mask == 0)

    offspring = jnp.concatenate((offs1, offs2))

    return offspring