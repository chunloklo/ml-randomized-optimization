""" Functions to implement the randomized optimization and search algorithms.
"""

# Author: Genevieve Hayes
# License: BSD 3 clause

import numpy as np
from .decay import GeomDecay
import time


def hill_climb(problem, max_iters=np.inf, restarts=0, init_state=None):
    """Use standard hill climbing to find the optimum for a given
    optimization problem.

    Parameters
    ----------
    problem: optimization object
        Object containing fitness function optimization problem to be solved.
        For example, :code:`DiscreteOpt()`, :code:`ContinuousOpt()` or
        :code:`TSPOpt()`.
    max_iters: int, default: np.inf
        Maximum number of iterations of the algorithm for each restart.
    restarts: int, default: 0
        Number of random restarts.
    init_state: array, default: None
        1-D Numpy array containing starting state for algorithm.
        If :code:`None`, then a random state is used.

    Returns
    -------
    best_state: array
        Numpy array containing state that optimizes the fitness function.
    best_fitness: float
        Value of fitness function at best state.

    References
    ----------
    Russell, S. and P. Norvig (2010). *Artificial Intelligence: A Modern
    Approach*, 3rd edition. Prentice Hall, New Jersey, USA.
    """
    if (not isinstance(max_iters, int) and max_iters != np.inf
            and not max_iters.is_integer()) or (max_iters < 0):
        raise Exception("""max_iters must be a positive integer.""")

    if (not isinstance(restarts, int) and not restarts.is_integer()) \
       or (restarts < 0):
        raise Exception("""restarts must be a positive integer.""")

    if init_state is not None and len(init_state) != problem.get_length():
        raise Exception("""init_state must have same length as problem.""")

    best_fitness = -1*np.inf
    best_state = None

    for _ in range(restarts + 1):
        # Initialize optimization problem
        if init_state is None:
            problem.reset()
        else:
            problem.set_state(init_state)

        iters = 0

        while iters < max_iters:
            iters += 1

            print("Iters: {}".format(iters))

            # Find neighbors and determine best neighbor
            problem.find_neighbors()
            print("finished finding neighbors")
            next_state = problem.best_neighbor()
            print("finished finding best")
            next_fitness = problem.eval_fitness(next_state)
            print("next step")

            # If best neighbor is an improvement, move to that state
            if next_fitness > problem.get_fitness():
                problem.set_state(next_state)

            else:
                break

        # Update best state and best fitness
        if problem.get_fitness() > best_fitness:
            best_fitness = problem.get_fitness()
            best_state = problem.get_state()

    best_fitness = problem.get_maximize()*best_fitness
    return best_state, best_fitness


def random_hill_climb(problem, max_attempts=10, max_iters=np.inf, restarts=0,
                      init_state=None, test=None, **kwargs):
    """Use randomized hill climbing to find the optimum for a given
    optimization problem.

    Parameters
    ----------
    problem: optimization object
        Object containing fitness function optimization problem to be solved.
        For example, :code:`DiscreteOpt()`, :code:`ContinuousOpt()` or
        :code:`TSPOpt()`.
    max_attempts: int, default: 10
        Maximum number of attempts to find a better neighbor at each step.
    max_iters: int, default: np.inf
        Maximum number of iterations of the algorithm.
    restarts: int, default: 0
        Number of random restarts.
    init_state: array, default: None
        1-D Numpy array containing starting state for algorithm.
        If :code:`None`, then a random state is used.

    Returns
    -------
    best_state: array
        Numpy array containing state that optimizes the fitness function.
    best_fitness: float
        Value of fitness function at best state.

    References
    ----------
    Brownlee, J (2011). *Clever Algorithms: Nature-Inspired Programming
    Recipes*. `<http://www.cleveralgorithms.com>`_.
    """
    start_time = time.time()

    scores = []
    valid_scores = []
    if (not isinstance(max_attempts, int) and not max_attempts.is_integer()) \
       or (max_attempts < 0):
        raise Exception("""max_attempts must be a positive integer.""")

    if (not isinstance(max_iters, int) and max_iters != np.inf
            and not max_iters.is_integer()) or (max_iters < 0):
        raise Exception("""max_iters must be a positive integer.""")

    if (not isinstance(restarts, int) and not restarts.is_integer()) \
       or (restarts < 0):
        raise Exception("""restarts must be a positive integer.""")

    if init_state is not None and len(init_state) != problem.get_length():
        raise Exception("""init_state must have same length as problem.""")

    best_fitness = -1*np.inf
    best_state = None

    for _ in range(restarts + 1):
        # print("restart!")
        # Initialize optimization problem and attempts counter
        print("restart!")
        if init_state is None:
            problem.reset()
        else:
            problem.set_state(init_state)

        # print(problem.eval_fitness(problem.random_neighbor()))

        attempts = 0
        iters = 0

        while (attempts < max_attempts) and (iters < max_iters):
            iters += 1

            # Find random neighbor and evaluate fitness
            next_state = problem.random_neighbor()
            next_fitness = problem.eval_fitness(next_state)
            scores.append(next_fitness)
            if (test is not None):
                valid_scores.append(test(next_state, **kwargs))
            # print(next_fitness)
            # print(attempts)



            # If best neighbor is an improvement,
            # move to that state and reset attempts counter

            if next_fitness > problem.get_fitness():
                print(next_fitness)
                problem.set_state(next_state)
                attempts = 0

            else:
                attempts += 1


        # Update best state and best fitness
        if problem.get_fitness() > best_fitness:
            best_fitness = problem.get_fitness()
            best_state = problem.get_state()

    best_fitness = problem.get_maximize()*best_fitness

    elapsed_time = time.time() - start_time
    print(time.strftime("%H:%M:%S", time.gmtime(elapsed_time)))
    return scores, valid_scores, best_state, best_fitness


def simulated_annealing(problem, schedule=GeomDecay(), max_attempts=10,
                        max_iters=np.inf, init_state=None, test=None, **kwargs):
    """Use simulated annealing to find the optimum for a given
    optimization problem.

    Parameters
    ----------
    problem: optimization object
        Object containing fitness function optimization problem to be solved.
        For example, :code:`DiscreteOpt()`, :code:`ContinuousOpt()` or
        :code:`TSPOpt()`.
    schedule: schedule object, default: :code:`mlrose.GeomDecay()`
        Schedule used to determine the value of the temperature parameter.
    max_attempts: int, default: 10
        Maximum number of attempts to find a better neighbor at each step.
    max_iters: int, default: np.inf
        Maximum number of iterations of the algorithm.
    init_state: array, default: None
        1-D Numpy array containing starting state for algorithm.
        If :code:`None`, then a random state is used.

    Returns
    -------
    best_state: array
        Numpy array containing state that optimizes the fitness function.
    best_fitness: float
        Value of fitness function at best state.

    References
    ----------
    Russell, S. and P. Norvig (2010). *Artificial Intelligence: A Modern
    Approach*, 3rd edition. Prentice Hall, New Jersey, USA.
    """

    start_time = time.time()

    scores = []
    valid_scores = []
    if (not isinstance(max_attempts, int) and not max_attempts.is_integer()) \
       or (max_attempts < 0):
        raise Exception("""max_attempts must be a positive integer.""")

    if (not isinstance(max_iters, int) and max_iters != np.inf
            and not max_iters.is_integer()) or (max_iters < 0):
        raise Exception("""max_iters must be a positive integer.""")

    if init_state is not None and len(init_state) != problem.get_length():
        raise Exception("""init_state must have same length as problem.""")

    # Initialize problem, time and attempts counter
    if init_state is None:
        problem.reset()
    else:
        problem.set_state(init_state)

    attempts = 0
    iters = 0

    while (attempts < max_attempts) and (iters < max_iters):
        temp = schedule.evaluate(iters)
        iters += 1
        # print(attempts)

        if temp == 0:
            break

        else:
            # Find random neighbor and evaluate fitness
            next_state = problem.random_neighbor()
            next_fitness = problem.eval_fitness(next_state)
            scores.append(next_fitness)
            if(test != None):
                valid_scores.append(test(next_state, **kwargs))


            # Calculate delta E and change prob
            delta_e = next_fitness - problem.get_fitness()
            prob = np.exp(delta_e/temp)

            # If best neighbor is an improvement or random value is less
            # than prob, move to that state and reset attempts counter
            print("Iterations: {} , {}".format(iters, next_fitness), end='\r')

            if (delta_e > 0) or (np.random.uniform() < prob):
                problem.set_state(next_state)
                attempts = 0
                # print(next_fitness)


            else:
                attempts += 1

    best_fitness = problem.get_maximize()*problem.get_fitness()
    best_state = problem.get_state()

    elapsed_time = time.time() - start_time
    print(time.strftime("%H:%M:%S", time.gmtime(elapsed_time)))

    return scores, valid_scores, best_state, best_fitness


def genetic_alg(problem, pop_size=200, mutation_prob=0.1, max_attempts=10,
                max_iters=np.inf, test=None, **kwargs):
    """Use a standard genetic algorithm to find the optimum for a given
    optimization problem.

    Parameters
    ----------
    problem: optimization object
        Object containing fitness function optimization problem to be solved.
        For example, :code:`DiscreteOpt()`, :code:`ContinuousOpt()` or
        :code:`TSPOpt()`.
    pop_size: int, default: 200
        Size of population to be used in genetic algorithm.
    mutation_prob: float, default: 0.1
        Probability of a mutation at each element of the state vector
        during reproduction, expressed as a value between 0 and 1.
    max_attempts: int, default: 10
        Maximum number of attempts to find a better state at each step.
    max_iters: int, default: np.inf
        Maximum number of iterations of the algorithm.

    Returns
    -------
    best_state: array
        Numpy array containing state that optimizes the fitness function.
    best_fitness: float
        Value of fitness function at best state.

    References
    ----------
    Russell, S. and P. Norvig (2010). *Artificial Intelligence: A Modern
    Approach*, 3rd edition. Prentice Hall, New Jersey, USA.
    """
    start_time = time.time()

    if pop_size < 0:
        raise Exception("""pop_size must be a positive integer.""")
    elif not isinstance(pop_size, int):
        if pop_size.is_integer():
            pop_size = int(pop_size)
        else:
            raise Exception("""pop_size must be a positive integer.""")

    if (mutation_prob < 0) or (mutation_prob > 1):
        raise Exception("""mutation_prob must be between 0 and 1.""")

    if (not isinstance(max_attempts, int) and not max_attempts.is_integer()) \
       or (max_attempts < 0):
        raise Exception("""max_attempts must be a positive integer.""")

    if (not isinstance(max_iters, int) and max_iters != np.inf
            and not max_iters.is_integer()) or (max_iters < 0):
        raise Exception("""max_iters must be a positive integer.""")

    # Initialize problem, population and attempts counter
    problem.reset()
    problem.random_pop(pop_size)
    attempts = 0
    iters = 0

    scores = []
    valid_scores = []

    while (attempts < max_attempts) and (iters < max_iters):
        iters += 1
        print("Iterations: {}".format(iters), end='\r')

        # Calculate breeding probabilities
        problem.eval_mate_probs()

        # Create next generation of population
        next_gen = []

        for _ in range(pop_size):
            # Select parents
            selected = np.random.choice(pop_size, size=2,
                                        p=problem.get_mate_probs())
            parent_1 = problem.get_population()[selected[0]]
            parent_2 = problem.get_population()[selected[1]]

            # Create offspring
            child = problem.reproduce(parent_1, parent_2, mutation_prob)
            next_gen.append(child)

        next_gen = np.array(next_gen)
        problem.set_population(next_gen)

        next_state = problem.best_child()
        next_fitness = problem.eval_fitness(next_state)

        # If best child is an improvement,
        # move to that state and reset attempts counter
        if next_fitness > problem.get_fitness():
            # print(next_fitness)
            problem.set_state(next_state)
            attempts = 0

        else:
            attempts += 1

        print(problem.get_maximize()*problem.get_fitness())
        scores.append(problem.get_maximize()*problem.get_fitness())
        if (test is not None):
            valid_scores.append(test(next_state, **kwargs))

    best_fitness = problem.get_maximize()*problem.get_fitness()
    best_state = problem.get_state()

    elapsed_time = time.time() - start_time
    print(time.strftime("%H:%M:%S", time.gmtime(elapsed_time)))
    return scores, valid_scores, best_state, best_fitness


def mimic(problem, pop_size=200, keep_pct=0.2, max_attempts=10,
          max_iters=np.inf):
    """Use MIMIC to find the optimum for a given optimization problem.

    Parameters
    ----------
    problem: optimization object
        Object containing fitness function optimization problem to be solved.
        For example, :code:`DiscreteOpt()` or :code:`TSPOpt()`.
    pop_size: int, default: 200
        Size of population to be used in algorithm.
    keep_pct: float, default: 0.2
        Proportion of samples to keep at each iteration of the algorithm,
        expressed as a value between 0 and 1.
    max_attempts: int, default: 10
        Maximum number of attempts to find a better neighbor at each step.
    max_iters: int, default: np.inf
        Maximum number of iterations of the algorithm.

    Returns
    -------
    best_state: array
        Numpy array containing state that optimizes the fitness function.
    best_fitness: float
        Value of fitness function at best state.

    References
    ----------
    De Bonet, J., C. Isbell, and P. Viola (1997). MIMIC: Finding Optima by
    Estimating Probability Densities. In *Advances in Neural Information
    Processing Systems* (NIPS) 9, pp. 424–430.

    Note
    ----
    MIMIC cannot be used for solving continuous-state optimization problems.
    """
    scores = []

    start_time = time.time()
    if problem.get_prob_type() == 'continuous':
        raise Exception("""problem type must be discrete or tsp.""")

    if pop_size < 0:
        raise Exception("""pop_size must be a positive integer.""")
    elif not isinstance(pop_size, int):
        if pop_size.is_integer():
            pop_size = int(pop_size)
        else:
            raise Exception("""pop_size must be a positive integer.""")

    if (keep_pct < 0) or (keep_pct > 1):
        raise Exception("""keep_pct must be between 0 and 1.""")

    if (not isinstance(max_attempts, int) and not max_attempts.is_integer()) \
       or (max_attempts < 0):
        raise Exception("""max_attempts must be a positive integer.""")

    if (not isinstance(max_iters, int) and max_iters != np.inf
            and not max_iters.is_integer()) or (max_iters < 0):
        raise Exception("""max_iters must be a positive integer.""")

    # Initialize problem, population and attempts counter
    problem.reset()
    problem.random_pop(pop_size)
    attempts = 0
    iters = 0

    while (attempts < max_attempts) and (iters < max_iters):
        iters += 1

        # Get top n percent of population
        problem.find_top_pct(keep_pct)

        # Update probability estimates
        problem.eval_node_probs()

        # Generate new sample
        new_sample = problem.sample_pop(pop_size)
        problem.set_population(new_sample)

        next_state = problem.best_child()

        next_fitness = problem.eval_fitness(next_state)
        print(problem.get_fitness())
        scores.append(problem.get_fitness())

        # If best child is an improvement,
        # move to that state and reset attempts counter
        if next_fitness > problem.get_fitness():
            problem.set_state(next_state)
            attempts = 0

        else:
            attempts += 1

    best_fitness = problem.get_maximize()*problem.get_fitness()
    best_state = problem.get_state().astype(int)

    elapsed_time = time.time() - start_time
    print(time.strftime("%H:%M:%S", time.gmtime(elapsed_time)))

    return scores, best_state, best_fitness
