# T-Order
# A Genetic Algorithm Implementation of the Traveling Salesman Problem.
# Alessandro Allegranzi
# 6/10/2024

# In this code, distance_matrix is a 2D numpy array where the element at the i-th row and j-th column 
# represents the distance between the i-th city and the j-th city. You would need to define this matrix 
# based on your specific problem instance. You can call the genetic_algorithm function with your list 
# of cities, the size of the population, the number of elite individuals, the mutation rate, and the 
# number of generations to run the algorithm. The function will return the best route found.

import numpy as np
import random

def create_route(city_list):
    route = random.sample(city_list, len(city_list))
    return route

def initial_population(city_list, pop_size):
    population = []
    for i in range(0, pop_size):
        population.append(create_route(city_list))
    return population

def compute_fitness(route):
    return 1 / np.sum([distance_matrix[route[i]][route[i + 1]] for i in range(len(route) - 1)])

def rank_routes(population):
    fitness_results = {i: compute_fitness(population[i]) for i in range(len(population))}
    return sorted(fitness_results.items(), key = lambda x : x[1], reverse = True)

def selection(pop_ranked, elite_size):
    selection_results = [pop_ranked[i][0] for i in range(elite_size)]
    for i in range(len(pop_ranked) - elite_size):
        pick = int(np.random.choice(len(pop_ranked), p = [j[1] for j in pop_ranked]))
        selection_results.append(pop_ranked[pick][0])
    return selection_results

def mating_pool(population, selection_results):
    matingpool = [population[i] for i in selection_results]
    return matingpool

def breed(parent1, parent2):
    child = []
    childP1 = []
    geneA = int(random.random() * len(parent1))
    geneB = int(random.random() * len(parent1))
    startGene = min(geneA, geneB)
    endGene = max(geneA, geneB)
    for i in range(startGene, endGene):
        childP1.append(parent1[i])
    childP2 = [item for item in parent2 if item not in childP1]
    child = childP1 + childP2
    return child

def breed_population(matingpool, elite_size):
    children = []
    length = len(matingpool) - elite_size
    pool = random.sample(matingpool, len(matingpool))
    for i in range(0, elite_size):
        children.append(matingpool[i])
    for i in range(0, length):
        child = breed(pool[i], pool[len(matingpool)-i-1])
        children.append(child)
    return children

def mutate(individual, mutation_rate):
    for swapped in range(len(individual)):
        if(random.random() < mutation_rate):
            swap_with = int(random.random() * len(individual))
            city1 = individual[swapped]
            city2 = individual[swap_with]
            individual[swapped] = city2
            individual[swap_with] = city1
    return individual

def mutate_population(population, mutation_rate):
    mutated_pop = []
    for ind in range(0, len(population)):
        mutated_ind = mutate(population[ind], mutation_rate)
        mutated_pop.append(mutated_ind)
    return mutated_pop

def next_generation(current_gen, elite_size, mutation_rate):
    pop_ranked = rank_routes(current_gen)
    selection_results = selection(pop_ranked, elite_size)
    matingpool = mating_pool(current_gen, selection_results)
    children = breed_population(matingpool, elite_size)
    next_gen = mutate_population(children, mutation_rate)
    return next_gen

def genetic_algorithm(city_list, pop_size, elite_size, mutation_rate, generations):
    pop = initial_population(city_list, pop_size)
    progress = []
    for i in range(0, generations):
        pop = next_generation(pop, elite_size, mutation_rate)
        best_route_index = rank_routes(pop)[0][0]
        progress.append(1 / rank_routes(pop)[0][1])
    best_route_index = rank_routes(pop)[0][0]
    best_route = pop[best_route_index]
    return best_route