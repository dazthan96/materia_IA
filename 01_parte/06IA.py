import array
import random
import numpy as np
from deap import algorithms, base, creator, tools
#minimizacion menor cantidad de ceros -1.0
#maximizacion mayor cantidad de unos 1.0
creator.create("FitnessMax", base.Fitness,weights = (1.0,))
creator.create("Individual", array.array, typecode = "b", fitness = creator.FitnessMax)
toolbox = base.Toolbox()

toolbox.register("attr_bool", random.randint, 0,1)
toolbox.register("individual", tools.initRepeat, creator.Individual, toolbox.attr_bool,100) #10
toolbox.register("population", tools.initRepeat, list, toolbox.individual)

def f_objetivo(individual):
    lista = individual
    lista.reverse()
    k=0
    numero = 0
    for i in lista:
        exp = 2 ** k
        numero = numero + i*exp
        k = k + 1
    resultado = (numero**2) + 1
    #print(individual)
    return (resultado),

toolbox.register("evaluate", f_objetivo)
toolbox.register("mate", tools.cxTwoPoint)
toolbox.register("mutate", tools.mutFlipBit, indpb=0.05)
toolbox.register("select", tools.selTournament, tournsize=3)

def main():
    random.seed(64)
    pop = toolbox.population(n=300) #20
    hof = tools.HallOfFame(1)
    stats = tools.Statistics(lambda ind:ind.fitness.values)
    
    stats.register("avg", np.mean)
    stats.register("std", np.std)
    stats.register("min", np.min)
    stats.register("max", np.max)
    pop, log = algorithms.eaSimple(population=pop,toolbox=toolbox, cxpb=0.5, mutpb=0.2, ngen=40, stats=stats, halloffame=hof, verbose=True)
    return pop, log, hof

if __name__=="__main__":
    pop, log, hof = main()

lista = [1,0,1,0,1,1,1,0,0,1]
lista.reverse()
numero,k = 0,0
for i in lista:
    exp = 2 ** k
    numero = numero + i*exp
    k = k + 1
print(numero)