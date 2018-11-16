import random
from operator import attrgetter
import geopy.distance  # We need this module for finding distance between two points
from copy import deepcopy
AIRPORTS_FILENAME   = 'airports.txt'
AIRCRAFTS_FILENAME  = 'aircrafts.txt'
PASSENGERS_FILENAME = 'passengers.txt'
NUMBER_OF_GENERATIONS = 1000
POPULATION_SIZE = 100
ELITISM_RATE = 10 # 10% elitism
budget = 0
airports = []
aircrafts = []
passengers = {}


class Individual:
    def __init__(self, chromosome):
        self._chromosome = chromosome
        self._fitness = self.get_fitness()

    def crossover(self, other):
        child = []

        
        crossover_point = random.randint(0, len(self._chromosome))
        # print(f"Crossover point: {crossover_point}\t")

        for gene1, gene2 in zip(self._chromosome, other._chromosome):
            prob = random.random()

            if prob < 0.45:
                child.append(gene1)
            elif prob < 0.9:
                child.append(gene2)
            else:
                child.append(Gene.create_random())
        # tmp = a2[:x].copy()
        # a2[:x], a1[:x]  = a1[:x], tmp
        # print(self._chromosome)
        # print()
        # print(other._chromosome)
        # child.extend(self._chromosome[:crossover_point].copy())
        # child.extend(other._chromosome[crossover_point:].copy())
        ch = Individual(child)
        # print(f"{self._fitness} , {other._fitness} = {ch._fitness}")
        return ch
        # return Individual(child)


    def get_fitness(self):
        fitness = 0
        for gene in self._chromosome:
            fitness += gene._income
        return fitness
        
    def __repr__(self):
        return str(self._fitness)

    @staticmethod
    def create(num_of_genes):
        chromosome = [Gene.create_random() for _ in range(num_of_genes)] 
        # chromosome = []
        # for _ in range(num_of_genes):
            

            # aircraft = Aircraft.random_unused_aircraft() # finds a random aircraft which is unused and we can buy with our budget
            # if aircraft is None: # if we don't have enough money to buy
            #     chromosome.append(Gene())
            #     continue
            # source = Airport.random_airport()
            # destination = Airport.random_airport()
            # while destination is source:
            #     destination = Airport.random_airport()
            # add_passengers(aircraft, source._name, destination._name)
            # chromosome.append(Gene(s=source, d=destination, aircraft=aircraft))
        return Individual(chromosome)

def add_passengers(aircraft, source, destination):
    global passengers
    if aircraft.capacity() < passengers[source][destination]: # aircraft will be filled
        passengers[source][destination] -= aircraft.capacity()
        aircraft._passengers = aircraft.capacity()
    else: # source airport will be empty
        aircraft._passengers = passengers[source][destination]
        passengers[source][destination] = 0



class Gene:
    def __init__(self, s=None, d=None, aircraft=None):
        self._from = s
        self._to = d
        self._aircraft = aircraft
        self._income = self.cal_income() if not self._aircraft is None else 0

    def get_distance(self): # s, d are airports
        coords_1 = (float(self._from._lat), float(self._from._lon))
        coords_2 = (float(self._to._lat), float(self._to._lon))
        return geopy.distance.vincenty(coords_1, coords_2).km

    def cal_income(self):
        income = 0
        # income = -(self._aircraft._price)  # first, the loss!
        distance = self.get_distance()
        income += distance * self._aircraft._psp / 100
        income -= distance * self._aircraft._psl / 100
        return income

    @staticmethod
    def create_random():
        aircraft = Aircraft.random_unused_aircraft() # finds a random aircraft which is unused and we can buy with our budget
        if aircraft is None: # if we don't have enough money to buy
            return Gene()
            # continue
        source = Airport.random_airport()
        destination = Airport.random_airport()
        while destination is source:
            destination = Airport.random_airport()
        add_passengers(aircraft, source._name, destination._name)
        return Gene(s=source, d=destination, aircraft=aircraft)


class Population:
    pass

class Airport:
    def __init__(self, name, lat, lon):
        self._name = name
        self._lat = lat
        self._lon = lon

    @staticmethod
    def random_airport():
        return random.choice(airports)

    def __repr__(self):
        return self._name

class Aircraft:
    def __init__(self, name, cap, price, psp, psl):
        self._name = name
        self._capacity = cap
        self._passengers = 0
        self._price = price
        self._psp = psp
        self._psl = psl
        self._used = False

    def __repr__(self):
        return self._name

    def use(self):
        global budget
        self._used = True
        budget -= self._price
    
    def is_used(self):
        return self._used
    
    def capacity(self):
        return self._capacity
    
    def reset(self):
        self._passengers = 0
        self._used = False

    @staticmethod
    def random_unused_aircraft():
        available_aircrafts = [aircraft for aircraft in aircrafts if (not aircraft.is_used() and budget >= aircraft._price)]
        if len(available_aircrafts) == 0:
            return None
        result = random.choice(available_aircrafts)
        result.use()
        return result

def set_passengers(filename):
    global passengers
    with open(filename) as f:
        for line in f:
            ap1_name, ap2_name, p12, p21 = line.split()
            if not ap1_name in passengers.keys():
                passengers[ap1_name] = {}
            if not ap2_name in passengers.keys():
                passengers[ap2_name] = {}
            passengers[ap1_name][ap2_name] = int(p12)
            passengers[ap2_name][ap1_name] = int(p21)
            
def set_airports(filename):
    result = []
    with open(filename) as f:
        for line in f:
            name, lat, lon = line.split()
            result.append(Airport(name, lat, lon))
    return result
            
def set_aircrafts(filename):
    result = []
    with open(filename) as f:
        for line in f:
            name, cap, price, psp, psl, max_num = line.split()
            for _ in range(int(max_num)):
                result.append(Aircraft(name, int(cap), int(price), int(psp), int(psl)))
    return result

def reset_aircrafts():
    for i in range(len(aircrafts)):
        aircrafts[i].reset()

def main():
    global budget, airports, aircrafts, passengers
    # budget = int(input())
    budget = input_budget = 10000
    # budget = 1000
    airports = set_airports(AIRPORTS_FILENAME)
    aircrafts = set_aircrafts(AIRCRAFTS_FILENAME)
    set_passengers(PASSENGERS_FILENAME)
    num_of_genes = len(aircrafts)
    population = []
    for _ in range(POPULATION_SIZE): # create initial population
        budget = input_budget
        reset_aircrafts()
        individual = Individual.create(num_of_genes)
        # print(individual)
        population.append(individual)

    
    for i in range(100): # number of generations

        population = sorted(population, key=lambda x : x._fitness, reverse=True)
        # print()
        print(f"Generation {i}\t Fitness={population[0]._fitness}")
        # for individual in population:
            # print(f'{individual._fitness}')
        new_generation = []

        s = int(ELITISM_RATE*POPULATION_SIZE/100)
        new_generation.extend(population[:s])

        s = int((100-ELITISM_RATE)*POPULATION_SIZE/100)
        for _ in range(s):
            par1 = random.choice(population[:50])
            par2 = random.choice(population[:50])
            child = par1.crossover(par2)
            # print(f"Crossovered. Parents' fitness: {par1._fitness}, {par2._fitness}\t Child's fitness: {child._fitness}")
            new_generation.append(child)

        population = new_generation


if __name__ == '__main__':
    main()