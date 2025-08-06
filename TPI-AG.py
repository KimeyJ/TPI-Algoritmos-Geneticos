import pyedflib
import random
import numpy as np
import random


def seizure_detection(Flo,Fhi,Wbg,Wfg,Te,d,Lmin,Tc):
    seizures = []
    file = pyedflib.EdfReader("chb01_03.edf")
    n_channels = file.signals_in_file
    for ch in range(n_channels):
        signals = file.readSignal(ch)
        signals = filterOut(signals,Flo,Fhi)
        backgrounds = segment(signals,Wbg)
        foregrounds = segment(signals,Wfg)
        energies = compute_energies(signals,backgrounds,foregrounds)
        seizure_points = compute_points(energies,Te)
        grouped_point = group_seizure(seizure_points,d,Lmin)
        seizures.append(grouped_point)
    seizures = aggregation(seizures,Tc)
    return seizures


def aggregation(seizures,Tc):
    print("xd")

def group_seizure(seizure_points,d,Lmin):
    groups = []
    l = -1
    i = 0
    while (i < len(seizure_points)):
        l = seizure_points[i]
        while(i+1 < len(seizure_points) and seizure_points[i+1]-l <= d):
            i = i+1
        r = seizure_points[i]
        i = i+1
        groups.append([l,r])
    filtered = []
    for i in groups:
        l = groups[i][0]
        r = groups[i][1]
        if (r-l >= Lmin):
            filtered.append([l,r])
    return filtered


def compute_points(energies,Te):
    ordered = sorted(energies)
    seizure_points = []
    percentile = np.percentile(ordered,Te)
    for i in range(0,len(energies)):
        if (energies[i] > percentile):
            seizure_points.append(i)
    return seizure_points


def compute_energies(signals,backgrounds,foregrounds):
    energies = []
    idx = 0
    for bg in backgrounds:
        Lbg = bg[0]
        Rbg = bg[1]
        Ebg = 0
        for i in range(Lbg,Rbg):
            Ebg = Ebg + (signals[i]*signals[i])
        Ebg = Ebg/(Rbg-Lbg)
        Lfg = foregrounds[idx][0]
        Rfg = foregrounds[idx][1]
        while(Lfg >= Lbg and Lfg < Rbg): # Inclusive-Exclusive -> [,)
            Efg = 0
            for i in range(Lfg,Rfg):
                Efg = Efg + (signals[i]*signals[i])
            Efg = Efg/(Rfg-Lfg)
            energies.append(Efg/Ebg)
            idx = idx + 1
            if (idx > len(foregrounds)):
                break
    return energies

        
def segment(signals,longi):
    segments = []
    tam = len(signals)
    l = 0
    while(l < tam):
        if (l+longi < tam):
            segments.append([l,l+longi])
            l = l+longi
        else:
            segments.append([l,tam-l])
            l = tam
    return segment


def filterOut(signals,Flo,Fhi):
    filtered = []
    for i in signals:
        if (i>Flo and i<Fhi):
            filtered.append(i)
    return filtered


def seizures_overlap(seizure1, seizure2): 
    start1, end1 = seizure1 
    start2, end2 = seizure2 
    return max(start1, start2) < min(end1, end2)

def compute_fitness(idx,real_seizures, detected_seizures): 
    matched_real = set()
    matched_detected = set()
    e = 0 #error de los limites 
    for i, r_seizure in enumerate(real_seizures):
        for j, d_seizure in enumerate(detected_seizures):
            if i not in matched_real and j not in matched_detected:
                ro, re = r_seizure
                do, de = d_seizure
                e += abs(ro-do) + abs(re-de)
                matched_real.add(i)
                matched_detected.add(j)
                break 
    Fp = len(detected_seizures) - len(matched_detected)
    Fn = len(real_seizures) - len(matched_real) 

    fitness = (10**6)*Fn + (10**4)*Fp + e  
    return fitness 


# Generate a chromosome with the given number of bits
def generate_chromosome(bits):
    chromosome = [random.randint(0, 1) for _ in range(bits)]
    chromosome = ''.join(map(str, chromosome))
    return chromosome

# Generate an initial population of 'num_chromosomes' individuals
def generate_random_population(num_chromosomes, bits):
    return [generate_chromosome(bits) for _ in range(num_chromosomes)]


def selectedByTournament(population, fit):    
    selected = []
    tam = len(fit)
    for _ in range(tam):
        idA = random.randint(0,tam-1)
        idB = random.randint(0,tam-1)
        if (fit[idA] > fit[idB]): selected.append(population[idA]) 
        else: selected.append(population[idB]) 
    return selected 


def reproduce(selected,prob_cross): 
    descent = []
    for i in range(0, len(selected) - 1,2):
        prob = random.random()
        if (prob <= prob_cross): 
            cut = random.randint(0,29) 
            father1 = selected[i]
            father2 = selected[i+1]
            
            child1 = father1[:cut] + father2[cut:] 
            child2 = father2[:cut] + father1[cut:] 
            
            descent.append(child1) 
            descent.append(child2) 
        else:
            descent.append(selected[i]) 
            descent.append(selected[i+1])
    return descent


def mutation(childs,prob_mut):
    new_childs = []
    for child in childs:
        new_child = ""
        prob = random.random()
        if (prob <= prob_mut): 
            lim_inf = random.randint(0,29) 
            lim_sup = random.randint(0,29) 

            if (lim_inf > lim_sup): lim_inf, lim_sup = lim_sup, lim_inf  
            
            segmento_inv = child[lim_inf:lim_sup+1][::-1] 
            new_child = child[:lim_inf] + segmento_inv + child[lim_sup+1:] 
            
            new_childs.append(new_child) 
        else: new_childs.append(child) 
    return new_childs 

def getBestIndividual(popu):
    mini = 1000000000000
    idMini = 0
    for i in range(len(popu)):
        if (compute_fitness(popu[i]))


def genetic_algorithm(size_population, tam_tournament, mutation_rate): 
    population = generate_random_population(size_population)
    real_seizure = []
    for i in range(10):
        selected = []
        fitnesses = []
        detected_per_chromo = []
        for i in range(len(population)): 
            [Flo,Fhi,Wbg,Wfg,Te,d,Lmin,Td] = transform_to_params() #Decodificar los parametros del cromosoma
            detected_per_chromo.append() #Calcular los ataques con los parametros dados
        for i in range(len(population)): fitnesses.append(compute_fitness(i,real_seizures, detected_seizures))
        while len(selected) < size_population:
            parents = selectedByTournament(population, fitnesses)
            childs = reproduce(parents, cross_rate)
            selected = mutation(childs, mutation_rate)
        population = selected
    return getBestIndividual(population)


if __name__ == "__main__":
    print("xd")



