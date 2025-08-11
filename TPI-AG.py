import pyedflib
import random
import numpy as np
import random
import bisect


def seizure_detection(Flo,Fhi,Wbg,Wfg,Te,d,Lmin,Tc,file): # Detect seizures with the given parameters
    seizures = []
    file = pyedflib.EdfReader(f"{file}.edf") # Read the edf the file
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


def aggregation(seizures,Tc,n_channels): # Detect seizures across the differents channels
    evs = []
    channels = set()
    detected = []
    for i in range(n_channels):
        for ran in seizures[i]:
            l = ran[0]
            r = ran[1]
            evs.append([l,i,1,r]) #1 -> Add a range
            evs.append([r+1,i,0,l]) #0 -> Delete a range
    evs = sorted(evs)
    for i in range(len(evs)): # Sweep line to check inteserctions
        if evs[i][2] == 0:
            [r,idx,op,l] = evs[i]
            if [idx,l,r-1] in channels: channels.remove([idx,l,r-1])
        else:
            [l,idx,op,r] = evs[i]
            channels.add([idx,l,r])
        if len(channels) >= Tc:
            lefts = []
            rights = []
            for [idx,l,r] in channels:
                lefts.append(l)
                rights.append(r)
            lefts = sorted(lefts)
            rights = sorted(rights)
            ld = 0
            rd = 0
            if (len(lefts)%2 == 0):
                ld = (lefts[len(lefts)/2]+lefts[(len(lefts)+1)/2])/2 # Median of a set with even length
            else:
                ld = lefts[(len(lefts)+1)/2] # Median of a set with odd length
            if (len(rights)%2 == 0):
                rd = (rights[len(rights)/2]+rights[(len(rights)+1)/2])/2 # Median of a set with even length
            else:
                rd = rights[(len(rights)+1)/2] # Median of a set with odd length
            detected.append([ld,rd])
            channels.clear()
    return sorted(detected)


def group_seizure(seizure_points,d,Lmin): # Group points that belong to the same seizure
    groups = []
    l = -1
    i = 0
    while (i < len(seizure_points)):
        l = seizure_points[i]
        while(i+1 < len(seizure_points) and seizure_points[i+1]-l <= d): # While the distance between the points is <= d
            i = i+1
        r = seizure_points[i]
        i = i+1
        groups.append([l,r]) # Add new seizure
    filtered = []
    for i in groups:
        l = groups[i][0]
        r = groups[i][1]
        if (r-l >= Lmin): # Filter out seizures shorter than Lmin
            filtered.append([l,r])
    return filtered


def compute_points(energies,Te): # Detect abnormal points
    ordered = sorted(energies)
    seizure_points = []
    percentile = np.percentile(ordered,Te)
    for i in range(0,len(energies)):
        if (energies[i] > percentile):
            seizure_points.append(i)
    return seizure_points


def compute_energies(signals,backgrounds,foregrounds): # Calculate energy ratio for every foreground and background window
    energies = []
    bgs = []
    idx = 0
    for bg in backgrounds: # Calculate energy of background windows
        Lbg = bg[0] # Left bound of the background window
        Rbg = bg[1] # Right bound of the background window
        Ebg = 0
        for i in range(Lbg,Rbg):
            Ebg = Ebg + (signals[i]*signals[i])
        Ebg = Ebg/(Rbg-Lbg) # Energy of the background window
        bgs.append(Ebg)
    for fg in foregrounds:
        Lfg = fg[0] # Left bound of the foreground window
        Rfg = fg[1] # Right bound of the foreground window
        Lbg = backgrounds[idx][0] # Left bound of the background window
        Rbg = backgrounds[idx][1] # Right bound of the background window
        Efg = 0
        for i in range(Lfg,Rfg):
            Efg = Efg + (signals[i]*signals[i])
        Efg = Efg/(Rfg-Lfg) # Energy of the foreground window
        if (Lfg > Rbg): # If the foreground does not intersect the background window
            idx = idx+1
            Lbg = backgrounds[idx][0] # Update left bound
            Rbg = backgrounds[idx][1] # Update right bound
        if (Lfg >= Lbg and Rfg <= Rbg): # If the foreground is fully contained in the background
            energies.append(Efg/bgs[idx])  # Energy ratio
        if (Lfg <= Rbg and Rfg > Rbg): # If the foreground is parcially contained in the background
            conA = (Rbg-Lfg)/(Rfg-Lfg) # Porcentage contained in the first background window
            idx = idx+1
            Lbg = backgrounds[idx][0] # Update left bound
            Rbg = backgrounds[idx][1] # Update right bound
            conB = (Rfg-Lbg)/(Rfg-Lfg) # Porcentage contained in the second background
            energies.append(Efg/((bgs[idx-1]*conA)+(bgs[idx]*conB))) # Energy ratio
    return energies

        
def segment(signals,longi): # Divide the signals in segments
    segments = []
    tam = len(signals)
    l = 0
    while(l < tam):
        if (l+longi < tam): # If is possible to construct a segment of lenght tam
            segments.append([l,l+longi])
            l = l+longi
        else: # Is is not possible to construct a segment of lenght tam
            segments.append([l,tam-l]) #
            l = tam
    return segment


def filterOut(signals,Flo,Fhi): # Filter out signal below Flo and above Fhi
    filtered = []
    for i in signals:
        if (i>Flo and i<Fhi):
            filtered.append(i)
    return filtered


def seizures_overlap(seizure1, seizure2): 
    start1, end1 = seizure1 
    start2, end2 = seizure2 
    return max(start1, start2) < min(end1, end2)

def compute_fitness(chromosome):
    [Flo,Fhi,Wfg,Wbg,Te,d,Lmin,Td] = transform_to_params(chromosome) # Get parameters from the chromosome
    detected_seizures = seizure_detection(Flo,Fhi,Wbg,Wfg,Te,d,Lmin,Td)
    samples = ["chb01","chb05","chb12","chb13"]
    tams = [42,39,42,62]
    real_seizures = []
    real_seizures.append([[[0,0]],[[0,0]],[[2996,3036]], [[1467,1494]]], [[0,0]],[[0,0]],[[0,0]],[[0,0]],[[0,0]],[[0,0]],
                         [[0,0]],[[0,0]],[[0,0]],[[0,0]],[[1732,1772]],[[1015,1066]],[[0,0]],[[1720,1810]],[[0,0]],[[0,0]],
                         [[327,420]],[[0,0]],[[0,0]],[[0,0]],[[0,0]],[[1862,1963]],[[0,0]],[[0,0]],[[0,0]],[[0,0]],
                         [[0,0]],[[0,0]],[[0,0]],[[0,0]],[[0,0]],[[0,0]],[[0,0]],[[0,0]],[[0,0]],[[0,0]],[[0,0]],[[0,0]]) #Chb01
    # Falta hacer lo mismo para los otros pacientes chequeando su summary.txt
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


def transform_to_params(chromosome):
    Flo = 0, Fhi = 256, Wfg=0 ,Wbg=0,Te=0,d=0,Lmin=0,Td=0
    x = 0
    for i in range(0,6): # 6 bits for Flo
        Flo = Flo + (chromosome[i]<<i)
    for i in range(0,6): # 6 bits for Fhi
        Fhi = Fhi - (chromosome[6+i]<<i)
    for i in range(0,4): # 4 bits for Wfg
        x = x + (chromosome[12+i]<<i)
    Wfg = 0.5 + (x/2)
    x = 0
    for i in range(0,5): # 5 bits for Wbg
        x = x + (chromosome[16+i]<<i)
    Wbg = 60*(1+x)
    x = 0
    for i in range(0,6): # 6 bits for Te
        x = x + (chromosome[21+i]<<i)
    Te = 100 - (x/10)
    x = 0
    for i in range(0,5): # 5 bits for d
        x = x + (chromosome[27+i]<<i)
    d = 1 + x
    x = 0
    for i in range(0,5): # 5 bits for Lmin
        x = x + (chromosome[32+i]<<i)
    Lmin = 1+x
    x = 0
    for i in range(0,4): # 4 bits for Td
        x = x + (chromosome[37+i]<<i)
    Td = (1+x)/16
    return [Flo,Fhi,Wfg,Wbg,Te,d,Lmin,Td]



# Generate a chromosome with the given number of bits
def generate_chromosome(bits):
    chromosome = [random.randint(0, 1) for _ in range(bits)]
    chromosome = ''.join(map(str, chromosome))
    return chromosome

# Generate an initial population of 'num_chromosomes' individuals
def generate_random_population(num_chromosomes, bits):
    return [generate_chromosome(bits) for _ in range(num_chromosomes)]


def selectedByTournament(population, fit,tam_tournament):    
    selected = []
    tam = len(fit)
    for _ in range(tam):
        to_battle = []
        best = np.inf
        idBest = 99999
        for i in range(tam_tournament):
            to_battle.append(random.randint(0,tam-1))
        for i in to_battle:
            if (fit[i] < best or (fit[i]==best and i<idBest)): 
                 best = fit[i]
                 idBest = i
        selected.append(population[idBest]) 
    return selected 


def reproduce(selected,prob_cross): 
    descent = []
    for i in range(0, len(selected) - 1,2):
        prob = random.random()
        if (prob <= prob_cross): 
            cut = random.randint(0,40) 
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
            lim_inf = random.randint(0,40) 
            lim_sup = random.randint(0,40) 

            if (lim_inf > lim_sup): lim_inf, lim_sup = lim_sup, lim_inf  
            
            segmento_inv = child[lim_inf:lim_sup+1][::-1] 
            new_child = child[:lim_inf] + segmento_inv + child[lim_sup+1:] 
            
            new_childs.append(new_child) 
        else: new_childs.append(child) 
    return new_childs 


def getBestIndividual(popu):
    mini = np.inf
    idMini = 0
    for i in range(len(popu)):
        fit = compute_fitness(popu[i]) 
        if (fit < mini):
            mini = fit
            idMini = i
    return [popu[idMini],mini]


def genetic_algorithm(size_population, tam_tournament, mutation_rate,cross_rate): 
    population = generate_random_population(size_population,41)
    for i in range(10):
        selected = []
        fitnesses = []   
        for i in range(len(population)): fitnesses.append(compute_fitness(population[i]))
        while len(selected) < size_population:
            parents = selectedByTournament(population, fitnesses,tam_tournament)
            childs = reproduce(parents, cross_rate)
            selected = mutation(childs, mutation_rate)
        population = selected
    return getBestIndividual(population)


if __name__ == "__main__":
    print(genetic_algorithm(30,5,0.01,0.3))



