import mne
import random
import numpy as np
import random
from concurrent.futures import ProcessPoolExecutor

def seizure_detection(Flo,Wbg,Wfg,Te,d,Lmin,Tc,name): # Detect seizures with the given parameters
    d = d*256
    Lmin = Lmin*256
    Wbg = Wbg*256
    Wfg = Wfg*256
    seizures = []
    print(name)
    file = mne.io.read_raw_edf(f"{name}.edf",preload=True) # Read the edf the file
    n_channels = len(file.ch_names)
    for ch in range(n_channels):
        name = file.ch_names[ch]
        signals = file[name]
        signals = filterOut(signals,Flo,file.info['sfreq'])
        backgrounds = segment(signals[0],Wbg)
        foregrounds = segment(signals[0],Wfg)
        energies = compute_energies(signals[0],backgrounds,foregrounds)
        seizure_points = compute_points(energies,Te)
        grouped_point = group_seizure(seizure_points,d,Lmin,foregrounds)
        seizures.append(grouped_point)
    seizures = aggregation(seizures,Tc,n_channels)
    return seizures



def aggregation(seizures,Tc,n_channels): # Detect seizures across the differents channels
    evs = []
    channels = set()
    detected = []
    for i in range(n_channels):
        for ran in seizures[i]:
            l = ran[0]
            r = ran[1]
            evs.append((l,i,1,r)) #1 -> Add a range
            evs.append((r+1,i,0,l)) #0 -> Delete a range
    evs = sorted(evs)
    for i in range(len(evs)): # Sweep line to check inteserctions
        if evs[i][2] == 0:
            (r,idx,op,l) = evs[i]
            if (idx,l,r-1) in channels: channels.remove((idx,l,r-1))
        else:
            (l,idx,op,r) = evs[i]
            channels.add((idx,l,r))
        if (len(channels)/n_channels) >= Tc:
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
                ld = (lefts[int(len(lefts)/2)]+lefts[int((len(lefts)+1)/2)])/2 # Median of a set with even length
            else:
                ld = lefts[int((len(lefts))/2)] # Median of a set with odd length
            if (len(rights)%2 == 0):
                rd = (rights[int(len(rights)/2)]+rights[int((len(rights)+1)/2)])/2 # Median of a set with even length
            else:
                rd = rights[int((len(rights))/2)] # Median of a set with odd length
            detected.append((ld,rd))
            #channels.clear()
    return sorted(detected)


def group_seizure(seizure_points,d,Lmin,foregrounds): # Group points that belong to the same seizure
    groups = []
    l = -1
    i = 0
    while (i < len(seizure_points)):
        idx = seizure_points[i]
        l = foregrounds[idx][0]
        while(i+1 < len(seizure_points) and abs(foregrounds[seizure_points[i+1]][0]-foregrounds[idx][1]) <= d): # While the distance between the points is <= d
            i = i+1
        idx = seizure_points[i]
        r = foregrounds[idx][1]
        i = i+1
        groups.append([l,r]) # Add new seizure
    filtered = []
    for i in range(len(groups)):
        l = groups[i][0]
        r = groups[i][1]
        if (r-l >= Lmin): # Filter out seizures shorter than Lmin
            filtered.append([l,r])
    print(groups)
    print(filtered)
    print(Lmin)
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
    foregrounds = sorted(foregrounds)
    backgrounds = sorted(backgrounds)
    for bg in backgrounds: # Calculate energy of background windows
        Lbg = bg[0] # Left bound of the background window
        Rbg = bg[1] # Right bound of the background window
        Ebg = 0
        for i in range(int(Lbg),int(Rbg)):
            Ebg = Ebg + (signals[i]*signals[i])
        Ebg = Ebg/(Rbg-Lbg+1) # Energy of the background window
        bgs.append(Ebg)
    for fg in foregrounds:
        Lfg = fg[0] # Left bound of the foreground window
        Rfg = fg[1] # Right bound of the foreground window
        Lbg = backgrounds[idx][0] # Left bound of the background window
        Rbg = backgrounds[idx][1] # Right bound of the background window\
        #print(int(Lfg),int(Rfg),int(Lbg),int(Rbg))
        if (int(Rfg) >= len(signals)): Rfg = len(signals)-1
        if (int(Lfg) >= len(signals)): Lfg = len(signals)-1
        Efg = 0
        for i in range(int(Lfg),int(Rfg)):
            Efg = Efg + (signals[i]*signals[i])
        Efg = Efg/(Rfg-Lfg+1) # Energy of the foreground window
        if (Lfg > Rbg): # If the foreground does not intersect the background window
            idx = idx+1
            #print(idx)
            if (idx == len(backgrounds)): break
            Lbg = backgrounds[idx][0] # Update left bound
            Rbg = backgrounds[idx][1] # Update right bound
        if (Lfg >= Lbg and Rfg <= Rbg): # If the foreground is fully contained in the background
            energies.append(Efg/bgs[idx])  # Energy ratio
        if (Lfg < Rbg and Rfg > Rbg): # If the foreground is parcially contained in the background
            conA = (Rbg-Lfg+1)/(Rfg-Lfg+1) # Porcentage contained in the first background window
            idx = idx+1
            Lbg = backgrounds[idx][0] # Update left bound
            Rbg = backgrounds[idx][1] # Update right bound
            conB = (Rfg-Lbg+1)/(Rfg-Lfg+1) # Porcentage contained in the second background
            energies.append(Efg/((bgs[idx-1]*conA)+(bgs[idx]*conB))) # Energy ratio
    return energies

        
def segment(signals,longi): # Divide the signals in segments
    segments = []
    tam = len(signals)
    l = 0
    while(l < tam):
        if (l+longi <= tam): # If is possible to construct a segment of lenght tam
            segments.append([l,l+longi])
            l = l+longi
        else: # Is is not possible to construct a segment of lenght tam
            segments.append([l,tam]) #
            l = tam
    return segments


def filterOut(signals,Flo,freq): # Filter out signal below Flo
    return mne.filter.filter_data(signals[0],sfreq=freq,l_freq=None ,h_freq=Flo,method='fir')

def evaluate_file(Flo, Wbg, Wfg, Te, d, Lmin, Tc, sample, n_files, reales_p, gen, ind):
    Fn = Fp = e = Tp = 0
    for k in range(n_files):
        print(f"GENERACION {gen} --- INDIVIDUO {ind} --- {sample}_{k}")

        detected_seizures = seizure_detection(Flo,Wbg, Wfg, Te, d, Lmin, Tc, f"samples/{sample}/{sample}_{k}")
        detected_seizures = sorted(detected_seizures)

        cant = 0
        i = j = 0
        reales = reales_p[k]

        while(i < len(detected_seizures) and j < len(reales)):
            ld, rd = detected_seizures[i]
            lr, rr = reales[j]
            if rr < ld:
                j += 1
            elif rd < lr:
                i += 1
            elif ((ld >= lr and ld <= rr) or (rd >= lr and rd <= rr) or (ld <= lr and rd >= rr)):
                print(f"Detecto ld={ld}, rd={rd}, lr={lr}, rr={rr}")
                cant += 1
                e += abs(ld - lr) + abs(rd - rr)
                i += 1
                j += 1
        Fp += abs(len(detected_seizures)-cant)
        Fn += abs(len(reales) - cant)
        Tp += cant
    print(f"---------------------{Fn},{Fp},{e},{Tp}---------------------")
    return Fn, Fp, e, Tp


def compute_fitness(chromosome,gen,ind):
    [Flo,Wfg,Wbg,Te,d,Lmin,Tc] = transform_to_params(chromosome) # Get parameters from the chromosome
    samples = ["chb01","chb05","chb12","chb13"]
    tams = [42,39,24,33]
    real_seizures = []
    real_seizures.append([[],[],[[2996*256,3036*256]], [[1467*256,1494*256]], [],[],[],[],[],[],
                         [],[],[],[],[[1732*256,1772*256]],[[1015*256,1066*256]],[],[[1720*256,1810*256]],[],[],
                         [[327*256,420*256]],[],[],[],[],[[1862*256,1963*256]],[],[],[],[],
                         [],[],[],[],[],[],[],[],[],[],[],[]]) #Chb01
    
    real_seizures.append([[],[],[], [], [],[[417*256,532*256]],[],[],[],[],
                         [],[],[[1086*256,1196*256]],[],[],[[2317*256,2413*256]],[[2451*256,2571*256]],[],[],[],
                         [],[[2348*256,2465*256]],[],[],[],[],[],[],[],[],
                         [],[],[],[],[],[],[],[],[]]) #Chb05
    
    real_seizures.append([[[1665*256,1726*256],[3415*256,3447*256]],[[1426*256,1439*256],[1591*256,1614*256],[1957*256,1977*256],[2798*256,2824*256]],
                          [[3082*256,3114*256],[3503*256,3535*256]],[[593*256,625*256],[811*256,856*256]], [[1085*256,1122*256]],[],[],[],
                          [[253*256,333*256],[425*256,522*256],[630*256,670*256]],[],
                          [[916*256,951*256],[1097*256,1124*256],[1728*256,1753*256],[1921*256,1963*256],[2388*256,2440*256],[2621*256,2669*256]],
                          [[181*256,215*256]],[[107*256,146*256],[554*256,592*256],[1163*256,1199*256],[1401*256,1447*256],[1884*256,1921*256],[3557*256,3584*256]],
                          [],[[2185*256,2206*256],[2427*256,2450*256]],[],[],[[653*256,680*256]],[],
                          [[1548*256,1573*256],[2798*256,2821*256],[2966*256,3009*256],[3146*256,3201*256],[3364*256,3410*256]],[],[],[],
                          [[699*256,750*256],[945*256,973*256],[1170*256,1199*256],[1676*256,1701*256],[2213*256,2236*256]]]) #Chb12
    
    
    real_seizures.append([[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[[2077*256,2121*256]],[[934*256,1004*256]],[],[],
                         [],[],[],[],[],[[142*256,173*256],[530*256,594*256]],[],[[458*256,478*256],[2436*256,2454*256]],[],[[2474*256,2491*256]],[[3339*256,3401*256]],
                         [[638*256,660*256]],[[851*256,916*256],[1626*256,1691*256],[2664*256,2721*256]]]) #Chb13
                         
    e = 0
    Fp = 0
    Fn = 0
    Tp = 0
    results = []
    with ProcessPoolExecutor(max_workers=len(samples)) as executor:
        futures = []
        for p,sample in enumerate(samples): # For every pacient
            futures.append(executor.submit(evaluate_file,Flo,Wbg,Wfg,Te,d,Lmin,Tc,sample,tams[p],real_seizures[p],gen,ind))
        for f in futures:
            results.append(f.result())
    Fn = sum(r[0] for r in results)
    Fp = sum(r[1] for r in results)
    e = sum(r[2] for r in results)
    Tp = sum(r[3] for r in results)
    fitness = (10**6)*Fn + (10**4)*Fp + e  
    print(f"Salio del loop con {Fn} {Fp} {e} {Tp}")
    return fitness 


def transform_to_params(chromosome):
    Flo = 1
    Wfg=0
    Wbg=0
    Te=0
    d=0
    Lmin=0
    Tc=0
    x = 0
    for i in range(0,6): # 6 bits for Flo
        Flo = Flo + (int(chromosome[i])<<i)
    for i in range(0,4): # 4 bits for Wfg
        x = x + (int(chromosome[6+i])<<i)
    Wfg = 0.5 + (x/2)
    x = 0
    for i in range(0,5): # 5 bits for Wbg
        x = x + (int(chromosome[10+i])<<i)
    Wbg = 60*(1+x)
    x = 0
    for i in range(0,6): # 6 bits for Te
        x = x + (int(chromosome[15+i])<<i)
    Te = 100 - (x/10)
    x = 0
    for i in range(0,5): # 5 bits for d
        x = x + (int(chromosome[21+i])<<i)
    d = 1 + x
    x = 0
    for i in range(0,5): # 5 bits for Lmin
        x = x + (int(chromosome[26+i])<<i)
    Lmin = 1+x
    x = 0
    for i in range(0,4): # 4 bits for Td
        x = x + (int(chromosome[31+i])<<i)
    Tc = (1+x)/16
    return [Flo,Wfg,Wbg,Te,d,Lmin,Tc]



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
    for _ in range(tam-2):
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
            cut = random.randint(0,34) 
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
            lim_inf = random.randint(0,34) 
            lim_sup = random.randint(0,34) 

            if (lim_inf > lim_sup): lim_inf, lim_sup = lim_sup, lim_inf  
            
            segmento_inv = child[lim_inf:lim_sup+1][::-1] 
            new_child = child[:lim_inf] + segmento_inv + child[lim_sup+1:] 
            
            new_childs.append(new_child) 
        else: new_childs.append(child) 
    return new_childs 

def elitism(population,fitnesses):
    mini1 = np.inf
    mini2 = np.inf
    id1 = 999999
    id2 = 999999
    for i in range(len(population)):
        if (fitnesses[i] <= mini1):
            mini2 = mini1
            id2 = id1
            mini1 = fitnesses[i]
            id1 = i
        elif (fitnesses[i] > mini1 and fitnesses[i]<=mini2):
            mini2 = fitnesses[i]
            id2 = i
    return [population[id1],population[id2]]

def getBestIndividual(popu):
    print("Entro en best ---------------")
    mini = np.inf
    idMini = 0
    for i in range(len(popu)):
        fit = compute_fitness(popu[i],50,i) 
        if (fit < mini):
            mini = fit
            idMini = i
    return [popu[idMini],mini]


def genetic_algorithm(size_population, tam_tournament, mutation_rate,cross_rate): 
    population = generate_random_population(size_population,35)
    for i in range(30):
        print(f"Generacion numero {i} -------------")
        selected = []
        fitnesses = []   
        for j in range(len(population)): 
            print(f"Entra individuo {j} de la gen {i}")
            fitnesses.append(compute_fitness(population[j],i,j))
            print(f"Sale individuo {j} de la gen {i}")
        while len(selected) < size_population:
            print("entro en seleccion")
            parents = selectedByTournament(population, fitnesses,tam_tournament)
            childs = reproduce(parents, cross_rate)
            selected = mutation(childs, mutation_rate)
            print(len(selected))
            selected = selected + elitism(population,fitnesses)
            print(len(selected), len(population))
        population = selected
    return getBestIndividual(population)


if __name__ == "__main__":
    print(genetic_algorithm(10,3,0.05,0.75))



