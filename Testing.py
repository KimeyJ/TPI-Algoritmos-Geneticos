import mne
import random
import numpy as np
import random

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

def evaluate_file(Flo, Wbg, Wfg, Te, d, Lmin, Tc, sample, n_files, reales_p):
    Fn = Fp = e = Tp = 0
    #print(n_files)
    for k in range(n_files):
        #print(f"entro con {k} de {n_files}")
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
    return [Fn, Fp, e, Tp]


def testing(chromosome):
    [Flo,Wfg,Wbg,Te,d,Lmin,Tc] = transform_to_params(chromosome) # Get parameters from the chromosome
    samples = ["chb01","chb02","chb03","chb04","chb05","chb06","chb07","chb08","chb09","chb10","chb11","chb12","chb13",
               "chb14","chb15","chb16","chb17","chb18","chb19","chb20","chb21","chb22","chb23","chb24"]
    tams = [42,36,38,42,39,18,19,20,19,25,35,24,33,26,40,19,21,36,30,29,33,31,9,22]
    real_seizures = []
    real_seizures.append([[],[],[[2996*256,3036*256]], [[1467*256,1494*256]], [],[],[],[],[],[],
                         [],[],[],[],[[1732*256,1772*256]],[[1015*256,1066*256]],[],[[1720*256,1810*256]],[],[],
                         [[327*256,420*256]],[],[],[],[],[[1862*256,1963*256]],[],[],[],[],
                         [],[],[],[],[],[],[],[],[],[],[],[]]) #Chb01
    
    real_seizures.append([[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[[130*256,212*256]],[[2972*256,3053*256]],[],[],[[3369*256,3378*256]],
                          [],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[]]) #Chb02

    real_seizures.append([[[362*256,414*256]],[[731*256,796*256]],[[432*256,501*256]],[[2162*256,2214*256]],[],[],[],[],[],[],[],[],[],[],[],
                          [],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[[1982*256,2029*256]],[[2592*256,2656*256]],[[1725*256,1778*256]],[],[]]) #Chb03
    
    real_seizures.append([[],[],[],[],[[7804*256,7853*256]],[],[],[[6446*256,6557*256]],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],
                          [[1679*256,1781*256],[3782*256,3898*256]],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],]) #Chb04
    
    real_seizures.append([[],[],[], [], [],[[417*256,532*256]],[],[],[],[],
                         [],[],[[1086*256,1196*256]],[],[],[[2317*256,2413*256]],[[2451*256,2571*256]],[],[],[],
                         [],[[2348*256,2465*256]],[],[],[],[],[],[],[],[],
                         [],[],[],[],[],[],[],[],[]]) #Chb05
    
    real_seizures.append([[[1724*256,1738*256],[7461*256,7476*256],[13525*256,13540*256]],[],[],[[327*256,347*256],[6211*256,6231*256]],[],[],[],[],
                          [[12500*256,12516*256]],[[10833*256,10845*256]],[],[[506*256,519*256]],[],[],[],[],[[7799*256,7811*256]],[[9387*256,9403*256]]]) #Chb06
    
    real_seizures.append([[],[],[],[],[],[],[],[],[],[],[],[[4920*256,5006*256]],[[3285*256,3381*256]],[],[],[],[],[],[[13688*256,13831*256]]]) #Chb07

    real_seizures.append([[[2670*256,2841*256]],[],[],[[2856*256,3046*256]],[],[[2988*256,3122*256]],[],[[2417*256,2577*256]],[],[],[],[],[],[],[],
                          [[2083*256,2347*256]],[],[],[],[]]) #Chb08
    
    real_seizures.append([[],[],[],[],[[12231*256,12295*256]],[],[[2951*256,3030*256],[9196*256,9267*256]],[],[],[],[],[],[],[],[],[],[],
                          [[5299*256,5361*256]]]) #Chb09
    
    real_seizures.append([[],[],[],[],[],[],[],[],[[6313*256,6348*256]],[],[],[],[],[],[],[],[[6888*256,6958*256]],[],[],[[2382*256,2447*256]],[],
                          [[3021*256,3079*256]],[[3801*256,3877*256]],[[4618*256,4707*256]],[[1383*256,1437*256]]]) #Chb10
    
    real_seizures.append([[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[[298*256,320*256]],
                          [[2695*256,2727*256]],[[1454*256,2206*256]]]) #Chb11
    
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
    
    real_seizures.append([[],[],[[1986*256,2000*256]],[[1372*256,1392*256],[2817*256,2839*256]],[[1911*256,1925*256]],[],[[1838*256,1879*256]],[],[],[],[],
                          [[3239*256,3259*256]],[[1039*256,1061*256]],[],[],[],[],[],[],[[2833*256,2849*256]],[],[],[],[],[],[],]) #Chb14
    
    real_seizures.append([[],[],[],[],[],[[272*256,397*256]],[],[],[],[[1082*256,1113*256]],[],[],[],[],[[1591*256,1748*256]],[],[[1925*256,1960*256]],[],
                          [[607*256,662*256]],[[760*256,965*256]],[],[[876*256,1066*256]],[],[],[[1751*256,1871*256]],[],[],[],[],
                          [[834*256,894*256],[2378*256,2497*256],[3362*256,3425*256]],[],[[3322*256,3429*256]],[[1108*256,1248*256]],[],[],
                          [[778*256,849*256]],[[263*256,318*256],[843*256,1020*256],[1524*256,1595*256],[2179*256,2250*256],[3428*256,3460*256]],[],
                          [[751*256,859*256]],[]]) #Chb15
    
    real_seizures.append([[],[],[],[],[],[],[],[],[],[[2290*256,2299*256]],[[1120*256,1129*256]],[],[],[[1854*256,1868*256]],[],[[1214*256,1220*256]],
                          [[227*256,236*256],[1694*256,1700*256],[2162*256,2170*256],[3290*256,3298*256]],[[627*256,635*256],[1909*256,1916*256]],[]]) #Chb16
    
    real_seizures.append([[[2282*256,2372*256]],[[3025*256,3140*256]],[],[],[],[],[],[],[],[[3136*256,3224*256]],[],[],[],[],[],[],[],[],[],[],[]]) #Chb17

    real_seizures.append([[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[[3477*256,3527*256]],[[541*256,571*256]],
                          [[2087*256,2155*256]],[[1908*256,1963*256]],[],[],[[2196*256,2264*256]],[[463*256,509*256]]]) #Chb18
    
    real_seizures.append([[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[[299*256,377*256]],[[2964*256,3041*256]],
                          [[3159*256,3240*256]]]) #Chb19
    
    real_seizures.append([[],[],[],[],[],[],[],[],[],[[94*256,123*256]],[[1440*256,1470*256],[2498*256,2537*256]],[[1971*256,2009*256]],
                          [[390*256,425*256],[1689*256,1738*256]],[[2226*256,2261*256]],[],[],[],[],[],[],[],[],[],[],[],[],[],[],
                          [[1393*256,1432*256]]]) #Chb20
    
    real_seizures.append([[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[[1288*256,1344*256]],[[2627*256,2677*256]],[[2003*256,2084*256]],
                          [[2553*256,2565*256]],[],[],[],[],[],[],[],[],[],[],[],]) #Chb21
    
    real_seizures.append([[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[[3367*256,3425*256]],[],[],[],[],[[3139*256,3213*256]],[],[],[],[],[],
                          [[1263*256,1335*256]],[],[],[]]) #Chb22
    
    real_seizures.append([[[3962*256,4075*256]],[],[[325*256,345*256],[5104*256,5151*256]],
                          [[2589*256,2660*256],[6885*256,6947*256],[8505*256,8532*256],[9580*256,9664*256]],[],[],[],[],[],]) #Chb23
    
    real_seizures.append([[[480*256,505*256],[2451*256,2476*256]],[[231*256,260*256],[2883*256,2908*256]],[[1088*256,1120*256],[1411*256,1438*256],[1745*256,1764*256]],
                          [[1229*256,1253*256]],[[38*256,60*256]],[[1745*256,1764*256]],[[3527*256,3597*256]],[[3288*256,3304*256]],[[1939*256,1966*256]],
                          [[3552*256,3569*256]],[[3515*256,3581*256]],[[2804*256,2872*256]]]) #Chb24
                         
    results = []
    for p,sample in enumerate(samples): # For every pacient
        #print(p, sample)
        results.append(evaluate_file(Flo,Wbg,Wfg,Te,d,Lmin,Tc,sample,tams[p],real_seizures[p]))
    for ans in results:
        print(f"{ans[0]} -- {ans[1]} -- {ans[2]} -- {ans[3]}")
    print(f"individuo= {chromosome}")
    print(f"Flo={Flo} - Wfg={Wfg} - Wbg={Wbg} - Te={Te} - d={d} - Lmin={Lmin} - Tc={Tc}")
    return 


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



if __name__ == "__main__":
    testing("00101010011010111011001111111110101")



