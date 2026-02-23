import time # sleep to slow searches
from sys import exit, float_info # nuitka seems to need this

from art_class import *


### GLOBALS

ALL_ROOTS = ["ROOT_MAIN", "ROOT_HUE", "ROOT_SAT", "ROOT_VAL", "ROOT_PAL", "ROOT_GRAD", "ROOT_TARGET", "ROOT_AIM"]

ARB = None
RIG = None


### INITIALIZE

def initialize_elements():

    # if elements are added or removed,
    # other functions need to be updated:
    # 
    # art.py: default settings (<ELEMENT> WEIGHT)
    # -- base frequency of isotopes of this element, in generating new artists
    # 
    # art_class.py: <ELEMENT>Concept
    # -- define a concept = distribution for its parameters
    #
    # art_scheme.py: random_concepts_<ELEMENT>
    # -- a function that generates n concepts for it
    #
    # art_scheme.py: conceive_<ELEMENT>
    # -- a function that applies a concept for it = samples from that distribution to choose parameters for it

    global ALL_ELEMENTS

    ALL_ELEMENTS["X"] = fx = Element("X", 0, 2.32, random_concepts_cart, conceive_cart, [0])
    # fx.func = lambda x,y: abs(x)-y[0] # because might be fed negative x from IT
    fx.func = lambda x,y: bound(x-y[0]) # might be fed negative x from IT
    fx.texter = lambda x,y: "X - "+param_to_str(y[0])
    fx.cxer = None
    
    ALL_ELEMENTS["Y"] = fy = Element("Y", 0, 2.32, random_concepts_cart, conceive_cart, [0])
    # fy.func = lambda x,y: abs(x)-y[0] # because might be fed negative y from IT
    fy.func = lambda x,y: bound(x-y[0]) # might be fed negative y from IT
    fy.texter = lambda x,y: "Y - "+param_to_str(y[0])
    fy.cxer = None
    
    ALL_ELEMENTS["RAND"] = frand = Element("RAND", 1, 25.51, random_concepts_rand, conceive_rand, [0,1])
    frand.func = lambda x,y: ity(rand(math.floor(x[0]*y[0])), rand(math.ceil(x[0]*y[0])), restrict(sigmoid(derestrict((x[0]*y[0]) % 1), 0, y[1])))
    frand.texter = lambda x,y: "RAND("+", ".join(["x"+str(c) for c in x]+[param_to_str(p) for p in y])+")"
    frand.cxer = lambda c,y: c[0] * y[0]
    
    ALL_ELEMENTS["INV"] = finv = Element("INV", 1, 13.99, lambda s: [None] * s, None, [])
    finv.func = lambda x,y: -x[0]
    finv.texter = lambda x,y: "- x"+str(x[0])
    finv.cxer = lambda c,y: c[0]
    
    ALL_ELEMENTS["POW"] = fpow = Element("POW", 1, 16.17, random_concepts_pow, conceive_pow, [0])
    fpow.func = lambda x,y: bound(tame_pow(x[0], y[0]))
    fpow.texter = lambda x,y: "x"+str(x[0])+" [^] "+param_to_str(y[0])
    fpow.cxer = lambda c,y: pow_complexity(c[0], y[0])
    
    ALL_ELEMENTS["POWER"] = fpower = Element("POWER", 2, 12.07, random_concepts_power, conceive_power, [0])
    fpower.func = lambda x,y: bound(tame_pow(y[0]*x[0], y[2]+y[1]*abs(x[1]))) if y[0] != 0 and x[0] != 0 else 0
    fpower.texter = lambda x,y: "("+param_to_str(y[0])+" * x"+str(x[0])+") [^] ("+param_to_str(y[1])+" * "+"|x"+str(x[1])+"| + "+param_to_str(y[2])+")"
    fpower.cxer = lambda c,y: sum(c) + 1
    
    ALL_ELEMENTS["SIGMOID"] = fsigmoid = Element("SIGMOID", 1, 19.24, random_concepts_sigmoid, conceive_sigmoid, [0, 1])
    fsigmoid.func = lambda x,y: bound(sigmoid(x[0], y[0], y[1]))
    fsigmoid.texter = lambda x,y: "SIGMOID" + "("+", ".join(["x"+str(x[0])] + [param_to_str(p) for p in y])+")"
    fsigmoid.cxer = lambda c,y: sigmoid_complexity(c[0], y[2])
    
    ALL_ELEMENTS["ARCFAN"] = farcfan = Element("ARCFAN", 2, 11.67, random_concepts_arcfan, conceive_arcfan, [0,1])
    farcfan.func = lambda x,y: arcfan(y[1]*(y[0]+x[0]), x[1])
    farcfan.texter = lambda x,y: "ARCFAN("+"(x"+str(x[0])+" + "+param_to_str(y[0])+") * "+param_to_str(y[1])+", x"+str(x[1])+")"
    farcfan.cxer = lambda c,y: sum(c) + 1
    
    ALL_ELEMENTS["SIN"] = fsin = Element("SIN", 1, 19.02, random_concepts_sin, conceive_sin, [0,1])
    fsin.func = lambda x,y: math.sin(y[0]*x[0]+y[1])
    fsin.texter = lambda x,y: "SIN("+param_to_str(y[0])+" * x"+str(x[0])+" + "+param_to_str(y[1])+")"
    fsin.cxer = lambda c,y: c[0] * (y[0] / (2*math.pi))
    
    ALL_ELEMENTS["SPIN"] = fspin = Element("SPIN", 3, 9.65, random_concepts_spin, conceive_spin, [0, 1, 2, 3, 4])
    fspin.func = lambda x,y: math.sin(y[0]*x[0]+y[1]-y[2]*arctango(y[4]*(y[3]+x[1]), x[2]))
    fspin.texter = lambda x,y: "SIN("+"x"+str(x[0])+" * "+param_to_str(y[0])+" + "+param_to_str(y[1])+" - "+param_to_str(y[2])+"*ARCTANGO((x"+str(x[1])+" + "+param_to_str(y[3])+") * "+param_to_str(y[4])+", x"+str(x[2])+"))"
    fspin.cxer = lambda c,y: sum(c) * (y[0] / (2*math.pi)) * abs(y[2])
    
    ALL_ELEMENTS["MINX"] = fminx = Element("MINX", -1, 17.08, random_concepts_minx, conceive_minx, [0])
    fminx.func = lambda x,y: max([(x[i]-1)*y[1][i]+1 for i in range(len(x))]) if y[0] else min([(x[i]-1)*y[1][i]+1 for i in range(len(x))])
    fminx.texter = lambda x,y: ("MAXX" if y[0] else "MINX") + "(["+", ".join(["x"+str(c) for c in x])+"], ["+", ".join([param_to_str(p) for p in y[1]])+"])"
    fminx.cxer = lambda c,y: sum(c)
    
    ALL_ELEMENTS["AMEAN"] = famean = Element("AMEAN", -1, 18.84, random_concepts_amean, conceive_amean, [])
    famean.func = lambda x,y: bound(dot([xx for xx in x], y[0]))
    famean.texter = lambda x,y: "["+", ".join(["x"+str(c) for c in x])+"] dot ["+", ".join([param_to_str(p) for p in y[0]])+"]"
    famean.cxer = lambda c,y: sum(c)
    
    ALL_ELEMENTS["GMEAN"] = fgmean = Element("GMEAN", -1, 12.86, lambda s: [None] * s, None, [])
    fgmean.func = lambda x,y: bound(geomean(x))
    fgmean.texter = lambda x,y: "("+" * ".join(["x"+str(c) for c in x])+")"+" [^] (1/"+str(len(x))+")"
    fgmean.cxer = lambda c,y: sum(c)
    
    ALL_ELEMENTS["IT"] = fit = Element("IT", 5, 28.77, random_concepts_it, conceive_it, [])
    fit.func = func_it
    fit.texter = lambda x,y: "ITERATE("+", ".join(["fx"+str(x[0]), "fx"+str(x[1]), "(x"+str(x[2])+"+1)/2"]+[param_to_str(p) for p in y[:2]])+")["+("0" if y[2] else "1")+"]"
    fit.cxer = lambda c,y: ((c[0]+c[1])/2)*(max(y[0]/2, 1))



### APPLY VERSION / SUBCONCEIVERS

def conceive_fan(fc):
    
    shift = betta(*fc.shift_dist)
    
    weight = 2 ** betta(*fc.logweight_dist)
    
    return shift, weight

def conceive_wave(wc):
    
    freq = 100 * betta(*wc.freqhood_dist) ** 4.3 + 1
    
    if random.random() <= wc.nice_chance: freq = int(freq)
    
    numer = int(100 * betta(*wc.numerhood_dist) ** 4.3 + 1)
    denom = int(100 * betta(*wc.denomhood_dist) ** 4.3 + 1)
    
    freq *= numer/denom
    
    if random.random() <= wc.unnice_chance:
        freq += betta(*wc.unnice_dist)
        freq = max(freq, 0)
        
    ###

    offreq = int(betta(*wc.offreqhood_dist) ** 4.3 * 100 + 1)
        
    if offreq == 1: offdex = 0 if betta(*wc.offdex_dist) <= 0.5 else 1
    else: offdex = round(betta(*wc.offdex_dist) * (offreq-1) + 0.5) / offreq
    
    offset = 2*math.pi * (offdex/offreq + betta(*wc.off_unnice_dist))
    
    ###
    
    factor = 2 * math.pi * freq
    
    return factor, offset
    
def conceive_weights(wc, num_weights):
    
    balanced_weights = [1] * int(betta(*wc.balanced_amt_dist) * num_weights)
    
    imbalanced_base = betta(*wc.imbalanced_base_dist)
    
    imbalanced_weights = [imbalanced_base ** (i+1) for i in range(num_weights - len(balanced_weights))]
    
    weights = sorted(balanced_weights + imbalanced_weights, reverse=True)
    
    if random.random() <= wc.reversing_chance: weights.reverse()
        
    weights = partial_shuffle(weights, betta(*wc.shufflity_dist))
    
    starting_index = int(betta(*wc.starting_index_dist) * len(weights))
    
    weights = [weights[(starting_index+i) % len(weights)] for i in range(len(weights))]
    
    return weights

def conceive_cart(atom, cords, concept):
    
    if random.random() <= concept.nice_chance:
    
        freq = int(100 * betta(*concept.freqhood_dist) ** 4.3 + 1)
        
        if freq == 1: pole = 0 if betta(*concept.index_dist) <= 0.5 else 1
            
        else: pole = round(betta(*concept.index_dist) * (freq-1) + 0.5) / freq
        
        if random.random() <= concept.unnice_chance:
        
            pole = min(max(pole + betta(*concept.unnice_dist) / freq, 0), 1)
        
    else:
    
        pole = betta(*concept.mean_dist)
        
    atom.params = [pole, concept.updating]
          
def conceive_rand(atom, cords, concept):

    frequency = int(betta(*concept.freqhood_dist) ** 4.3 * 100 + 1)

    squarity = betta(*concept.squarity_dist)

    atom.params = [frequency, squarity]
                
def conceive_pow(atom, cords, concept):
    
    exponent = 2 ** betta(*concept.logex_dist)
    
    atom.params = [exponent]
    
def conceive_power(atom, cords, concept):

    if random.random() <= concept.weighing_base_chance:
    
        base_weight = betta(*concept.base_weight_dist)
        
        exponent_weight = betta(*concept.exp_weight_dist)
        
        exponent_lift = 0
        
    else:
    
        base_weight = 1
        
        exponent_weight = betta(*concept.exp_weight_dist)
        
        exponent_lift = betta(*concept.exp_lifthood_dist) * (1 - exponent_weight)
    
    atom.params = [base_weight, exponent_weight, exponent_lift]
      
def conceive_sigmoid(atom, cords, concept):
    
    midpoint = betta(*concept.midpoint_dist)

    squarity = betta(*concept.squarity_dist)
    
    atom.params = [midpoint, squarity]
           
def conceive_arcfan(atom, cords, concept):
    
    shift, weight = conceive_fan(concept.fc)
    
    atom.params = [shift, weight]
    
def conceive_sin(atom, cords, concept):

    factor, offset = conceive_wave(concept.wc)
    
    atom.params = [factor, offset]
    
def conceive_spin(atom, cords, concept):

    factor, offset = conceive_wave(concept.wc)
    
    num_arms = int(betta(*concept.armhood_dist) ** 3 * 14 + 1)
    
    if random.random() > concept.antiwise_chance: num_arms *= -1
    
    spiral_shift, spiral_weight = conceive_fan(concept.fc)
    
    atom.params = [factor, offset, num_arms, spiral_shift, spiral_weight]
    
def conceive_minx(atom, cords, concept):
    
    maxing = random.random() <= concept.maxing_chance
    
    weights = conceive_weights(concept.wc, len(atom.children))
    
    norm = max(weights)
    weights = list(map(lambda x: x/norm, weights))
    
    atom.params = [maxing, weights]
            
def conceive_amean(atom, cords, concept):

    weights = conceive_weights(concept.wc, len(atom.children))
        
    # normalize(weights)
    
    atom.params = [weights]
    
def conceive_it(atom, cords, concept):
    
    iterance_0 = betta(*concept.iterance_0_dist)
    iterance_1 = betta(*concept.iterance_1_dist)
    
    iterance_low, iterance_high = sorted((iterance_0, iterance_1))
    
    is_xlike = random.random() <= concept.xlike_chance
    
    atom.params = [iterance_low, iterance_high, is_xlike]



### GENERATE VERSION / CONCEPTS FOR SUBCONCEIVERS
   
def random_concepts_fan(num_concepts):

    shift_mids = [-1, 0, 1, random.uniform(-1, 1)]
    shift_mids_w = reweight([1, 2, 1, ARB], 5)
    
    shift_sharps = [0, 0.5, 1, None, random.random()]
    shift_sharps_w = reweight([1, 1, 1, 2*RIG, ARB], 5)
    
    shift_boosts = [0, 0.125, random.random()**3]
    shift_boosts_w = reweight([2, 1, ARB], 5)

    logweight_lownesses = [0, 0.5, 1, random.random()**2*10, random.random()**2*10]
    logweight_lownesses_w = reweight([1, 1, 1, ARB, ARB], 5)
    
    logweight_mids = [0, random.uniform(-10, 10), random.uniform(-10, 10)]
    logweight_mids_w = reweight([2, ARB, ARB], 5)

    logweight_highnesses = [0, 0.5, 1, random.random()**2*10, random.random()**2*10]
    logweight_highnesses_w = reweight([1, 1, 1, ARB, ARB], 5)
    
    logweight_sharps = [0, 0.5, 1, None, random.random()]
    logweight_sharps_w = reweight([1, 1, 1, 2*RIG, ARB], 5)
    
    logweight_boosts = [0, 0.125, random.random()**3]
    logweight_boosts_w = reweight([2, 1, ARB], 5)

    concepts = []
    
    for i in range(num_concepts):
    
        shift_mid = random.choices(shift_mids, shift_mids_w, k=1)[0]
        shift_sharp = random.choices(shift_sharps, shift_sharps_w, k=1)[0]
        shift_boost = random.choices(shift_boosts, shift_boosts_w, k=1)[0]
        shift_dist = (-1, shift_mid, 1, shift_sharp, shift_boost)
    
        logweight_lowness = random.choices(logweight_lownesses, logweight_lownesses_w, k=1)[0]
        logweight_mid = random.choices(logweight_mids, logweight_mids_w, k=1)[0]
        logweight_highness = random.choices(logweight_highnesses, logweight_highnesses_w, k=1)[0]
        logweight_sharp = random.choices(logweight_sharps, logweight_sharps_w, k=1)[0]
        logweight_boost = random.choices(logweight_boosts, logweight_boosts_w, k=1)[0]
        logweight_dist = (logweight_mid-logweight_lowness, logweight_mid, logweight_mid+logweight_highness, logweight_sharp, logweight_boost)
        
        concepts.append(FanConcept(shift_dist, logweight_dist))
        
    return concepts
  
def random_concepts_weights(num_concepts):
    
    balanced_amt_mids = [0, 1, random.random(), random.random()]
    balanced_amt_mids_w = reweight([1, 1, ARB, ARB], 5)
    
    balanced_amt_sharps = [0, 0.5, 1, None, random.random()]
    balanced_amt_sharps_w = reweight([1, 1, 1, 2*RIG, ARB], 5)
    
    imbalanced_base_mids = [1/(random.random()**3*4+1), 1/5, 1, 5, random.random()**3*4+1]
    imbalanced_base_mids_w = reweight([ARB, 1, 3, 1, ARB], 5)
    
    imbalanced_base_sharps = [0, 0.5, 1, None, random.random()]
    imbalanced_base_sharps_w = reweight([1, 1, 1, 2*RIG, ARB], 5)
    
    reversing_chances = [0, 0.5, 1, random.random()]
    reversing_chances_w = reweight([1, 1, 1, ARB], 5)
    
    shufflity_mids = [0, 1, random.random()**2]
    shufflity_mids_w = reweight([1, 1, ARB], 5)
    
    shufflity_sharps = [0, 0.5, 1, None, random.random()]
    shufflity_sharps_w = reweight([1, 1, 1, 2*RIG, ARB], 5)
    
    starting_index_mids = [0, random.random(), random.random()]
    starting_index_mids_w = reweight([2, ARB, ARB], 5)
    
    starting_index_sharps = [0, 0.5, 1, None, random.random()]
    starting_index_sharps_w = reweight([1, 1, 1, 2*RIG, ARB], 5)
    
    concepts = []
        
    for i in range(num_concepts):
        
        balanced_amt_mid = random.choices(balanced_amt_mids, balanced_amt_mids_w, k=1)[0]
        balanced_amt_sharp = random.choices(balanced_amt_sharps, balanced_amt_sharps_w, k=1)[0]
        balanced_amt_dist = (0, balanced_amt_mid, 1, balanced_amt_sharp, 0)
        
        imbalanced_base_mid = random.choices(imbalanced_base_mids, imbalanced_base_mids_w, k=1)[0]
        imbalanced_base_sharp = random.choices(imbalanced_base_sharps, imbalanced_base_sharps_w, k=1)[0]
        imbalanced_base_dist = (0, imbalanced_base_mid, 5, imbalanced_base_sharp, 0)
    
        reversing_chance = random.choices(reversing_chances, reversing_chances_w, k=1)[0]
        
        shufflity_mid = random.choices(shufflity_mids, shufflity_mids_w, k=1)[0]
        shufflity_sharp = random.choices(shufflity_sharps, shufflity_sharps_w, k=1)[0]
        shufflity_dist = (0, shufflity_mid, 1, shufflity_sharp, 0)
        
        starting_index_mid = random.choices(starting_index_mids, starting_index_mids_w, k=1)[0]
        starting_index_sharp = random.choices(starting_index_sharps, starting_index_sharps_w, k=1)[0]
        starting_index_dist = (0, starting_index_mid, 1, starting_index_sharp, 0)
        
        concepts.append(WeightConcept(balanced_amt_dist, imbalanced_base_dist, reversing_chance, shufflity_dist, starting_index_dist))
        
    return concepts

def random_concepts_wave(num_concepts):


    nice_chances = [0, 0.5, 1, random.random()**0.5]
    nice_chances_w = reweight([1, 1, 2, ARB], 5)
    
    unnice_chances = [0, 0.5, 1, random.random()**2]
    unnice_chances_w = reweight([2, 1, 1, ARB], 5)
    
    off_unnice_chances = [0, 0.5, 1, random.random()**2]
    off_unnice_chances_w = reweight([2, 1, 1, ARB], 5)
    
    ######
    
    freqhood_mids = [0, 0.35, 0.45, random.random()]
    freqhood_mids_w = reweight([1, 1, 1, ARB*3], 5)
    
    freqhood_sharps = [0, 0.5, 1, None, random.random()]
    freqhood_sharps_w = reweight([1, 1, 1, 2*RIG, ARB], 5)
    
    freqhood_boosts = [0, 0.125, random.random()**3]
    freqhood_boosts_w = reweight([2, 1, ARB], 5)
    
    ###
    
    numerhood_mids = [0, 0.35, 0.45, random.random()**2]
    numerhood_mids_w = reweight([1, 1, 1, ARB], 5)
    
    numerhood_sharps = [0, 0.5, 1, None, random.random()]
    numerhood_sharps_w = reweight([1, 1, 1, 2*RIG, ARB], 5)
    
    ###
    
    denomhood_mids = [0, 0.35, 0.45, random.random()**2]
    denomhood_mids_w = reweight([1, 1, 1, ARB], 5)
    
    denomhood_sharps = [0, 0.5, 1, None, random.random()]
    denomhood_sharps_w = reweight([1, 1, 1, 2*RIG, ARB], 5)
    
    ###
    
    unnice_mids = [-0.5, 0, 0.5, random.random()**2*random.choice((-1, 1))]
    unnice_mids_w = reweight([1, 3, 1, ARB*2], 5)
    
    unnice_sharps = [0, 0.5, 1, None, random.random()]
    unnice_sharps_w = reweight([1, 1, 1, 2*RIG, ARB], 5)
    
    ######
    
    offreqhood_mids = [0, 0.35, 0.45, random.random()**2]
    offreqhood_mids_w = reweight([1, 1, 1, ARB], 5)
    
    offreqhood_sharps = [0, 0.5, 1, None, random.random()]
    offreqhood_sharps_w = reweight([1, 1, 1, 2*RIG, ARB], 5)
    
    offreqhood_boosts = [0, 0.125, random.random()**3]
    offreqhood_boosts_w = reweight([2, 1, ARB], 5)
    
    ###
    
    offdex_mids = [0, 0.5, 1, random.random()]
    offdex_mids_w = reweight([1, 1, 1, ARB], 5)
    
    offdex_sharps = [0, 0.5, 1, None, random.random()]
    offdex_sharps_w = reweight([1, 1, 1, 2*RIG, ARB], 5)
    
    ###
    
    off_unnice_mids = [-0.5, 0, 0.5, random.random()**2*random.choice((-1, 1))]
    off_unnice_mids_w = reweight([1, 3, 1, ARB*2], 5)
    
    off_unnice_sharps = [0, 0.5, 1, None, random.random()]
    off_unnice_sharps_w = reweight([1, 1, 1, 2*RIG, ARB], 5)
    
    
    concepts = []
    
    for i in range(num_concepts):
    
    
        nice_chance = random.choices(nice_chances, nice_chances_w, k=1)[0]
        
        unnice_chance = random.choices(unnice_chances, unnice_chances_w, k=1)[0]
        
        off_unnice_chance = random.choices(off_unnice_chances, off_unnice_chances_w, k=1)[0]
        
        freqhood_mid = random.choices(freqhood_mids, freqhood_mids_w, k=1)[0]
        freqhood_sharp = random.choices(freqhood_sharps, freqhood_sharps_w, k=1)[0]
        freqhood_boost = random.choices(freqhood_boosts, freqhood_boosts_w, k=1)[0]
        freqhood_dist = (0, freqhood_mid, 1, freqhood_sharp, freqhood_boost)
        
        numerhood_mid = random.choices(numerhood_mids, numerhood_mids_w, k=1)[0]
        numerhood_sharp = random.choices(numerhood_sharps, numerhood_sharps_w, k=1)[0]
        numerhood_dist = (0, numerhood_mid, 1, numerhood_sharp, 0)
        
        denomhood_mid = random.choices(denomhood_mids, denomhood_mids_w, k=1)[0]
        denomhood_sharp = random.choices(denomhood_sharps, denomhood_sharps_w, k=1)[0]
        denomhood_dist = (0, denomhood_mid, 1, denomhood_sharp, 0)
        
        unnice_mid = random.choices(unnice_mids, unnice_mids_w, k=1)[0]
        unnice_sharp = random.choices(unnice_sharps, unnice_sharps_w, k=1)[0]
        unnice_dist = (-1, unnice_mid, 1, unnice_sharp, 0)
        
        offreqhood_mid = random.choices(offreqhood_mids, offreqhood_mids_w, k=1)[0]
        offreqhood_sharp = random.choices(offreqhood_sharps, offreqhood_sharps_w, k=1)[0]
        offreqhood_boost = random.choices(offreqhood_boosts, offreqhood_boosts_w, k=1)[0]
        offreqhood_dist = (0, offreqhood_mid, 1, offreqhood_sharp, offreqhood_boost)
        
        offdex_mid = random.choices(offdex_mids, offdex_mids_w, k=1)[0]
        offdex_sharp = random.choices(offdex_sharps, offdex_sharps_w, k=1)[0]
        offdex_dist = (0, offdex_mid, 1, offdex_sharp, 0)
        
        off_unnice_mid = random.choices(off_unnice_mids, off_unnice_mids_w, k=1)[0]
        off_unnice_sharp = random.choices(off_unnice_sharps, off_unnice_sharps_w, k=1)[0]
        off_unnice_dist = (-1, off_unnice_mid, 1, off_unnice_sharp, 0)
        
        concepts.append(WaveConcept(nice_chance, unnice_chance, off_unnice_chance, freqhood_dist, numerhood_dist, denomhood_dist, unnice_dist, offreqhood_dist, offdex_dist, off_unnice_dist))
        
    return concepts
  
def random_concepts_cart(num_concepts):

    nice_chances = [0, 0.5, 1, random.random()]
    nice_chances_w = reweight([1, 1, 1, ARB], 5)
    
    unnice_chances = [0, 0.5, 1, random.random()]
    unnice_chances_w = reweight([1, 1, 1, ARB], 5)
    
    mean_mids = [0, 0.5, 1, random.random()]
    mean_mids_w = reweight([1, 1, 1, ARB], 5)
    
    mean_sharps = [0, 0.5, 1, None, random.random()]
    mean_sharps_w = reweight([1, 1, 1, 2*RIG, ARB], 5)
    
    mean_boosts = [0, 0.125, random.random()**3]
    mean_boosts_w = reweight([2, 1, ARB], 5)
    
    freqhood_mids = [0, 0.35, 0.45, random.random()]
    freqhood_mids_w = reweight([1, 1, 1, ARB*3], 5)
    
    freqhood_sharps = [0, 0.5, 1, None, random.random()]
    freqhood_sharps_w = reweight([1, 1, 1, 2*RIG, ARB], 5)
    
    freqhood_boosts = [0, 0.125, random.random()**3]
    freqhood_boosts_w = reweight([2, 1, ARB], 5)
    
    index_mids = [0, 1, random.random()]
    index_mids_w = reweight([1, 1, ARB], 5)
    
    index_sharps = [0, 0.5, 1, None, random.random()]
    index_sharps_w = reweight([1, 1, 1, 2*RIG, ARB], 5)
    
    index_boosts = [0, 0.125, random.random()**3]
    index_boosts_w = reweight([2, 1, ARB], 5)
    
    unnice_mids = [-0.5, 0, 0.5, random.uniform(-0.5, 0.5)]
    unnice_mids_w = reweight([1, 2, 1, ARB], 5)
    
    unnice_sharps = [0, 0.5, 1, None, random.random()]
    unnice_sharps_w = reweight([1, 1, 1, 2*RIG, ARB], 5)
    
    unnice_boosts = [0, 0.125, random.random()**3]
    unnice_boosts_w = reweight([2, 1, ARB], 5)
    
    updates = [True, False]
    updates_w = reweight([1, 1], 5)
    
    concepts = []
    
    for i in range(num_concepts):
    
        nice_chance = random.choices(nice_chances, nice_chances_w, k=1)[0]
        
        unnice_chance = random.choices(unnice_chances, unnice_chances_w, k=1)[0]
        
        mean_mid = random.choices(mean_mids, mean_mids_w, k=1)[0]
        mean_sharp = random.choices(mean_sharps, mean_sharps_w, k=1)[0]
        mean_boost = random.choices(mean_boosts, mean_boosts_w, k=1)[0]
        mean_dist = (0, mean_mid, 1, mean_sharp, mean_boost)
        
        freqhood_mid = random.choices(freqhood_mids, freqhood_mids_w, k=1)[0]
        freqhood_sharp = random.choices(freqhood_sharps, freqhood_sharps_w, k=1)[0]
        freqhood_boost = random.choices(freqhood_boosts, freqhood_boosts_w, k=1)[0]
        freqhood_dist = (0, freqhood_mid, 1, freqhood_sharp, freqhood_boost)
        
        index_mid = random.choices(index_mids, index_mids_w, k=1)[0]
        index_sharp = random.choices(index_sharps, index_sharps_w, k=1)[0]
        index_boost = random.choices(index_boosts, index_boosts_w, k=1)[0]
        index_dist = (0, index_mid, 1, index_sharp, index_boost)
        
        unnice_mid = random.choices(unnice_mids, unnice_mids_w, k=1)[0]
        unnice_sharp = random.choices(unnice_sharps, unnice_sharps_w, k=1)[0]
        unnice_boost = random.choices(unnice_boosts, unnice_boosts_w, k=1)[0]
        unnice_dist = (-0.5, unnice_mid, 0.5, unnice_sharp, unnice_boost)
        
        updating = random.choices(updates, updates_w, k=1)[0]
        
        concepts.append(CartConcept(nice_chance, unnice_chance, mean_dist, freqhood_dist, index_dist, unnice_dist, updating))
        
    return concepts
        
def random_concepts_rand(num_concepts):

    freqhood_mids = [0, 0.35, 0.45, random.random()]
    freqhood_mids_w = reweight([1, 1, 1, ARB*3], 5)
    
    freqhood_sharps = [0, 0.5, 1, None, random.random()]
    freqhood_sharps_w = reweight([1, 1, 1, 2*RIG, ARB], 5)
    
    freqhood_boosts = [0, 0.125, random.random()**3]
    freqhood_boosts_w = reweight([2, 1, ARB], 5)
    
    square_mids = [0, 0.5, 1, random.random(), random.random()]
    square_mids_w = reweight([1, 1, 1, ARB, ARB], 5)
    
    square_sharps = [0, 0.5, 1, None, random.random()]
    square_sharps_w = reweight([1, 1, 1, 2*RIG, ARB], 5)
    
    square_boosts = [0, 0.125, random.random()**3]
    square_boosts_w = reweight([2, 1, ARB], 5)
    
    concepts = []
    
    for i in range(num_concepts):
        
        freqhood_mid = random.choices(freqhood_mids, freqhood_mids_w, k=1)[0]
        freqhood_sharp = random.choices(freqhood_sharps, freqhood_sharps_w, k=1)[0]
        freqhood_boost = random.choices(freqhood_boosts, freqhood_boosts_w, k=1)[0]
        freqhood_dist = (0, freqhood_mid, 1, freqhood_sharp, freqhood_boost)
        
        square_mid = random.choices(square_mids, square_mids_w, k=1)[0]
        square_sharp = random.choices(square_sharps, square_sharps_w, k=1)[0]
        square_boost = random.choices(square_boosts, square_boosts_w, k=1)[0]
        square_dist = (0, square_mid, 1, square_sharp, square_boost)
        
        concepts.append(RandConcept(freqhood_dist, square_dist))
        
    return concepts
                
def random_concepts_pow(num_concepts):

    logex_lownesses = [0, 0.5, 1, random.random()**2*15, random.random()**2*15]
    logex_lownesses_w = reweight([1, 1, 1, ARB, ARB], 5)
    
    logex_mids = [-1, 0, 1, random.uniform(-15, 15), random.uniform(-15, 15)]
    logex_mids_w = reweight([2, 2, 2, ARB, ARB], 5)
    
    logex_highnesses = [0, 0.5, 1, random.random()**2*15, random.random()**2*15]
    logex_highnesses_w = reweight([1, 1, 1, ARB, ARB], 5)
    
    logex_sharps = [0, 0.5, 1, None, random.random()]
    logex_sharps_w = reweight([1, 1, 1, 2*RIG, ARB], 5)
    
    logex_boosts = [0, 0.125, random.random()**3]
    logex_boosts_w = reweight([2, 1, ARB], 5)
    
    concepts = []
    
    for i in range(num_concepts):
    
        logex_lowness = random.choices(logex_lownesses, logex_lownesses_w, k=1)[0]
    
        logex_mid = random.choices(logex_mids, logex_mids_w, k=1)[0]
    
        logex_highness = random.choices(logex_highnesses, logex_highnesses_w, k=1)[0]
    
        logex_sharp = random.choices(logex_sharps, logex_sharps_w, k=1)[0]
    
        logex_boost = random.choices(logex_boosts, logex_boosts_w, k=1)[0]
        
        logex_dist = (logex_mid-logex_lowness, logex_mid, logex_mid+logex_highness, logex_sharp, logex_boost)
    
        concepts.append(PowConcept(logex_dist))
        
    return concepts

def random_concepts_power(num_concepts):

    weighing_base_chances = [0, 0.5, 1, random.random()]
    weighing_base_chances_w = reweight([1, 1, 1, ARB], 5)
    
    base_weight_mids = [1, random.random(), random.random()]
    base_weight_mids_w = reweight([2, ARB, ARB], 5)
    
    base_weight_sharps = [0, 0.5, 1, None, random.random()]
    base_weight_sharps_w = reweight([1, 1, 1, 2*RIG, ARB], 5)
    
    exp_weight_mids = [1, random.random(), random.random()]
    exp_weight_mids_w = reweight([2, ARB, ARB], 5)
    
    exp_weight_sharps = [0, 0.5, 1, None, random.random()]
    exp_weight_sharps_w = reweight([1, 1, 1, 2*RIG, ARB], 5)
    
    exp_lifthood_mids = [0, 1, random.random()]
    exp_lifthood_mids_w = reweight([1, 1, ARB], 5)
    
    exp_lifthood_sharps = [0, 0.5, 1, None, random.random()]
    exp_lifthood_sharps_w = reweight([1, 1, 1, 2*RIG, ARB], 5)
    
    concepts = []
    
    for i in range(num_concepts):
    
        weighing_base_chance = random.choices(weighing_base_chances, weighing_base_chances_w, k=1)[0]
    
        base_weight_mid = random.choices(base_weight_mids, base_weight_mids_w, k=1)[0]
        base_weight_sharp = random.choices(base_weight_sharps, base_weight_sharps_w, k=1)[0]
        base_weight_dist = (0, base_weight_mid, 1, base_weight_sharp, 0)
    
        exp_weight_mid = random.choices(exp_weight_mids, exp_weight_mids_w, k=1)[0]
        exp_weight_sharp = random.choices(exp_weight_sharps, exp_weight_sharps_w, k=1)[0]
        exp_weight_dist = (0, exp_weight_mid, 1, exp_weight_sharp, 0)
    
        exp_lifthood_mid = random.choices(exp_lifthood_mids, exp_lifthood_mids_w, k=1)[0]
        exp_lifthood_sharp = random.choices(exp_lifthood_sharps, exp_lifthood_sharps_w, k=1)[0]
        exp_lifthood_dist = (0, exp_lifthood_mid, 1, exp_lifthood_sharp, 0)
        
        concepts.append(PowerConcept(weighing_base_chance, base_weight_dist, exp_weight_dist, exp_lifthood_dist))
        
    return concepts

def random_concepts_sigmoid(num_concepts):
    
    midpoint_mids = [-1, -0.5, 0, 0.5, 1, random.random(), random.random(), random.random()]
    midpoint_mids_w = reweight([1, 1, 2, 1, 1, ARB, ARB, ARB], 5)
    
    midpoint_sharps = [0, 0.5, 1, None, random.random()]
    midpoint_sharps_w = reweight([1, 1, 1, 2*RIG, ARB], 5)
    
    squarity_mids = [0, 0.5, 1, random.random(), random.random()]
    squarity_mids_w = reweight([1, 1, 1, ARB, ARB], 5)
    
    squarity_sharps = [0, 0.5, 1, None, random.random()]
    squarity_sharps_w = reweight([2, 2, 1, 3*RIG, ARB], 5)
    
    concepts = []
    
    for i in range(num_concepts):
        
        midpoint_mid = random.choices(midpoint_mids, midpoint_mids_w, k=1)[0]
        midpoint_sharp = random.choices(midpoint_sharps, midpoint_sharps_w, k=1)[0]
        midpoint_dist = (-1, midpoint_mid, 1, midpoint_sharp, 0)
        
        squarity_mid = random.choices(squarity_mids, squarity_mids_w, k=1)[0]
        squarity_sharp = random.choices(squarity_sharps, squarity_sharps_w, k=1)[0]
        squarity_dist = (0, squarity_mid, 1, squarity_sharp, 0)
        
        concepts.append(SigmoidConcept(midpoint_dist, squarity_dist))
        
    return concepts

def random_concepts_arcfan(num_concepts):

    fconcepts = random_concepts_fan(num_concepts)
        
    return [ArcfanConcept(fconcepts[i]) for i in range(num_concepts)]

def random_concepts_sin(num_concepts):

    wconcepts = random_concepts_wave(num_concepts)
        
    return [SinConcept(wconcepts[i]) for i in range(num_concepts)]

def random_concepts_spin(num_concepts):

    wconcepts = random_concepts_wave(num_concepts)
    
    fconcepts = random_concepts_fan(num_concepts)
    
    antiwise_chances = [0, 0.5, 1, random.random()]
    antiwise_chances_w = reweight([1, 1, 1, ARB], 5)
    
    armhood_mids = [0, 0.5, random.random()]
    armhood_mids_w = reweight([1, 1, ARB], 5)
    
    armhood_sharps = [0, 0.5, 1, None, random.random()]
    armhood_sharps_w = reweight([1, 1, 1, 2*RIG, ARB], 5)
    
    armhood_boosts = [0, 0.125, random.random()**3]
    armhood_boosts_w = reweight([2, 1, ARB], 5)
    
    concepts = []
    
    for i in range(num_concepts):
    
        wconcept = wconcepts[i]
        
        fconcept = fconcepts[i]
        
        antiwise_chance = random.choices(antiwise_chances, antiwise_chances_w, k=1)[0]
        
        armhood_mid = random.choices(armhood_mids, armhood_mids_w, k=1)[0]
        armhood_sharp = random.choices(armhood_sharps, armhood_sharps_w, k=1)[0]
        armhood_boost = random.choices(armhood_boosts, armhood_boosts_w, k=1)[0]
        armhood_dist = (0, armhood_mid, 1, armhood_sharp, armhood_boost)
        
        concepts.append(SpinConcept(wconcept, fconcept, antiwise_chance, armhood_dist))
        
    return concepts

def random_concepts_minx(num_concepts):
    
    maxing_chances = [0, 0.5, 1, random.random()]
    maxing_chances_w = reweight([1, 1, 1, ARB], 3)
    
    wconcepts = random_concepts_weights(num_concepts)
    
    concepts = []
        
    for i in range(num_concepts):
    
        maxing_chance = random.choices(maxing_chances, maxing_chances_w, k=1)[0]
        
        wconcept = wconcepts[i]
        
        concepts.append(MinxConcept(maxing_chance, wconcept))
        
    return concepts
        
def random_concepts_amean(num_concepts):

    wconcepts = random_concepts_weights(num_concepts)

    concepts = []
    
    for i in range(num_concepts):
    
        wconcept = wconcepts[i]
        
        concepts.append(AmeanConcept(wconcept))
        
    return concepts

def random_concepts_it(num_concepts):

    it_endhoods = [0, random.random()**2, random.random()**2, random.random()**2, random.random()**2]
    it_endhoods_w = reweight([3, ARB, ARB, ARB, ARB], 5)
    
    it_sharps = [0, 0.5, 1, None, random.random(), random.random()]
    it_sharps_w = reweight([1, 1, 1, 2*RIG, ARB, ARB], 5)
    
    it_boosts = [0, 0.08, random.random()**3, random.random()**3]
    it_boosts_w = reweight([2, 1, ARB, ARB], 5)
    
    xlike_chances = [0, 0.5, 1, random.random(), random.random(), random.random()]
    xlike_chances_w = reweight([1, 1, 1, ARB, ARB, ARB], 5)
    
    concepts = []
    
    for i in range(num_concepts):
    
        it_dists = []
    
        for j in range(2):
    
            it_ends = random.choices(it_endhoods, it_endhoods_w, k=2)
            it_low = ity(1, SETTINGS["MAX ITERATIONS"], min(it_ends))
            it_high = ity(1, SETTINGS["MAX ITERATIONS"], max(it_ends))
            it_mid = (it_low+it_high)/2
        
            it_sharp = random.choices(it_sharps, it_sharps_w, k=1)[0]
        
            it_boost = random.choices(it_boosts, it_boosts_w, k=1)[0]
            
            it_dist = (it_low, it_mid, it_high, it_sharp, it_boost)
            
            it_dists.append(it_dist)
            
        xlike_chance = random.choices(xlike_chances, xlike_chances_w, k=1)[0]
    
        concepts.append(ItConcept(*it_dists, xlike_chance))
        
    return concepts
    
def random_concepts(element, num_concepts):

    num_concept_series = int(num_concepts * (random.random() ** 2)) + 1
    
    series_lengths = random_partition(num_concepts, num_concept_series)
    
    concepts = []
    
    for series_length in series_lengths:
    
        concepts += ALL_ELEMENTS[element].concept_generator(series_length)
        
    return concepts



### APPLY VERSION

def apply_controller(controller, artist):

    cartist = copy.deepcopy(artist)

    for isotope in cartist.values():
    
        if isotope.element in ALL_ROOTS or isotope.ID not in controller: continue
        
        control = controller[isotope.ID]
        
        isotope.options = [isotope.options[i] for i in control.submute]
        
        isotope.freedom *= control.constraint
        
    return cartist

def apply_conceiver(thicket, conceiver, artist, reseeding):
        
    isotopeIDs = [k for k in artist if k not in ALL_ROOTS]
    
    i = 0
        
    for isotopeID in isotopeIDs:
    
        for atom in thicket:
            
            if atom.isotopeID != isotopeID: continue
    
            if reseeding:
                for j in range(25):
                    if i == int(len(thicket)*j/25): reseed()
        
            concept = conceiver[atom.isotopeID][atom.element]
            
            subconceiver = ALL_ELEMENTS[atom.element].subconceiver
            if subconceiver != None: subconceiver(atom, thicket, concept)
                
            i += 1

def collect_correlatees(cor, thicket, thicket_indices):

    pairs = []
    paired_cords = []
    
    loops = 0
    
    for j in thicket_indices:
    
        atom = thicket[j]
    
        if atom.isotopeID == cor.isotopeID_0 and atom.element == cor.element_0:
            
            first = atom
            
            if cor.relation_r == 0:
                ups = cor.relation_c
                downs = cor.relation_c
                
            elif cor.relation_r < 0:
                ups = cor.relation_c
                downs = cor.relation_c - cor.relation_r
                
            elif cor.relation_r > 0:
                ups = cor.relation_c + cor.relation_r
                downs = cor.relation_c
                
            relatives = set([atom])
            
            for i in range(ups):
            
                new_relatives = set([])
                
                for rel in relatives: new_relatives |= set([thicket[ID] for ID in rel.parents])
                    
                relatives = new_relatives
            
            ###
            
            siblings = set([])
            
            for rel in relatives:
            
                sibling_IDs = set(sum([thicket[ID].children for ID in rel.parents], [])).difference(set([rel.ID]))
                siblings |= set(thicket[ID] for ID in sibling_IDs)
                
            relatives = siblings
            
            ###
                
            for i in range(downs):
            
                new_relatives = set([])
                
                for rel in relatives: new_relatives |= set([thicket[ID] for ID in rel.children])
                    
                relatives = new_relatives
                
            usable_relatives = [r for r in relatives if r not in paired_cords and r.isotopeID == cor.isotopeID_1 and r.element == cor.element_1]
                
            if len(usable_relatives) > 0:
                
                second = random.choice(usable_relatives)
                
                pairs.append((first, second))
                
                paired_cords += [first, second]
                    
        loops += 1
                    
    return pairs
    
def apply_correlator(thicket, correlator, reseeding):

    thicket_indices = list(range(len(thicket)))
    
    if reseeding: reseed()
    thicket_indices = partial_shuffle(thicket_indices, 0.125)
    
    if reseeding: reseed()
    thicket_indices = partial_shuffle(thicket_indices, 0.125)
    
    if reseeding: reseed()
    thicket_indices = partial_shuffle(thicket_indices, 0.125)

    for cor in correlator:
    
        if cor.strength == 0: continue
    
        pairs = collect_correlatees(cor, thicket, thicket_indices)
        
        firsts = [p[0].params[cor.param_index_0] for p in pairs]
        seconds = [p[1].params[cor.param_index_1] for p in pairs]
        
        if cor.strength > 0:
        
            sorted_firsts = sorted(firsts)
            sorted_seconds = sorted(seconds)
        
        else:
        
            sorted_firsts = sorted(firsts)
            sorted_seconds = sorted(seconds, reverse=True)
            
        correlated_sorted_firsts = sorted_firsts
        correlated_sorted_seconds = partial_shuffle(seconds, 1 - abs(cor.strength))
            
        shuffled_indices = partial_shuffle(range(len(pairs)), 1)
        
        correlated_firsts = [correlated_sorted_firsts[i] for i in shuffled_indices]
        correlated_seconds = [correlated_sorted_seconds[i] for i in shuffled_indices]
            
        for i in range(len(pairs)):
        
            pairs[i][0].params[cor.param_index_0] = correlated_firsts[i]
            pairs[i][1].params[cor.param_index_1] = correlated_seconds[i]

    return
    
def apply_calmer(calmer):

    g = Calm()
    
    reseed()
    
    g.hue = betta(*calmer.hue_dist)
    
    reseed()
    
    g.sat = betta(*calmer.sat_dist)
    
    reseed()
    
    g.val = betta(*calmer.val_dist)
    
    reseed()
    
    g.pal = betta(*calmer.pal_dist)
    
    reseed()
    
    g.gradience_weight, gradience_lifthood = doubetta(*calmer.gradience_dd)
    g.gradience_lift = (1 - g.gradience_weight) * gradience_lifthood
    
    reseed()
    
    g.unarm = betta(*calmer.unarm_dist)
    g.unaim = betta(*calmer.unaim_dist)
    
    return g



### GENERATE VERSION

def random_prism():

    low = SETTINGS["MIN COLORS"]
    high = SETTINGS["MAX COLORS"]
    middle = SETTINGS["MID COLORS"]

    mid = betta(low, middle, high+0.5, SETTINGS["STANDARD SHARP"], SETTINGS["STANDARD BOOST"])
    
    sharp = betta(0, 0.5, 1, SETTINGS["STANDARD SHARP"], SETTINGS["STANDARD BOOST"])
    
    return (low, mid, high+0.5, sharp, SETTINGS["STANDARD BOOST"])

def random_puddle():

    dists = []

    for lmh in ([(0, 1, SETTINGS["DIRTINESS MAX"])] * 3) + ([(0, 0, SETTINGS["STILLNESS MAX"])] * 3):

        endpoint_0 = betta(*lmh, SETTINGS["STANDARD SHARP"], SETTINGS["STANDARD BOOST"])
        endpoint_1 = betta(*lmh, SETTINGS["STANDARD SHARP"], SETTINGS["STANDARD BOOST"])
        
        low, high = sorted((endpoint_0, endpoint_1))
        mid = random.uniform(low, high)
        
        sharp = random.random()
        boost = random.random() ** SETTINGS["STANDARD EXP"]
        
        dists.append((low, mid, high, sharp, boost))

    return Puddle(*dists)

def random_pasteller():

    frac_mid = betta(0, SETTINGS["DESAT FRAC MID"], 1, SETTINGS["PASTELLER SHARP"], SETTINGS["PASTELLER BOOST"])
    
    amt_sharp = betta(0, SETTINGS["DESAT AMT SHARP MID"], 1, SETTINGS["PASTELLER SHARP"], SETTINGS["PASTELLER BOOST"])
    
    return Pasteller(frac_mid, amt_sharp)
    
def random_permuter():

    endpoint_0 = random.uniform(1, SETTINGS["NUM PALETTES MAX"]+0.999)
    endpoint_1 = random.uniform(1, SETTINGS["NUM PALETTES MAX"]+0.999)
    
    low, high = sorted((endpoint_0, endpoint_1))
    mid = random.uniform(low, high)
    
    sharp = random.random()
    boost = random.random() ** SETTINGS["STANDARD EXP"]
    
    return (low, mid, high, sharp, boost)

def random_procession():

    return betta(1, 1.8, 5, SETTINGS["STANDARD SHARP"], SETTINGS["STANDARD BOOST"])

def random_controller(artist):

    controller = {}
    
    control_chance = random.random() ** SETTINGS["STANDARD EXP"]
    
    for isotope in artist.values():
    
        if isotope.element in ALL_ROOTS: continue
        if random.random() > control_chance: continue
        
        og_indices = range(len(isotope.options))
        num_keeping = int((1 - (random.random() ** SETTINGS["LARGE EXP"])) * len(og_indices) + 0.9999)
        indices = random.choices(og_indices, k=num_keeping)
        submute = partial_shuffle(indices, random.random() ** SETTINGS["LARGE EXP"])
        
        constraint = 1 - (random.random() ** SETTINGS["STANDARD EXP"])
        
        controller[isotope.ID] = Control(submute, constraint)
        
    return controller

def random_conceiver(artist, initial_conceiver):

    global ARB
    global RIG
    
    exmax = SETTINGS["ARBRIG EXPONENT MAX"]
    
    arbex_mid = ity(-exmax, exmax, SETTINGS["ARBITRARINESS MIDHOOD"])
    arbex = betta(-exmax, arbex_mid, exmax, SETTINGS["ARBITRARINESS SHARP"], SETTINGS["STANDARD BOOST"])
    ARB = 2 ** arbex
    
    rigex_mid = ity(-exmax, exmax, SETTINGS["RIGIDITY MIDHOOD"])
    rigex = betta(-exmax, rigex_mid, exmax, SETTINGS["RIGIDITY SHARP"], SETTINGS["STANDARD BOOST"])
    RIG = 2 ** rigex
    

    conceptless_ftks = {}
    
    for k in artist:
    
        if artist[k].element in ALL_ROOTS: continue
    
        element = artist[k].element
        
        if k in initial_conceiver:
            initial_elements = conceiver[k].keys()
        else:
            initial_elements = []
        
        if element not in initial_elements:
        
            if element in conceptless_ftks: conceptless_ftks[element].append(k)
            else: conceptless_ftks[element] = [k]
            
    conceiver = copy.deepcopy(initial_conceiver)
            
    for element in conceptless_ftks:
    
        num_isotopes = len(conceptless_ftks[element])
    
        concepts = random_concepts(element, num_isotopes)
        
        for i in range(num_isotopes):
        
            k = conceptless_ftks[element][i]
        
            if k in conceiver: conceiver[k][element] = concepts[i]
            else: conceiver[k] = {element: concepts[i]}
            
    return conceiver

def random_correlator(artist):
    
    correlatable_kfps = []
    
    for isotope in artist.values():
    
        if isotope.element in ALL_ROOTS: continue
    
        correlatable_kfps += [(isotope.ID, isotope.element, p) for p in ALL_ELEMENTS[isotope.element].correlatable_params]
            
    max_num_correlations = int(len(correlatable_kfps)/2)
            
    num_correlations = int(betta(0, SETTINGS["CORRELATION RATE MID"], 1, SETTINGS["CORRELATION RATE SHARP"], 0.1)*max_num_correlations)
    
    correlating_kfps = random.sample(correlatable_kfps, k=num_correlations*2)
    
    correlations = []
    
    for i in range(num_correlations):
    
        cor = Correlation()
        
        cor.isotopeID_0 = correlating_kfps[2*i][0]
        cor.element_0 = correlating_kfps[2*i][1]
        cor.param_index_0 = correlating_kfps[2*i][2]
        
        cor.isotopeID_1 = correlating_kfps[2*i+1][0]
        cor.element_1 = correlating_kfps[2*i+1][1]
        cor.param_index_1 = correlating_kfps[2*i+1][2]
        
        cor.relation_c = int(betta(-1, -1, 5, 0.75, 0.05))
        cor.relation_r = int(betta(0, 0, 6, 0.75, 0.05)) * random.choice((1, -1))
        
        cor.strength = random.uniform(-1, 1)
        
        correlations.append(cor)

    return correlations
        
def random_calmer():
    
    hue_dist = (0, 0, 1, betta(0, 1, 1, SETTINGS["STANDARD SHARP"], SETTINGS["STANDARD BOOST"]), 0)
    sat_dist = (0, 0, 1, random.random() ** SETTINGS["SAT CALMER SHARP EXPLET"], SETTINGS["STANDARD BOOST"])
    val_dist = (0, 0, 1, random.random() ** SETTINGS["VAL CALMER SHARP EXPLET"], SETTINGS["STANDARD BOOST"])
    
    pal_dist = (0, random.random(), 2, random.random(), random.random()**SETTINGS["STANDARD EXP"])
    
    gradience_weight_dist = (0, random.random(), 1, random.random(), random.random()**SETTINGS["STANDARD EXP"])
    gradience_lifthood_dist = (0, random.random(), 1, random.random(), random.random()**SETTINGS["STANDARD EXP"])
    
    gradience_wb_chance = random.random()**SETTINGS["LARGE EXP"]
    gradience_wt_chance = random.random()**SETTINGS["LARGE EXP"]
    gradience_lb_chance = random.random()**SETTINGS["LARGE EXP"]
    gradience_lt_chance = random.random()**SETTINGS["LARGE EXP"]
    
    gradience_dd = (gradience_weight_dist, gradience_wb_chance, gradience_wt_chance, gradience_lifthood_dist, gradience_lb_chance, gradience_lt_chance)
    
    a = random.random()
    if a <= 0.1: unarm_mid = 0
    elif a <= 0.2: unarm_mid = 1
    else: unarm_mid = random.random()
    unarm_dist = (0, unarm_mid, 1, random.random(), random.random()**SETTINGS["STANDARD EXP"])
    
    unaim_mid = betta(1, SETTINGS["UNAIM MID"], SETTINGS["UNAIM MAX"], SETTINGS["STANDARD SHARP"], SETTINGS["STANDARD BOOST"])
    unaim_dist = (1, unaim_mid, SETTINGS["UNAIM MAX"], random.random(), random.random()**SETTINGS["STANDARD EXP"])
    
    g = Calmer(hue_dist, sat_dist, val_dist, pal_dist, gradience_dd, unarm_dist, unaim_dist)
    
    return g
    


### GENERATE THICKET

def validate_thicket(thicket, artist):
            
    print("validating thicket...")

    for atom in thicket:
    
        parent_tope = artist[atom.isotopeID]
        
        
        child_list = [thicket[child].isotopeID for child in atom.children]
        parent_list = parent_tope.options+parent_tope.defaults
        if child_list in parent_list or child_list == parent_list:
            # print("good")
            pass
        else:
            print("bad: ")
            print("children: "+str(atom.children))
            print("options: "+str(parent_tope.options))
            print("defaults: "+str(parent_tope.defaults))
            exit(0)

def dagsort(thicket, roots):

    original_thicket = copy.deepcopy(thicket)

    tiers = []
    
    tiered_IDs = set([])
    
    while True:
    
        tier = []
        
        for atom in thicket:
        
            if (atom.ID not in tiered_IDs) and (len(set(atom.children).difference(tiered_IDs)) == 0): tier.append(atom.ID)
                
        if len(tier) == 0: break
                
        tiers.append(tier)
        
        tiered_IDs |= set(tier)
                
    index_map = {}
    
    for tier in tiers:
        for ID in tier: index_map[ID] = len(index_map)
            
    for atom in thicket:
    
        try: atom.ID = index_map[atom.ID]
        except: return None, None
        
        atom.children = list(map(index_map.get, atom.children))
        
        atom.parents = list(map(index_map.get, atom.parents))
        
        del atom.descendants
        del atom.ancestors
        del atom.pseudoch
        del atom.pseudopa
        del atom.pseudode
        del atom.pseudoan
        
    thicket.sort(key = lambda x: x.ID)
    
    roots = list(map(index_map.get, roots))
        
    return thicket, roots
  
def prune_roots(thicket):

    roots = [None] * len(ALL_ROOTS)

    for atom in thicket:
    
        if atom.element in ALL_ROOTS:
        
            roots[ALL_ROOTS.index(atom.element)] = atom.children[0]
            thicket[atom.children[0]].parents.remove(atom.ID)
    
    renumber = {i : i for i in range(len(thicket))}

    thicket[:] = [atom for atom in thicket if atom.element not in ALL_ROOTS]
    
    for i in range(len(thicket)):
    
        atom = thicket[i]
        renumber[atom.ID] = i
        atom.ID = i
        
    for atom in thicket:
    
        atom.children = list(map(renumber.get, atom.children))
        atom.parents = list(map(renumber.get, atom.parents))
        atom.descendants = set(map(renumber.get, atom.descendants))
        atom.ancestors = set(map(renumber.get, atom.ancestors))
        
    roots = list(map(renumber.get, roots))
            
    return roots
    
def find_graft(sought_tope, thicket, bud):

    possible_grafts = []

    for atom in thicket:
    
        if atom.isotopeID != sought_tope: continue
        
        if bud.ID in atom.descendants|atom.pseudode: continue
        
        possible_grafts.append(atom.ID)
        
    if len(possible_grafts) == 0: return None
    else: return random.choice(possible_grafts)
    
def new_atom(artist, tope, thicket, bud):
                
    atom = Atom(artist[tope].element)
    atom.isotopeID = tope
    atom.ID = len(thicket)
    
    if ALL_ELEMENTS[atom.element].fex == -1:
        atom.fex = random.randrange(artist[tope].infexes[0], artist[tope].infexes[1]+1)
    else:
        atom.fex = ALL_ELEMENTS[atom.element].fex
        
    atom.children = []
    atom.parents = []
    atom.descendants = set([atom.ID])
    atom.ancestors = set([atom.ID])
    
    atom.pseudoch, atom.pseudopa = [], []
    atom.pseudode, atom.pseudoan = set([]), set([])
    
    return atom
    
def complete_andescendants(parent, child, thicket):

    # takes parent & child atom and augments all relationship fields
    # of each to reflect that status

    ## parent ancestors
    for ancestor_ID in parent.ancestors|parent.pseudoan:
        thicket[ancestor_ID].pseudode |= (child.descendants|child.pseudode)
    
    # parent
    parent.pseudoch.append(child.ID)
    parent.pseudode |= (child.descendants|child.pseudode)
    
    # child
    child.pseudopa.append(parent.ID)
    child.pseudoan |= (parent.ancestors|parent.pseudoan)
    
    # child descendants
    for descendant_ID in child.descendants|child.pseudode:
        thicket[descendant_ID].pseudoan |= (parent.ancestors|parent.pseudoan)

def integrate_pseudo(thicket, pseudothicket):
                    
    thicket += pseudothicket
    
    for atom in thicket:
    
        if atom.pseudoch:
            atom.children += atom.pseudoch
            atom.pseudoch = []
            
        if atom.pseudopa:
            atom.parents += atom.pseudopa
            atom.pseudopa = []
            
        if atom.pseudoan:
            atom.ancestors |= atom.pseudoan
            atom.pseudoan = set([])
            
        if atom.pseudode:
            atom.descendants |= atom.pseudode
            atom.pseudode = set([])

def clear_pseudo(thicket):

        for atom in thicket:
        
            atom.pseudoch = []
            atom.pseudopa = []
            atom.pseudoan = set([])
            atom.pseudode = set([])

def grow_bud(bud, buds, thicket, artist, tope_counts):

    tope = artist[bud.isotopeID]

    options = partial_shuffle(tope.options, tope.freedom)
    defaults = tope.defaults
    
    bud_grown = False
    
    for option in options+defaults:
        
        pseudothicket = []
        pseudotc = {}
        pseudobuds = []
        
        option_succeeded = True
        
        for oc in option:
            
            if (tope_counts|pseudotc)[oc] >= artist[oc].max_count: graftings = [True]
            elif random.random() <= 0: graftings = [True, False]
            else: graftings = [False]
            
            oc_added = False
            
            for grafting in graftings:
            
                if grafting:
                
                    graft = find_graft(oc, thicket+pseudothicket, bud)
                    
                    if graft == None:
                        continue
                    else:
                        complete_andescendants(bud, (thicket+pseudothicket)[graft], thicket+pseudothicket)
                        oc_added = True
                        break
                
                else:                    
                
                    atom = new_atom(artist, oc, thicket+pseudothicket, bud)
                    
                    pseudothicket.append(atom)
                    
                    if ALL_ELEMENTS[atom.element].fex != 0: pseudobuds.append(atom.ID)
                    
                    complete_andescendants(bud, atom, thicket+pseudothicket)
                    
                    if oc in pseudotc: pseudotc[oc] += 1
                    else: pseudotc[oc] = 1
                    
                    oc_added  = True
                    break
                    
            if not oc_added:
                option_succeeded = False
                break
                
        if option_succeeded:
        
            integrate_pseudo(thicket, pseudothicket)
            
            buds += pseudobuds
            
            for tope, count in pseudotc.items():
                if tope in tope_counts: tope_counts[tope] += count
                else: tope_counts[tope] = count
        
            bud_grown = True
            
            break
            
        else:
        
            clear_pseudo(thicket)
            
        if bud_grown:
            break
                
    if bud_grown:

        buds.remove(bud.ID)
        
    else:
        print("No option or default succeeded.")
        exit(0)

def plant_roots(buds, thicket, artist, reseeding):

    if reseeding:
        reseed()
        shuffled_roots = partial_shuffle(ALL_ROOTS, 0.125)
        reseed()
        shuffled_roots = partial_shuffle(shuffled_roots, 0.125)
        reseed()
        shuffled_roots = partial_shuffle(shuffled_roots, 0.125)

    for root_name in ALL_ROOTS:
    
        root_tope = next(tope for tope in artist.values() if tope.element == root_name)
        
        root = Atom()
        
        root.element = root_name
    
        root.isotopeID = root_tope.ID
        
        root.ID = len(thicket)
        
        root.fex = 1
                
        root.children, root.parents = [], []
        root.descendants, root.ancestors = set([root.ID]), set([root.ID])
        
        root.pseudoch, root.pseudopa = [], []
        root.pseudode, root.pseudoan = set([]), set([])
        
        thicket.append(root)
        
        buds.append(root.ID)

def choose_bud(buds, thicket):

    least_tope = min(thicket[bud_ID].isotopeID for bud_ID in buds)
    
    bud_contenders = [thicket[bud_ID] for bud_ID in buds if thicket[bud_ID].isotopeID == least_tope]
    
    return random.choice(bud_contenders)

def grow_thicket(artist, reseeding, overgrowth_allowed):

    thicket, roots, buds = [], [], []
    
    tope_counts = {tope : 0 for tope in artist}
    
    num_loops = 0
    
    plant_roots(buds, thicket, artist, reseeding)
    
    if reseeding: initial_seed_ptyndex = SEEDER.seed_ptyndex
        
    while len(buds) > 0:
    
        if reseeding:
            if num_loops % 10 == 0 and num_loops < 240: reseed()
    
        bud = choose_bud(buds, thicket)
        
        grow_bud(bud, buds, thicket, artist, tope_counts)
        
        num_loops += 1
        
        if (not overgrowth_allowed) and (num_loops > SETTINGS["OVERGROWTH CUTOFF"]): return None, None
        
    if reseeding: SEEDER.seed_ptyndex = initial_seed_ptyndex + 24
        
    roots = prune_roots(thicket)
        
    thicket, roots = dagsort(thicket, roots)
            
    return thicket, roots

def grow_trim_thicket(artist, version):

    if SETTINGS["MANUAL THICKET"]: return manual_thicket()
    
    thicket, roots = grow_thicket(artist, True, True)
    
    apply_conceiver(thicket, version.conceiver, artist, True)
    
    apply_correlator(thicket, version.correlator, True)
    
    return thicket, roots



### ANALYZE ARTIST

def flatten_options(options):
        
    flattened = []
    
    for option in options:
        if isinstance(option, int): flattened.append(option)
        else: flattened += option
        
    return set(flattened)

def find_roots(artist):

    return set([ID for ID in artist if artist[ID].element in ALL_ROOTS])

def find_enders(artist):

    enders = set([isotope.ID for isotope in artist.values() if isotope.element in ["X", "Y"]])
    
    while True:
    
        new_enders = set([])
        
        for isotope in artist.values():
        
            if isotope.element in ALL_ROOTS: continue
        
            if isotope.ID in enders: continue
        
            options = flatten_options(isotope.options)
            
            if isotope.ID not in options and len(options.difference(enders)) == 0:
            
                new_enders.add(isotope.ID)
            
        if len(new_enders) == 0: break

        enders |= new_enders
    
    return enders
            
def validate_options(options, artist, ID, root_IDs, enders, is_defaults):

    option_IDs = []
    
    for o in options:
        
        if isinstance(o, list):
        
            if ID in root_IDs and len(o) > 1: return "Multi-element list illegally listed as an option for a root."
        
            for p in o:
            
                if isinstance(p, int):
                    option_IDs.append(p)
                    
                else: return "Unexpected type "+str(type(p))+" within list in options."
                
                if is_defaults and p not in enders: return "Default "+str(p)+" not in enders."
                
        else: return "Unexpected type "+str(type(o))+" in options."
            
    for option_ID in option_IDs:
    
            if option_ID in root_IDs: return "Root illegally listed as a subtype option."
            
            if option_ID not in artist: return "Option ID ("+str(option_ID)+") not in artist."
            
    return None

def validate_isotope_basic(key_ID, artist, root_IDs):
    
    if not isinstance(key_ID, int): return "Illegal type ("+str(type(key_ID))+") used as key ID."
        
    if key_ID < 0: return "Negative integer ("+str(type(key_ID))+") used as key ID."

    isotope = artist[key_ID]
    ID = isotope.ID
    element = isotope.element
    options = isotope.options
    
    if ID != key_ID: return "Isotope ID "+str(ID)+" does not match key ID ("+str(key_ID)+")."
    if not isinstance(element, str): return "Element should be provided as str but instead is "+str(type(element))+"."
            
    if element not in ALL_ROOTS and element not in ALL_ELEMENTS: return "Unexpected element with ID "+str(ID)+": "+str(element)+"."
            
    if not isinstance(options, list): return "Isotope ID "+str(ID)+": "+"Options should be provided as list but instead are "+str(type(options))+"."
            
    is_root = ID in root_IDs
    has_fertile_element = (not is_root) and ALL_ELEMENTS[element].fex != 0
    if (is_root or has_fertile_element) and len(options) == 0: return "No options provided for isotope "+str(isotope.ID)+"."
    
    return None

def validate_artist(artist):

    ## length
    if len(artist) < len(ALL_ROOTS)+1: return "Scheme too short (length: "+str(len(artist))+")."
        
    ## has X or Y
    found_cart = any((isotope.element in ["X", "Y"]) for isotope in artist.values())
    if not found_cart: return "Scheme has no cart."
    
    root_IDs = find_roots(artist)
        
    ## validate basics for each isotope
    for key_ID in artist:
    
        result = validate_isotope_basic(key_ID, artist, root_IDs)
        if result != None: return result
        
    enders = find_enders(artist)
            
    ## validate options & defaults for each isotope
    for isotope in artist.values():
    
        result = validate_options(isotope.options, artist, isotope.ID, root_IDs, enders, False)
        if result != None: return result
        
        result = validate_options(isotope.defaults, artist, isotope.ID, root_IDs, enders, True)
        
    roots = set([artist[ID].element for ID in root_IDs])
    
    ## has all necessary roots
    missing_roots = set(ALL_ROOTS).difference(roots)
    if len(missing_roots) != 0: return "Scheme is missing roots: "+", ".join(list(missing_roots))
        
    return None



### GENERATE ARTIST
    
def get_accessibles(known_accessibles, artist):
    
    while True:
    
        new_accessibles = set([])
        
        for acc in known_accessibles:
            new_accessibles |= set(sum(artist[acc].options, []))
            
        new_accessibles = new_accessibles.difference(known_accessibles)
            
        if len(new_accessibles) == 0: break
        
        known_accessibles |= new_accessibles
        
    return known_accessibles

def cull_unused_isotopes(artist, roots):

    used = sorted(list(get_accessibles(roots, artist)))
    
    ID_map = {used[i]: i for i in range(len(used))}
    
    new_artist = {}
    
    for ID in used:
    
        isotope = artist[ID]
        
        isotope.ID = ID_map[isotope.ID]
        
        isotope.options = [list(map(ID_map.get, opt)) for opt in isotope.options]
        isotope.defaults = [list(map(ID_map.get, d)) for d in isotope.defaults]
    
        new_artist[ID_map[ID]] = isotope
        
    return new_artist

def add_defaults(artist, enders, isotopeIDs, option_reqsets):

    for isotopeID in isotopeIDs:
    
        isotope = artist[isotopeID]
        
        if isotopeID in enders:
            isotope.defaults = copy.deepcopy(isotope.options)
            continue
    
        if isotope.element in ALL_ROOTS: fexes = [1]
        elif ALL_ELEMENTS[isotope.element].fex == -1: fexes = list(range(isotope.infexes[0], isotope.infexes[1]+1))
        else: fexes = [ALL_ELEMENTS[isotope.element].fex]
        default_length = random.choice(fexes)
        
        req_part = [random.choice(list(enders.intersection(reqset))) for reqset in option_reqsets]
        unreq_part = random.choices(list(enders), k=default_length-len(req_part))
        
        default = req_part + unreq_part
        
        isotope.defaults = [default]
        
        isotope.options.insert(random.randrange(len(isotope.options)+1), default)

def add_options(artist, usables, isotopeIDs, option_reqsets, ocount_dist, ohood_dist):

    for isotopeID in isotopeIDs:
    
        isotope = artist[isotopeID]
    
        num_options  = int(betta(*ocount_dist))
    
        if isotope.element in ALL_ROOTS: fexes = [1]
        elif ALL_ELEMENTS[isotope.element].fex == -1: fexes = list(range(isotope.infexes[0], isotope.infexes[1]+1))
        else: fexes = [ALL_ELEMENTS[isotope.element].fex]
        option_length_max = max(fexes)
        
        options = []
        
        for _ in range(num_options):
        
            option_length = max(int(ity(1, option_length_max+1, betta(*ohood_dist))), option_length_max)
            
            req_part = [random.choice(list(reqset)) for reqset in option_reqsets]
            
            unreq_part = random.choices(list(usables), k=option_length-len(req_part))
            
            option = req_part + unreq_part
            
            random.shuffle(option)
            
            options.append(option)
            
        artist[isotopeID].options = options
    
def add_isotopes(artist, num_to_add, element_weight_pairs, infex_dist, loose_dist, limit_dist):

    element_list, element_weights = tuple(zip(*element_weight_pairs))

    isotopeIDs_to_add = list(range(len(artist), len(artist)+num_to_add))

    for isotopeID in isotopeIDs_to_add:

        element = random.choices(element_list, element_weights, k=1)[0]
        
        if ALL_ELEMENTS[element].fex == -1: infexes = tuple(sorted([int(betta(*infex_dist)), int(betta(*infex_dist))]))
        else: infexes = ()
    
        artist[isotopeID] = Isotope(isotopeID, element, infexes, betta(*loose_dist), int(betta(*limit_dist)), [], [])
        
    return set(isotopeIDs_to_add)
      
def building_foundation(artist, loose_dist):
    
    for i in range(len(ALL_ROOTS)):
        artist[i] = Isotope(i, ALL_ROOTS[i], (), betta(*loose_dist), 1, None, None)
    
    return set(range(len(ALL_ROOTS)))

def building_parameters():
    
    element_list = list(ALL_ELEMENTS.keys())
    element_weights = [SETTINGS[element+" WEIGHT"] for element in element_list]
    element_weights = reweight(element_weights, SETTINGS["ELEMENT REWEIGHT DEPTH"], SETTINGS["ELEMENT REWEIGHT BREADTH"], SETTINGS["ELEMENT REWEIGHT SHARP"])
    element_weight_pairs = list(zip(element_list, element_weights))
    
    infex_dist = (2, 2, SETTINGS["INFEX MAX"], SETTINGS["INFEX SHARP"], SETTINGS["STANDARD BOOST"])
    
    loose_dist = (0, SETTINGS["LOOSENESS MID"], 1, SETTINGS["LOOSENESS SHARP"], SETTINGS["STANDARD BOOST"])
    
    limit_dist = (1, 1, SETTINGS["ISOTOPE REPLICATE LIMIT MAX"], SETTINGS["ISOTOPE REPLICATE LIMIT SHARP"], SETTINGS["STANDARD BOOST"])
    
    perfect_ocount_sharp = max(1 - 6.55/(SETTINGS["ARTIST SIZE MID"]**0.67), 0)
    ocount_sharp_ex = 2**((0.5-SETTINGS["OPTION COUNT SHARPHOOD"])*4)
    ocount_sharp = perfect_ocount_sharp ** ocount_sharp_ex
    
    ocount_dist = (1, 1, SETTINGS["OPTION COUNT MAX"], ocount_sharp, SETTINGS["STANDARD BOOST"])
    
    ohood_dist = (0, SETTINGS["OPTION LENGTHHOOD MID"], 1, SETTINGS["OPTION LENGTHHOOD SHARP"], SETTINGS["STANDARD BOOST"])
    
    return element_weight_pairs, infex_dist, loose_dist, limit_dist, ocount_dist, ohood_dist
    
def building_counts():

    mid = SETTINGS["ARTIST SIZE MID"]
    low = mid / (1 + SETTINGS["ARTIST SIZE SPREAD"])
    high = mid * (1 + SETTINGS["ARTIST SIZE SPREAD"])

    num_risotopes = max(int(betta(low, mid, high, SETTINGS["ARTIST SIZE SHARP"], SETTINGS["STANDARD BOOST"])), 3)

    kk_weights = [SETTINGS["KK"+kk+" WEIGHT"] for kk in ["X", "Y", "DEFX", "DEFY", "XY", "DEFXY", "INDEF"]]
    
    kk_weights = reweight(kk_weights, SETTINGS["KK REWEIGHT DEPTH"], SETTINGS["KK REWEIGHT BREADTH"], SETTINGS["KK REWEIGHT SHARP"])
    
    norm = sum(kk_weights)/(num_risotopes-1)
    
    kk_counts = list(map(lambda x: round(x/norm), kk_weights))
    
    num_xs = max(kk_counts[0], 1)
    num_ys = max(kk_counts[1], 1)
    num_defxs = kk_counts[2]
    num_defys = kk_counts[3]
    num_xys = max(kk_counts[4]-1, 0)
    num_defxys = kk_counts[5]
    num_indefs = kk_counts[6]
    
    return num_xs, num_ys, num_defxs, num_defys, num_xys, num_defxys, num_indefs
        
def random_artist():


    ### setup ###
    
    num_xs, num_ys, num_defxs, num_defys, num_xys, num_defxys, num_indefs = building_counts()

    fw_pairs, infex_dist, loose_dist, limit_dist, ocount_dist, ohood_dist = building_parameters()

    artist = {}
    
    
    ### generate isotopes ###

    roots = building_foundation(artist, loose_dist)
        
    xs = add_isotopes(artist, num_xs, filter(lambda x: x[0] == "X", fw_pairs), infex_dist, loose_dist, limit_dist)
    ys = add_isotopes(artist, num_ys, filter(lambda x: x[0] == "Y", fw_pairs), infex_dist, loose_dist, limit_dist)
    defxs = add_isotopes(artist, num_defxs, filter(lambda x: ALL_ELEMENTS[x[0]].fex != 0, fw_pairs), infex_dist, loose_dist, limit_dist)
    defys = add_isotopes(artist, num_defys, filter(lambda x: ALL_ELEMENTS[x[0]].fex != 0, fw_pairs), infex_dist, loose_dist, limit_dist)
    xyes = add_isotopes(artist, 1, filter(lambda x: ALL_ELEMENTS[x[0]].fex not in (0, 1), fw_pairs), infex_dist, loose_dist, limit_dist)
    xys = add_isotopes(artist, num_xys, filter(lambda x: ALL_ELEMENTS[x[0]].fex not in (0, 1), fw_pairs), infex_dist, loose_dist, limit_dist)
    defxys = add_isotopes(artist, num_defxys, filter(lambda x: ALL_ELEMENTS[x[0]].fex != 0, fw_pairs), infex_dist, loose_dist, limit_dist)
    indefs = add_isotopes(artist, num_indefs, filter(lambda x: ALL_ELEMENTS[x[0]].fex != 0, fw_pairs), infex_dist, loose_dist, limit_dist)
    
    
    ###
    
    usables = set(artist).difference(roots)
    
    ### isotopeIDs of each kk, and the isotopeID pools their children must (at least partly) draw from
    ko_pairs = []
    ko_pairs.append([defxs, [xs|defxs]])
    ko_pairs.append([defys, [ys|defys]])
    ko_pairs.append([xys, [xs|defxs, ys|defys]])
    ko_pairs.append([defxys, [xyes|xys|defxys]])
    ko_pairs.append([indefs, []])
    ko_pairs.append([roots, [xyes|xys|defxys]])
    
    for kop in ko_pairs: add_options(artist, usables, *kop, ocount_dist, ohood_dist)
    add_options(artist, xs|ys, xyes, [xs, ys], ocount_dist, ohood_dist)
    
    enders = find_enders(artist)
    
    for kop in ko_pairs: add_defaults(artist, enders, *kop)
    add_defaults(artist, enders, xyes, [xs, ys])
    
    artist = cull_unused_isotopes(artist, roots)
                
    for isotope in artist.values():
        if len(isotope.options) < 2: isotope.freedom = 0
        
    validate_artist(artist)
    
    return artist



### MANUAL

def manual_artist():

    artist = {}
    
    for i in range(len(ALL_ROOTS)):
        artist[i] = Isotope(i, ALL_ROOTS[i], (), 0, 1, [[6]], [])
    
    artist[6] = Isotope(6, "X", (), 0, 1, [], [])
    
    return artist

def manual_thicket():

    thicket = []

    ### X
    atom = Atom("X")
    atom.params = [0.5]
    atom.children = []
    atom.ID = 0
    thicket.append(atom)

    ### Y
    atom = Atom("Y")
    atom.params = [0.5]
    atom.children = []
    atom.ID = 1
    thicket.append(atom)

    ### X * 4
    atom = Atom("AMEAN")
    atom.params = [[1,1,1,1,1,1,1]]
    atom.children = [0, 0, 0, 0,0,0,0]
    atom.ID = 2
    thicket.append(atom)

    ### Y * 4
    atom = Atom("AMEAN")
    atom.params = [[1,1,1,1]]
    atom.children = [1, 1, 1, 1]
    atom.ID = 3
    thicket.append(atom)
    
    ### x^2
    atom = Atom("POW")
    atom.params = [2]
    atom.children = [0]
    atom.ID = 4
    thicket.append(atom)
    
    ### y^2
    atom = Atom("POW")
    atom.params = [2]
    atom.children = [1]
    atom.ID = 5
    thicket.append(atom)
    
    ### -y^2
    atom = Atom("INV")
    atom.params = []
    atom.children = [5]
    atom.ID = 6
    thicket.append(atom)
    
    ### x*y
    atom = Atom("GMEAN")
    atom.params = [[1, 1]]
    atom.children = [0, 1]
    atom.ID = 7
    thicket.append(atom)
    
    ### 2*x*y
    atom = Atom("AMEAN")
    atom.params = [[1, 1]]
    atom.children = [7, 7]
    atom.ID = 8
    thicket.append(atom)
    
    ### iterator x
    atom = Atom("AMEAN")
    atom.params = [[1, 1, 1]]
    atom.children = [4, 6, 2]
    atom.ID = 9
    thicket.append(atom)
    
    ### iterator y
    atom = Atom("AMEAN")
    atom.params = [[1, 1]]
    atom.children = [8, 3]
    atom.ID = 10
    thicket.append(atom)
    
    ### iterator right
    atom = Atom("IT")
    atom.params = [30, 30, True]
    atom.children = [2, 3, 9, 10, 0]
    atom.ID = 11
    thicket.append(atom)
    
    ### iterator left
    atom = Atom("IT")
    atom.params = [30, 30, False]
    atom.children = [2, 3, 9, 10, 0]
    atom.ID = 12
    thicket.append(atom)
    
    ### right squared
    atom = Atom("POW")
    atom.params = [2]
    atom.children = [11]
    atom.ID = 13
    thicket.append(atom)
    
    ### left squared
    atom = Atom("POW")
    atom.params = [2]
    atom.children = [12]
    atom.ID = 14
    thicket.append(atom)
    
    ### norm
    atom = Atom("AMEAN")
    atom.params = [[1, 1]]
    atom.children = [13, 14]
    atom.ID = 15
    thicket.append(atom)
        
    roots = [15] * len(ALL_ROOTS)
        
    return thicket, roots
