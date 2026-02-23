import random
import math # sin, cos, tan, atan, pi, exp, log
import copy # copy.copy
import json # define encoder for saving/loading general objects
import re # validate artist/version names
import platform # determine OS to determine correct subprocess command to display image, and correct way to detect interrupt
import sys

ON_WINDOWS = platform.system() in ["win32", "Windows"]

# import msvcrt or curses to detect key presses for render/search interrupt
if ON_WINDOWS:
    import msvcrt
    CURSE = None
else:
    import curses
    CURSE = curses.initscr()
    CURSE.nodelay(True)



### META
        
class SimpleEncoder(json.JSONEncoder):
    def default(self, o):
        return {"class" : type(o).__name__} | o.__dict__
        
class Automator():
 
    def __init__(self):
    
        self.on = False
        self.code = None
        self.index = 0
        self.commands = []
        self.save_s = []
        self.save_n = []



### MULTIPTYCH MGMT

class Seeder:
    
    def __init__(self):
    
        self.seed_ptychs = None
        self.painting_index = None
        self.seed_ptyndex = None



### PTYCH
        
class Glow():

    def __init__(self):

        self.is_circular = None
        self.height = None
        self.color = None

class Border():

    def __init__(self):
    
        self.height = None
        self.period = None
        self.amplitude = None
        self.exponent = None

class Palette():

    def __init__(self, colors=None, quicks=None, base_colors=None):
    
        self.colors = colors
        self.quicks = quicks
        self.base_colors = base_colors
                                     
class Whirl():

    def __init__(self, palettes=None, thicket=None, roots=None, calm=None):
    
        self.palettes = palettes
        self.thicket = thicket
        self.roots = roots
        self.calm = calm

class Painting():

    def __init__(self, whirl = None, glow = None, borders = None):
    
        self.whirl = whirl
        self.glow = glow
        self.borders = borders
        
class Ptych():

    def __init__(self, width = None, height = None, layout = None):
    
        self.canvas_width = width
        self.canvas_height = height
        self.layout = layout
        self.paintings = []
        self.info = {}
        


### THICKET

class Ordict():

    def __init__(self, o, d):
    
        self.o = o
        self.d = d
        
class Thicketmaster():

    def __init__(self, thicket = None, master = None): 

        self.thicket = thicket
        self.master = master

class Element():

    def __init__(self, name=None, fex=None, time=None, concept_generator=None, subconceiver=None, correlatable_params=None, func=None, texter=None, cxer=None):
    
        self.name = name
        self.fex = fex
        self.time = time
        self.concept_generator = concept_generator
        self.subconceiver = subconceiver
        self.correlatable_params = correlatable_params
        
        self.func = func
        self.texter = texter
        self.cxer = cxer

class Atom():

    def __init__(self, element = None):
    
        self.element = element
        if element != None: self.func = ALL_ELEMENTS[element].func
        
        self.isotopeID = None
        self.ID = None
        self.fex = None
        self.children = None
        self.parents = None
        
        self.params = None
        self.cx = None
        
    def __str__(self):
    
        parts = []
        parts.append("ID: "+str(self.ID))
        parts.append("isotope: "+str(self.isotopeID))
        parts.append("element: "+str(self.element))
        parts.append("children: "+str(self.children))
        parts.append("parents: "+str(self.parents))
        parts.append("params: "+str(self.params))
        parts.append("cx: "+str(self.cx))
    
        return "; ".join(parts)



### ARTIST & VERSION MISC.

class Isotope():

    def __init__(self, ID=None, element=None, infexes=None, freedom=None, max_count=None, options=None, defaults=None):
        
        self.ID = ID
        self.element = element
        self.infexes = infexes
        self.freedom = freedom
        self.max_count = max_count
        self.options = options
        self.defaults = defaults
        
    def __repr__(self):
    
        parts = []
        parts.append("ID: "+str(self.ID))
        parts.append("element: "+str(self.element))
        parts.append("infexes: "+str(self.infexes))
        parts.append("freedom: "+str(self.freedom))
        parts.append("max_count: "+str(self.max_count))
        parts.append("options: "+str(self.options))
        parts.append("defaults: "+str(self.defaults))
    
        return "\n".join(parts)
 
class Pasteller():

    def __init__(self, frac_mid=None, amt_sharp=None):
    
        self.frac_mid = frac_mid
        self.amt_sharp = amt_sharp

class Puddle():

    def __init__(self, hue_dirtiness_dist=None, sat_dirtiness_dist=None, val_dirtiness_dist=None, hue_stillness_dist=None, sat_stillness_dist=None, val_stillness_dist=None):
    
        self.hue_dirtiness_dist = hue_dirtiness_dist
        self.sat_dirtiness_dist = sat_dirtiness_dist
        self.val_dirtiness_dist = val_dirtiness_dist
        
        self.hue_stillness_dist = hue_stillness_dist
        self.sat_stillness_dist = sat_stillness_dist
        self.val_stillness_dist = val_stillness_dist
    
class Control():

    def __init__(self, submute=None, constraint=None):
    
        self.submute = submute # list of indices
        self.constraint = constraint # float
       
class Calm():

    def __init__(self):
    
        self.hue = None
        self.sat = None
        self.val = None
        
        self.pal = None
        
        self.gradience_weight = None
        self.gradience_lift = None
        
        self.unarm = None
        self.unaim = None
        
class Calmer():

    def __init__(self, hue_dist=None, sat_dist=None, val_dist=None, pal_dist=None, gradience_dd=None, unarm_dist=None, unaim_dist=None):
    
        self.hue_dist = hue_dist
        self.sat_dist = sat_dist
        self.val_dist = val_dist
        
        self.pal_dist = pal_dist
        
        self.gradience_dd = gradience_dd
        
        self.unarm_dist = unarm_dist
        self.unaim_dist = unaim_dist

class Correlation():

    def __init__(self):
    
        self.isotopeID_0 = None
        self.element_0 = None
        self.param_index_0 = None
    
        self.isotopeID_1 = None
        self.element_1 = None
        self.param_index_1 = None
        
        self.relation_c = None
        self.relation_r = None
        
        self.strength = None
        
    def __str__(self):
    
        return "; ".join(key+": "+str(value) for key, value in vars(self).items())
  
class Version():

    def __init__(self, prism=None, puddle=None, pasteller=None, permuter=None, procession=None, controller=None, conceiver=None, correlator=None, calmer=None):
    
        self.prism = prism # tuple (dist)
        self.puddle = puddle # Puddle
        self.pasteller = pasteller # Pasteller
        self.permuter = permuter # tuple (dist)
        self.procession = procession # float
        
        self.controller = controller # dict
        
        self.conceiver = conceiver # dict
        self.correlator = correlator # list
        
        self.calmer = calmer # Calmer

class Everything():

    def __init__(self):
        
        self.artist = None
        self.artist_name = None
        self.version = None
        self.version_name = None
        self.ptych = None
        self.ptych_name = None



### MINOR CONCEPTS

class FanConcept():

    def __init__(self, shift_dist=None, logweight_dist=None):
    
        self.shift_dist = shift_dist
        self.logweight_dist = logweight_dist

class WeightConcept():

    def __init__(self, balanced_amt_dist=None, imbalanced_base_dist=None, reversing_chance=None, shufflity_dist=None, starting_index_dist=None):
    
        self.balanced_amt_dist = balanced_amt_dist
        self.imbalanced_base_dist = imbalanced_base_dist
        self.reversing_chance = reversing_chance
        self.shufflity_dist = shufflity_dist
        self.starting_index_dist = starting_index_dist

class WaveConcept():

    def __init__(self, nice_chance=None, unnice_chance=None, off_unnice_chance=None, freqhood_dist=None, numerhood_dist=None, denomhood_dist=None, unnice_dist=None, offreqhood_dist=None, offdex_dist=None, off_unnice_dist=None):
            
        self.nice_chance = nice_chance
        self.unnice_chance = unnice_chance
        self.off_unnice_chance = off_unnice_chance
        self.freqhood_dist = freqhood_dist
        self.numerhood_dist = numerhood_dist
        self.denomhood_dist = denomhood_dist
        self.unnice_dist = unnice_dist
        self.offreqhood_dist = offreqhood_dist
        self.offdex_dist = offdex_dist
        self.off_unnice_dist = off_unnice_dist
        
    def __str__(self):
        return "\n".join([var+": "+str(val) for var, val in self.__dict__.items()])



### MAJOR CONCEPTS

class CartConcept():

    def __init__(self, nice_chance=None, unnice_chance=None, mean_dist=None, freqhood_dist=None, index_dist=None, unnice_dist=None, updating=None):
    
        self.nice_chance = nice_chance
        self.unnice_chance = unnice_chance
        self.mean_dist = mean_dist
        self.freqhood_dist = freqhood_dist
        self.index_dist = index_dist
        self.unnice_dist = unnice_dist
        self.updating = updating
        
class RandConcept():

    def __init__(self, freqhood_dist=None, squarity_dist=None):
    
        self.freqhood_dist = freqhood_dist
        self.squarity_dist = squarity_dist
     
class PowConcept():

    def __init__(self, logex_dist=None):
        
        self.logex_dist = logex_dist
     
class PowerConcept():

    def __init__(self, weighing_base_chance=None, base_weight_dist=None, exp_weight_dist=None, exp_lifthood_dist=None):
    
        self.weighing_base_chance = weighing_base_chance
        self.base_weight_dist = base_weight_dist
        self.exp_weight_dist = exp_weight_dist
        self.exp_lifthood_dist = exp_lifthood_dist
         
class SigmoidConcept():

    def __init__(self, midpoint_dist=None, squarity_dist=None):
            
        self.midpoint_dist = midpoint_dist
        self.squarity_dist = squarity_dist
            
class ArcfanConcept():

    def __init__(self, fc=None):
    
        self.fc = fc
   
class SinConcept():

    def __init__(self, wc=None):
            
        self.wc=wc
        
class SpinConcept():

    def __init__(self, wc=None, fc=None, antiwise_chance=None, armhood_dist=None):
        
        self.wc = wc
        self.fc = fc
        self.antiwise_chance = antiwise_chance
        self.armhood_dist = armhood_dist
                
class MinxConcept():

    def __init__(self, maxing_chance=None, wc=None):
    
        self.maxing_chance = maxing_chance
        self.wc = wc
            
class AmeanConcept():

    def __init__(self, wc=None):
    
        self.wc = wc
   
class ItConcept():

    def __init__(self, iterance_0_dist=None, iterance_1_dist=None, xlike_chance=None):
    
        self.iterance_0_dist = iterance_0_dist
        self.iterance_1_dist = iterance_1_dist
        self.xlike_chance = xlike_chance




### OTHER GLOBALS

SETTINGS = {}

ALL_ELEMENTS = {}

SEEDER = Seeder()

UNRANDOM = random.Random(0)
RANDS = [UNRANDOM.uniform(-1, 1) for _ in range(1000)]

HUE_ADJUSTMENTS = [(0, 0.044), (0.125, -0.028), (0.25, -0.074), (0.375, 0.007), (0.625, 0.081), (0.75, 0.044), (0.875, 0.0956)]



### THICKET

def validate_answer(answer, answers, atom):

    if isinstance(answer, complex) or answer < -SETTINGS["BOUNDARY"] or answer > SETTINGS["BOUNDARY"]:
        print("bad answer: "+str(answer))
        print("element: "+atom.element)
        print("params: "+"; ".join([str(p) for p in atom.params]))
        print("children: "+"; ".join([str(answers[ch]) for ch in atom.children]))
        exit(0)

def thicket_descendants(thicket, ID, descendants):
    
    for ch in thicket[ID].children:
    
        if ch not in descendants:
        
            thicket_descendants(thicket, ch, descendants)

    descendants.append(ID)
    
def fill_descendants(thicket):

    for atom in thicket:
        
        descendants = []
        thicket_descendants(thicket, atom.ID, descendants)
        atom.descendants = descendants
            
def calculate_up(ix, iy, tm, questions, answers, printing=False):

    for index in tm.master:
    
        atom = tm.thicket[index]
        
        if atom.ID not in answers:
    
            if atom.element == "X":
                if len(questions) == 2: answers[atom.ID] = ix
                else: answers[atom.ID] = atom.func(ix, atom.params)
                
            elif atom.element == "Y":
                if len(questions) == 2: answers[atom.ID] = iy
                else: answers[atom.ID] = atom.func(iy, atom.params)
                
            elif atom.element == "IT":
                recalculating_descendants = [d for d in atom.descendants if d in tm.thicket[atom.children[2]].descendants+tm.thicket[atom.children[3]].descendants+atom.children[:2]]
                start_IDs = atom.children[:2]
                arrest = {start_ID : answers[start_ID] for start_ID in start_IDs}
                answers[atom.ID] = atom.func(start_IDs, arrest, answers[atom.children[4]], atom.children[2:4], atom.params, Thicketmaster(tm.thicket, recalculating_descendants))
            
            else:
                answers[atom.ID] = atom.func(tuple([answers[ch] for ch in atom.children]), atom.params)
            
        if printing:
            print(str(atom.ID)+" "+str(atom.element)+" "+str(answers[atom.ID]))
            
        ## remove this check for speed
        # validate_answer(answers[atom.ID], answers, atom)
        ##
            
    return [answers[q] for q in questions]
    
def func_it(start_IDs, arrest, iterance_argument, end_IDs, params, tm):

    # printing = random.random() <= 0.00009
    printing = False
        
    iterance = ity(params[0], params[1], (math.atan(iterance_argument)/math.pi) + 0.5)
    
    it_low = int(iterance)
    it_high = math.ceil(iterance)
    it_amt = iterance % 1
    
    ix = arrest[start_IDs[0]]
    iy = arrest[start_IDs[1]]
    
    count_it = 0
    
    if printing:
        print("\n\niterating with initial values "+str(ix)+", "+str(iy))
    
    while True:
    
        if count_it == it_low:
            if params[2]: final_low = ix
            else: final_low = iy
            
        if count_it == it_high:
            if params[2]: final_high = ix
            else: final_high = iy
            return ity(final_low, final_high, it_amt)
    
        ix, iy = calculate_up(ix, iy, tm, end_IDs, copy.copy(arrest), printing)
        
        if printing:
            print("new values are "+str(ix)+", "+str(iy))

        count_it += 1
    
    
    
### MATH

def ity(start, end, amt):

    return ((end - start) * amt) + start

def yti(start, end, x):

    return (x - start)/(end - start)

def hypot(delta_1, delta_2):

    return (delta_1 * delta_1 + delta_2 * delta_2) ** 0.5
             
def sigmoid(x, m, a):

    # \arctan\left(\left(x-m\right)\cdot\left(3500^{a}-1\right)\right)\cdot\frac{2}{\pi}
    
    return math.atan((x-m)*(3500**a-1)) * 2 / math.pi
       
def bound(x):

    return min(max(x, -SETTINGS["BOUNDARY"]), SETTINGS["BOUNDARY"])
    
def unit_bound(x):

    return min(max(x, -1), 1)
       
def cosoften(x):

    return (1 - math.cos(math.pi * x)) / 2
    
def derestrict(x):

    return x*2-1
 
def restrict(x):
    return (x+1)/2
 
def tame_pow(base, ex):

    if base == 0: return 0

    if base < 0 and ex % 1 != 0:
        try:
            return -((-base) ** ex)
        except OverflowError:
            return -sys.float_info.max        
        
    try:
        return base ** abs(ex)
    except OverflowError:
        return sys.float_info.max
    
def arctango(delta_y, delta_x):

    if delta_x == 0:
    
        if delta_y > 0: return math.pi/2
        else: return 3*math.pi/2
            
    elif delta_x > 0:
        return math.atan(delta_y/delta_x) % (2*math.pi)
        
    else:
        return math.atan(delta_y/delta_x) + math.pi
    
def arcfan(delta_y, delta_x):

    if delta_x == 0:
        return 0

    if delta_x > 0:
        return abs(math.atan(delta_y/delta_x)) * (2/math.pi) - 1
        
    else:
        return 1 - abs(math.atan(delta_y/delta_x)) * (2/math.pi)



### MATH  / LISTS

def dot(x, y):

    return sum([x[j] * y[j] for j in range(len(x))])
    
def geomean(x):

    product = 1
    
    for i in range(len(x)): product *= x[i]
    
    return product
    
    # return product ** (1/len(x))
    return tame_pow(product, 1/len(x))

def normalize(x):

    norm = sum(x)
    
    if norm == 0:
        for i in range(len(x)): x[i] = 1/len(x)
        return
    
    for i in range(len(x)): x[i] /= norm
 


### RANDOM

def betta(low, mid, high, sharp, boost):
    
    if high == low or sharp == None: return mid

    if random.random() <= boost: return ity(low, high, random.random())
    
    n = (mid - low)/(high - low)
    
    s = math.exp(10*(sharp-abs(n - 0.5)*0.5)-2)
    
    return ity(low, high, random.betavariate(n * s + 1, (1 - n) * s + 1))

def doubetta(dist_0, bottom_chance_0, top_chance_0, dist_1, bottom_chance_1, top_chance_1):
    
    a = random.random()
    
    if a <= bottom_chance_0:
        result_0 = 0
        
    elif a <= bottom_chance_0 + top_chance_0:
        result_0 = 1
        
    else:
        result_0 = betta(*dist_0)
    
    b = random.random()
    
    if b <= bottom_chance_1:
        result_1 = 0
        
    elif b <= bottom_chance_1 + top_chance_1:
        result_1 = 1
        
    else:
        result_1 = betta(*dist_1)
        
    return result_0, result_1

def reseed():

    global SEEDER
    
    if SEEDER.painting_index == None: return
    
    random.seed(SEEDER.seed_ptychs[SEEDER.painting_index][SEEDER.seed_ptyndex])
    
    SEEDER.seed_ptyndex += 1
    
    # print(str(SEEDER.seed_ptyndex))
    
    pass
        
def generate_seeds(num_paintings):

    global SEEDER

    ## base
    typical_sticks  = [random.randrange(10**10) for i in range(100)]
    sticks = [copy.copy(typical_sticks) for j in range(num_paintings)]
    
    ## twiddle
    for i in range(num_paintings):
    
        # don't change the 81 constant
        # this is the number of points during ptych generation where random can be reseeded
        #
        # art.py: generate_palettes: 23
        # art.py: generate_borders: 1
        # art.py: generate_glow: 1
        # art_scheme.py: generate_themed_thicket: generate_thicket: 27
        # art_scheme.py: generate_themed_thicket: conceive_thicket: 25
        # art_scheme.py: generate_themed_thicket: correlate_thicket: 3
        # art_scheme.py: apply_calmer: 6
        
        chosen_indices = random.sample(range(86), k = int(betta(SETTINGS["TWIDDLE LOW"], SETTINGS["TWIDDLE MID"], SETTINGS["TWIDDLE HIGH"], SETTINGS["TWIDDLE SHARP"], SETTINGS["TWIDDLE BOOST"])))
        
        for j in chosen_indices:
            sticks[i][j] = random.randrange(10 ** 10)
            
            
    SEEDER.seed_ptychs = sticks
    
def rand(natural_index):

    global RANDS
    
    if natural_index <= 0: index = abs(2*natural_index)
    else: index = 2*natural_index - 1

    if index < len(RANDS): return RANDS[index]
    
    RANDS += [UNRANDOM.uniform(-1, 1) for _ in range(int(index * 1.5)-len(RANDS))]
    
    return RANDS[index]
    
def reweight(weights, base, coverage = None, sharp = None):

    out_weights = copy.copy(weights)
    
    if coverage == None: coverage = SETTINGS["CONCEPT REWEIGHT BREADTH"]
    if sharp == None: sharp = SETTINGS["CONCEPT REWEIGHT SHARP"]
    
    # for i in range(int(betta(0, 1, len(weights)+0.5, 0.5, 0.1))): out_weights[random.randrange(len(out_weights))] *= (random.uniform(1, base) ** random.uniform(-1,1))
    
    num_reweights = int(len(weights) * betta(0, coverage, 2, sharp, SETTINGS["STANDARD BOOST"]))
    for i in range(num_reweights): out_weights[random.randrange(len(out_weights))] *= (random.uniform(1, base) ** random.uniform(-1,1))
    
    return out_weights
    
def biased_choice(options, freedom):

    return options[min(max(round(betta(-0.5, -0.5, len(options) - 0.5, 1.5 - 2*freedom, freedom/2)), 0), len(options)-1)]
    
def partial_shuffle(x, amt):
    
    return [x[i] for i in sorted(list(range(len(x))), key=lambda v: v + random.random()*(amt*len(x)+1)**1.5)]
 
def weighted_choose(options, k, weights):

    chosen = []
    
    available = copy.copy(options)
    
    avail_weights = copy.copy(weights)
    
    while len(chosen) < k:
    
        choice = random.choices(range(len(available)), k=1, weights=avail_weights)[0]
        
        chosen.append(available[choice])
        
        del available[choice]
        del avail_weights[choice]
        
    return chosen
      
def random_partition(m, n):

    part = [1] * n
    
    cins = sorted([0]+[int(random.random()*(m-n)) for i in range(n-1)]+[m-n])
    
    for i in range(n):
    
        part[i] += cins[i+1]-cins[i]
        
    return part
    


### COLOR

def huemanize(h):

    if h < HUE_ADJUSTMENTS[0][0]:
        h0 = HUE_ADJUSTMENTS[-1][0]
        h = h + 1
        h1 = HUE_ADJUSTMENTS[0][0] + 1
        
        shift0 = HUE_ADJUSTMENTS[-1][1]
        shift1 = HUE_ADJUSTMENTS[0][1]
        
    elif h >= HUE_ADJUSTMENTS[-1][0]:
        h0 = HUE_ADJUSTMENTS[-1][0]
        h1 = HUE_ADJUSTMENTS[0][0] + 1
        
        shift0 = HUE_ADJUSTMENTS[-1][1]
        shift1 = HUE_ADJUSTMENTS[0][1]
        
    else:
        index1 = next(i for i in range(len(HUE_ADJUSTMENTS)) if h < HUE_ADJUSTMENTS[i][0])
        index0 = index1 - 1
    
        h0 = HUE_ADJUSTMENTS[index0][0]
        h1 = HUE_ADJUSTMENTS[index1][0]
    
        shift0 = HUE_ADJUSTMENTS[index0][1]
        shift1 = HUE_ADJUSTMENTS[index1][1]
    
    shift = ity(shift0, shift1, (h - h0)/(h1 - h0))
    
    shifted = (h + shift) % 1
    
    return shifted
 
def HSV_to_RGB(hsv):

    hue = hsv[0]
    sat = hsv[1]
    val = hsv[2]

    c = sat*val
    x = c * (1 - abs((hue/60)%2 - 1))
    m = val-c
    
    if hue < 60:
        red_prime = c
        green_prime = x
        blue_prime = 0
    elif hue < 120:
        red_prime = x
        green_prime = c
        blue_prime = 0
    elif hue < 180:
        red_prime = 0
        green_prime = c
        blue_prime = x
    elif hue < 240:
        red_prime = 0
        green_prime = x
        blue_prime = c
    elif hue < 300:
        red_prime = x
        green_prime = 0
        blue_prime = c
    else:
        red_prime = c
        green_prime = 0
        blue_prime = x
        
    red = int((red_prime+m)*255)
    green = int((green_prime+m)*255)
    blue = int((blue_prime+m)*255)
    
    return (red, green, blue)
    
def RGB_to_HSV(rgb):

    red = rgb[0]
    green = rgb[1]
    blue = rgb[2]
    
    red_prime = red/255
    green_prime = green/255
    blue_prime = blue/255
    
    Cmax = max((red_prime, green_prime, blue_prime))
    Cmin = min((red_prime, green_prime, blue_prime))
    
    delta = Cmax - Cmin
    
    if delta == 0:    
        hue = 0        
    elif Cmax == red_prime:
        hue = 60 * (((green_prime - blue_prime)/delta) % 6)        
    elif Cmax == green_prime:    
        hue = 60 * (((blue_prime - red_prime)/delta) + 2)        
    elif Cmax == blue_prime:    
        hue = 60 * (((red_prime - green_prime)/delta) + 4)
        
    if Cmax == 0:
        sat = 0
    else:
        sat = delta/Cmax
        
    val = Cmax
    
    return (hue, sat, val)

def HSV_to_XYZ(hsv):

    x = hsv[2] * hsv[1] * math.cos(hsv[0]*math.pi/180)
    y = hsv[2] * hsv[1] * math.sin(hsv[0]*math.pi/180)
    z = hsv[2] - 1
    
    return (x, y, z)
    
def XYZ_to_HSV(cart):

    h = arctango(cart[1], cart[0]) * 180 / math.pi
    v = cart[2] + 1
    s = ((cart[1] * cart[1] + cart[0] * cart[0]) ** 0.5) / v
    
    return [h, s, v]

def RGB_blend(color1, color2, amt):
    
    red = int(ity(color1[0], color2[0], amt))
    green = int(ity(color1[1], color2[1], amt))
    blue = int(ity(color1[2], color2[2], amt))
    
    return (red, green, blue)

def HSV_blend(color1, color2, amt):

    cart1 = HSV_to_XYZ(color1)
    cart2 = HSV_to_XYZ(color2)
    
    x = ity(cart1[0], cart2[0], amt)
    y = ity(cart1[1], cart2[1], amt)
    z = ity(cart1[2], cart2[2], amt)
    
    return XYZ_to_HSV((x,y,z))
      


### I/O
    
def param_to_str(param):

    if isinstance(param, bool):
        return str(param)
    elif isinstance(param, (int, float)):
        return str(round(float(param), 4))
    else:
        return str(param)

def is_valid_scathe_name(name):

    return bool(re.fullmatch("[^ _!<>:\"/\\|\\?\\*]+", name))
    
def is_valid_version_pair(name):

    splat = name.split("_")

    return len(splat) == 2 and is_valid_scathe_name(splat[0]) and is_valid_scathe_name(splat[1])
    


### INTERRUPT

def check_interrupt():
    
    if ON_WINDOWS:
        return msvcrt.kbhit() and msvcrt.getch() == b'h'
        
    else:
        if CURSE.getch() == 9: return True
        CURSE.refresh()
        return False