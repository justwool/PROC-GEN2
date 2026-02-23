import itertools # cartesian product
import copy # deepcopyi objects
import json # save & load objects
import os # navigate filesystem
import shutil # copy cached bmp to data folder
import re # match filenames & user inputs
import datetime # timestamp-based filenames
import subprocess # display image after rendering
import traceback # manually print traceback, to keep window open after crash
from sys import exit # nuitka seems to need this

import art_bmp
from art_class import *
from art_scheme import *
from art_render import *



### GLOBALS


AUTO = Automator()

CONSTANTS_FILENAME = "art_constants.txt"
SETTINGS_FILENAME = "art_settings.txt"

ARTISTS = None
VERSIONS = None



### GENERATE PALETTE

def generate_components(num_colors, puddle):
    
    reseed()
    
    hue_dirtiness = betta(*puddle.hue_dirtiness_dist)
    sat_dirtiness = betta(*puddle.sat_dirtiness_dist)
    val_dirtiness = betta(*puddle.val_dirtiness_dist)
    
    hue_stillness = betta(*puddle.hue_stillness_dist)
    sat_stillness = betta(*puddle.sat_stillness_dist)
    val_stillness = betta(*puddle.val_stillness_dist)

    reseed()

    num_primary_hues = int(ity(2, num_colors+1, random.uniform(0,1)) ** (hue_stillness+1))
    num_primary_sats = int(ity(1, num_colors+1, random.uniform(0,1)) ** (sat_stillness+1))
    num_primary_vals = int(ity(2, num_colors+1, random.uniform(0,1)) ** (val_stillness+1))
    
    reseed()
    
    num_hue_blocks = int(ity(num_primary_hues, num_primary_hues * (1+hue_dirtiness), random.uniform(0,1)))
    num_sat_blocks = int(ity(num_primary_sats, num_primary_sats * (1+sat_dirtiness), random.uniform(0,1)))
    num_val_blocks = int(ity(num_primary_vals, num_primary_vals * (1+val_dirtiness), random.uniform(0,1)))
    
    reseed()
    chosen_hue_parts = [int(num_hue_blocks*random.random()) for _ in range(num_primary_hues)]
    
    reseed()
    chosen_sat_parts = [int(num_sat_blocks*random.random()) for _ in range(num_primary_sats)]
    
    reseed()
    chosen_val_parts = [int(num_val_blocks*random.random()) for _ in range(num_primary_vals)]
        
    reseed()
    
    if SEEDER.seed_ptyndex != None and SEEDER.painting_index != 0:
        first_hue_seed = SEEDER.seed_ptychs[0][SEEDER.seed_ptyndex]
        random.seed(first_hue_seed)
        
    hue_shift = random.uniform(-1,1)
    
    reseed()
    sat_shift = random.uniform(-1,1)
    
    reseed()
    val_shift = random.uniform(-1,1)
    
    hue_fractions = [n/num_hue_blocks for n in chosen_hue_parts]
    sat_fractions = [n/num_sat_blocks for n in chosen_sat_parts]
    val_fractions = [n/num_val_blocks for n in chosen_val_parts]
    
    least_hue = sorted(hue_fractions)[0]
    hue_fractions = list(map(lambda x: x - least_hue, hue_fractions))
    
    hues = [huemanize((frac + hue_shift) % 1)*360 for frac in hue_fractions]
    sats = [(frac + sat_shift) % 1 for frac in sat_fractions]
    vals = [(frac + val_shift) % 1 for frac in val_fractions]
    
    return hues, sats, vals

def pastelize_components(sats, vals, pasteller):

    reseed()
    
    num_to_desat = int(betta(0, pasteller.frac_mid, 1, SETTINGS["DESAT FRAC SHARP"], SETTINGS["DESAT FRAC BOOST"]) * len(sats))
    
    typical_desat_amt = betta(0, 0, 1, pasteller.amt_sharp, SETTINGS["DESAT AMT BOOST"])
    
    reseed()

    desat_indices = random.sample(range(len(sats)), num_to_desat)
    
    for i in desat_indices:
    
        if random.random() <= SETTINGS["UNSAT CHANCE"]: sats[i] = 0
        else: sats[i] *= (1 - typical_desat_amt)
    
    reseed()
        
    for i in range(len(vals)):
        vals[i] = restrict(sigmoid(derestrict(vals[i]), derestrict(SETTINGS["HYPERVAL MIDPOINT"]), typical_desat_amt*SETTINGS["HYPERVAL AMT"]*0.25))

def combine_components(hues, sats, vals, num_colors):


    full_product = [list(x) for x in list(itertools.product(*[hues, sats, vals]))]
    
    if len(full_product) < num_colors:
        to_dupes = copy.deepcopy(random.sample(full_product * math.ceil((num_colors - len(full_product))/len(full_product)), k = num_colors - len(full_product)))
        full_product += to_dupes
    
    reseed()
    full_product = partial_shuffle(full_product, 0.125)
    reseed()
    full_product = partial_shuffle(full_product, 0.125)
    reseed()
    full_product = partial_shuffle(full_product, 0.125)
    
    reseed()
    hsvs = random.sample(full_product, num_colors)
    
    max_val = max([hsv[2] for hsv in hsvs])
    if max_val <= SETTINGS["MIN MAX VALUE"]:
        for hsv in hsvs:
            if hsv[2] == max_val:
                hsv[2] = SETTINGS["MIN MAX VALUE"]
                break
    
    min_val = min([hsv[2] for hsv in hsvs])
    if min_val >= SETTINGS["MAX MIN VALUE"]:
        for hsv in hsvs:
            if hsv[2] == min_val:
                hsv[2] = SETTINGS["MAX MIN VALUE"]
                break
    
    reseed()
    random.shuffle(hsvs)
    
    return hsvs
    
def palettize_colors(base_colors, num_palettes, procession):
    
    thick_mids = [random.uniform(0.05, 1), random.uniform(0.05, 1)]
    thick_sharps = [random.uniform(-0.5, 1.5), random.uniform(-0.5, 1.5)]
    first_cutoff = random.random()
    base_thicks = []
    for _ in base_colors:
        thickdex = 0 if random.random() <= first_cutoff else 1
        base_thicks.append(betta(0.05, thick_mids[thickdex], 1, thick_sharps[thickdex], 0.1))
    base_thicks.sort()
    
    reseed()
    
    shuffled_color_indices = partial_shuffle(range(len(base_colors)), random.random() ** procession)
    shuffled_thick_indices = partial_shuffle(shuffled_color_indices, random.random() ** procession)

    palettes = [Palette() for _ in range(num_palettes)]
    
    for i in range(num_palettes):
    
        if i == int(num_palettes*0/3): reseed()
        if i == int(num_palettes*1/3): reseed()
        if i == int(num_palettes*2/3): reseed()
    
        palette = palettes[i]
    
        colors = copy.deepcopy(base_colors)
        thicks = copy.deepcopy(base_thicks)

        hue_weight = random.uniform(-1, 1)
        sat_weight = random.uniform(-1, 1)
        val_weight = random.uniform(-1, 1)
    
        component_weights = [hue_weight, sat_weight, val_weight]
    
        colors.sort(key = lambda x: dot(x, component_weights))
        sorted_color_indices = sorted(range(len(colors)), key = lambda x: dot(colors[x], component_weights))
        
        # yes this is correct
        sorted_colors = [colors[j] for j in sorted_color_indices]
        sorted_thicks = [thicks[j] for j in sorted_color_indices]
    
        # yes this is correct
        shuffled_colors = [sorted_colors[j] for j in shuffled_color_indices]
        shuffled_thicks = [sorted_thicks[j] for j in shuffled_thick_indices]
        
        palette.colors = shuffled_colors
        
        normalize(shuffled_thicks)
        palette.quicks = [sum(shuffled_thicks[:i+1]) for i in range(len(shuffled_thicks))]
        
        palette.base_colors = base_colors
        
    reseed()
    random.shuffle(palettes)
    
    return palettes
    
def generate_palettes(prism, permuter, pasteller, puddle, procession):

    ### first reseed & randomness for a painting
    reseed()

    num_colors = int(betta(*prism))
    
    hues, sats, vals = generate_components(num_colors, puddle)
        
    pastelize_components(sats, vals, pasteller)
    
    colors = combine_components(hues, sats, vals, num_colors)
    
    num_palettes = int(betta(*permuter))
    
    palettes = palettize_colors(colors, num_palettes, procession)
    
    return palettes



### GENERATE OTHER
    
def generate_glow(palette):

    reseed()

    g = Glow()
    
    g.is_circular = (random.uniform(0, 1) <= SETTINGS["CIRCULAR GLOW CHANCE"])
    
    g.height = betta(0, SETTINGS["GLOW HEIGHT MID"], SETTINGS["GLOW HEIGHT MAX"], 0.5, 0)
    
    g.color = HSV_to_RGB(random.choice(palette))
    
    return g

def generate_borders(palette):

    reseed()

    num_min = SETTINGS["NUM BORDERS MIN"]
    num_max = SETTINGS["NUM BORDERS MAX"]+1
    num_borders = int(betta(num_min, (num_min+num_max)/2, num_max, SETTINGS["NUM BORDERS SHARP"], SETTINGS["NUM BORDERS BOOST"]))
    
    borders = []
    
    total_height = random.uniform(SETTINGS["BORDER HEIGHT MIN"], SETTINGS["BORDER HEIGHT MAX"])
    
    dividers = [0] + sorted([random.random()*total_height for i in range(num_borders)]) + [total_height]
    
    for i in range(num_borders):
    
        height = ity(dividers[i], dividers[i+1], random.random())
        
        if random.random() <= SETTINGS["FLAT BORDER CHANCE"]: amplitude = 0
        else: amplitude = min(height-dividers[i], dividers[i+1]-height) * random.random()
        
        frequency = betta(SETTINGS["BORDER FREQUENCY MIN"], SETTINGS["BORDER FREQUENCY MID"], SETTINGS["BORDER FREQUENCY MAX"], SETTINGS["BORDER FREQUENCY SHARP"], SETTINGS["BORDER FREQUENCY BOOST"])
        
        if random.uniform(0,1) <=  SETTINGS["BORDER NEGEX CHANCE"]:
            exponent = 2 ** betta(SETTINGS["BORDER LOGEX MIN"], 0, 0, SETTINGS["BORDER LOGEX SHARP"], SETTINGS["BORDER LOGEX BOOST"])
        else:
            exponent = 2 ** betta(0, 0, SETTINGS["BORDER LOGEX MAX"], SETTINGS["BORDER LOGEX SHARP"], SETTINGS["BORDER LOGEX BOOST"])
        
        border = Border()
        
        border.height = height
        border.period = 1/frequency
        border.exponent = exponent
        border.amplitude = amplitude
        border.color = HSV_to_RGB(random.choice(palette))
            
        borders.append(border)
    
    borders = list(reversed(borders))
    
    return borders



### MANAGE PTYCH

def choose_num_paintings():
    
    if random.random() <= SETTINGS["SINGLE PAINTING CHANCE"]: return 1

    return int(betta(2, 2, 6, SETTINGS["NUM PAINTINGS SHARP"], SETTINGS["NUM PAINTINGS BOOST"]))

def arrange_ptych(num_paintings, canvas_width, canvas_height):

    if num_paintings == 0: return []
    
    if num_paintings == 7: return [2, 3, 2]
            
    if canvas_width <= canvas_height:
        if num_paintings == 6: layout = [3, 3]
        else: layout = [num_paintings]
        
    else:
        if num_paintings == 6: layout = [2, 2, 2]
        else: layout = [1] * num_paintings
        
    return layout

def synchronize_borders(painting, num_paintings, ptych):

    if SEEDER.painting_index == 0: return
    
    # first half: only repeat alternates
    if SEEDER.painting_index <= int((num_paintings-1)/2):
    
        if SEEDER.painting_index % 2 == 0: painting.borders = copy.deepcopy(ptych.paintings[0].borders)
        
    # second half: mirror first half
    else: painting.borders = copy.deepcopy(ptych.paintings[(num_paintings - SEEDER.painting_index) - 1].borders)
    


### ARTIST SEARCH

def calc_time(thicket, times):
    
    total_time = 0
    
    for atom in thicket:
    
        if atom.element != "IT":
            total_time += ALL_ELEMENTS[atom.element].time
        
        elif atom.element == "IT":
                
            if times[atom.children[2]] == None: time_0, times = calc_time(thicket[:atom.children[2]+1], times)
            else: time_0 = times[atom.children[2]]
            
            if times[atom.children[3]] == None: time_1, times = calc_time(thicket[:atom.children[3]+1], times)
            else: time_1 = times[atom.children[3]]
            
            total_time += (time_0 + time_1) * (atom.params[1]-1) + ALL_ELEMENTS["IT"].time
            
    times[len(thicket)-1] = total_time
            
    return total_time, times
            
def calc_speed(thicket):
    
    speed = 1000 / (calc_time(thicket, [None]*len(thicket))[0])
    
    return speed
       
def calc_intricacy(whirl, resolution, x_range, y_range):

    tm = Thicketmaster(whirl.thicket, list(range(len(whirl.thicket))))
    
    num_good = 0
    
    for _ in range(SETTINGS["INTRICACY SAMPLE SIZE"]):
    
        first_pixel = (random.randrange(*x_range), random.randrange(*y_range))
        
        if first_pixel[0] == x_range[0]: x_diffs = (0, 1)
        elif first_pixel[0] == x_range[1]-1: x_diffs = (-1, 0)
        else: x_diffs = (-1, 0, 1)
        
        if first_pixel[1] == y_range[0]: y_diffs = (0, 1)
        elif first_pixel[1] == y_range[1]-1: y_diffs = (-1, 0)
        else: y_diffs = (-1, 0, 1)
        
        second_pixels = [(first_pixel[0]+x_diff, first_pixel[1]+y_diff) for x_diff in x_diffs for y_diff in y_diffs]
        second_pixels = [pixel for pixel in second_pixels if pixel != first_pixel]
        
        second_pixel = random.choice(second_pixels)
        
        start = time.time()
        first_color = whirl_pixel(first_pixel[0]/resolution, first_pixel[1]/resolution, whirl, tm)
        end = time.time()
        if end-start > SETTINGS["EMPIRICAL PIXEL TIME CUTOFF"]:
            if SETTINGS["PRINT FAILURES"]: print("empirical speed too low")
            return None
        tm = Thicketmaster(whirl.thicket, list(range(len(whirl.thicket))))
        second_color = whirl_pixel(second_pixel[0]/resolution, second_pixel[1]/resolution, whirl, tm)
        
        first_vector = HSV_to_XYZ(first_color)
        second_vector = HSV_to_XYZ(second_color)
        
        differences = [abs(first_vector[i]-second_vector[i]) for i in range(3)]
        
        distance = math.sqrt(sum([diff * diff for diff in differences]))
        
        if distance >= SETTINGS["INTRICACY CONTRAST CUTOFF"]: num_good += 1
        
    return num_good/SETTINGS["INTRICACY SAMPLE SIZE"]
       
def calc_intricacies(thicket, roots, version):

    ### copied from art.py & art_render.py - avoid duplicating this code
    canvas_width, canvas_height = random.sample((SETTINGS["ANALYSIS RESOLUTION"], int(SETTINGS["ANALYSIS RESOLUTION"] / random.uniform(1, 2))), k = 2)
    palettes = generate_palettes(version.prism, version.permuter, version.pasteller, version.puddle, version.procession)
    fill_descendants(thicket)
    whirl = Whirl(palettes, thicket, roots, apply_calmer(version.calmer))
    
    edge_lengths = [canvas_width/i for i in range(1, SETTINGS["ANALYSIS NUM REGIONS"]+1)]
    edge_lengths += [canvas_height/i for i in range(1, SETTINGS["ANALYSIS NUM REGIONS"]+1)]
    
    edge_lengths.sort(reverse=True)
    
    resulting_squares = lambda x: (canvas_width//x) * (canvas_height//x)
    
    edge_length = next(el for el in edge_lengths if resulting_squares(el) >= SETTINGS["ANALYSIS NUM REGIONS"])
    
    width_count = int(canvas_width/edge_length)
    height_count = int(canvas_height/edge_length)
    
    oversector_width = canvas_width / width_count
    oversector_height = canvas_height / height_count
    
    intricacies = []
    
    for i in range(width_count*height_count):
    
        x_min = int(oversector_width*(i%width_count) + (oversector_width-edge_length)/2)
        x_max = int(x_min+edge_length+1)
    
        y_min = int(oversector_height*int(i/width_count) + (oversector_width-edge_length)/2)
        y_max = int(y_min+edge_length+1)
        
        intricacy = calc_intricacy(whirl, SETTINGS["ANALYSIS RESOLUTION"], (x_min, x_max), (y_min, y_max))
        if intricacy == None: return None, None
        
        intricacies.append(intricacy)
        
    return min(intricacies), max(intricacies)
    
def analyze_soft(high_intricacies):

    sorted_intricacies = sorted(high_intricacies)

    median_high_intricacy = sorted_intricacies[int(len(high_intricacies)/2)]
    
    return median_high_intricacy >= SETTINGS["MIN MEDIAN INTRICACY"]
       
def analyze_hard(thicket, roots, version, artist):
    
    apply_conceiver(thicket, version.conceiver, artist, False)
    apply_correlator(thicket, version.correlator, False)
        
    speed = calc_speed(thicket)
        
    failed = False
    
    if not speed >= SETTINGS["MIN LOWEST SPEED"]:
        failed = True
        if SETTINGS["PRINT FAILURES"]: print("speed too low: "+str(speed))
        
    if failed: return False, None
    
    low_intricacy, high_intricacy = calc_intricacies(thicket, roots, version)
    
    if low_intricacy == None: return False, None
    
    if not high_intricacy >= SETTINGS["MIN LOWEST INTRICACY"]:
        failed = True
        if SETTINGS["PRINT FAILURES"]: print("intricacy too low")
        
    return True, high_intricacy
            
def find_version(artist, artist_searching, artist_name, max_attempts):

    try_forever = (max_attempts == None)

    ## announce
    if not artist_searching:
        if artist_name == None: artist_name = "(unnamed artist)"
        print("\nSearching for a new version of ~ "+artist_name+"...")
             
    num_tries = 0
         
    while try_forever or num_tries < max_attempts:
   
        ## slow search
        if SETTINGS["SEARCH SLEEP MS"] >= 15:
            if num_tries % SETTINGS["SEARCH SLOWING PERIOD"] == 0:
                time.sleep(SETTINGS["SEARCH SLEEP MS"]/1000)
                
        version = Version(random_prism(), random_puddle(), random_pasteller(), random_permuter(), random_procession(), random_controller(artist), random_conceiver(artist, {}), random_correlator(artist), random_calmer())
        
        controlled_artist = apply_controller(version.controller, artist)
        
        high_intricacies = []
        
        acceptable = True
        
        for _ in range(SETTINGS["REQUIREMENT STRINGENCY"]):
    
            thicket, roots = grow_thicket(artist, False, False)
            if not thicket:
                if SETTINGS["PRINT FAILURES"]: print("overgrown thicket")
                acceptable = False
                break
            
            acceptable, high_intricacy = analyze_hard(thicket, roots, version, artist)
            if not acceptable: break
            
            high_intricacies.append(high_intricacy)
            
            
        if acceptable:
        
            accepted = analyze_soft(high_intricacies)
            
            if accepted:
                        
                if artist_searching: print("\n\nArtist and version found!")
                else: print("\n\nVersion found!")
                
                return version
        
        num_tries += 1
        
        if not artist_searching:
            word = "version" if num_tries == 1 else "versions"
            ending = "\r" if not SETTINGS["PRINT FAILURES"] else "\n"
            print("Tried "+str(num_tries)+" "+word+".", end=ending)
            
        if check_interrupt(): return True
        
    return None

def find_artist_and_version():

    print("\nSearching for a new artist...\n")

    num_tries = 0
    while True:

        artist = random_artist()
        
        version = find_version(artist, True, None, SETTINGS["NEW ARTIST NUM VERSIONS"])
        if version == True: return True, True
        
        elif version != None:
            return artist, version
        
        else:
            
            num_tries += 1
            
            word = "artists" if num_tries != 1 else "artist"
            print("Tried "+str(num_tries)+" "+word+".", end="\r")


  
### I/O: GRAPHS & TEXTS
                
def formulize(thicket, roots, deqs):

    for index in range(len(thicket)):
    
        atom = thicket[index]

        ftext = ALL_ELEMENTS[atom.element].texter(atom.children, atom.params)
        ftext = ftext.replace("+ -", "- ").replace("- -", "+ ")
        
        deq = "x"+str(index)+" = "+ftext
        
        if index in roots:
            root = ALL_ROOTS[roots.index(index)].replace("ROOT_", "")
            deq = "\n"+root+"\n"+deq
        
        deqs[index] = (index, deq)

def ptych_to_text(ptych):

    texts = []

    for i in range(len(ptych.paintings)):
    
        whirl = ptych.paintings[i].whirl
    
        thicket = whirl.thicket
        roots = whirl.roots

        root_id = roots[0]
        
        root = next(c for c in thicket if c.ID == root_id)
        
        deqs = {}
        formulize (thicket, roots, deqs)
        
        ids = sorted(list(deqs), key = lambda x:deqs[x][0], reverse=True)
        
        lines = [deqs[eid][1] for eid in ids]
        
        text = "\n".join(lines)
        
        texts.append(text)
        
    full_text = "\n\n".join(texts)
    
    return full_text

def ptych_to_graph(ptych):

    nodes = []
    
    edges = []

    for i in range(len(ptych.paintings)):
    
        thicket = ptych.paintings[i].whirl.thicket
        roots = ptych.paintings[i].whirl.roots

        for atom in thicket:
        
            code = str(atom.ID) + "_" + str(i)
            
            roots = [ALL_ROOTS[j].replace("ROOT_","") for j in range(len(roots)) if roots[j] == atom.ID]
            if len(roots) > 0: name = ", ".join(roots)+": "+atom.element
            else: name = atom.element
        
            node = code + " " + name
            
            nodes.append(node)
            
            for child in atom.children:
            
                child_code = str(child) + "_" + str(i)
            
                edge = code + " " + child_code
                edges.append(edge)
                
    full_graph = "\n".join(nodes) + "\n#\n" + "\n".join(edges)
    
    return full_graph
    
def artist_to_graph(artist):

    nodes = []
    
    edges = set([])

    for isotope in artist.values():
    
        node = str(isotope.ID)+" "+isotope.element+"_"+str(isotope.ID)
        nodes.append(node)
        
        for olist in isotope.options:
        
            for o in olist:
        
                edge = str(isotope.ID) + " " + str(o)
                edges.add(edge)
            
    full_graph = "\n".join(nodes) + "\n#\n" + "\n".join(list(edges))
    
    return full_graph


        
### I/O: SETTINGS
 
def validate_settings():

    reqs = []
    
    reqs.append(isinstance(SETTINGS["DEFAULT RESOLUTION"], int))
    reqs.append(SETTINGS["DEFAULT RESOLUTION"] > 0)
    reqs.append(isinstance(SETTINGS["SEARCH SLOWING"], (int, float)))
    reqs.append(0 <= SETTINGS["SEARCH SLOWING"])
    reqs.append(isinstance(SETTINGS["RENDER SLOWING"], (int, float)))    
    reqs.append(0 <= SETTINGS["RENDER SLOWING"])
    # reqs.append(isinstance(SETTINGS["MEDIAN STRIATION MAX"], (int, float)))
    # reqs.append(isinstance(SETTINGS["HIGH STRIATION MAX"], (int, float)))
    # reqs.append(0 <= SETTINGS["MEDIAN STRIATION MAX"] <= SETTINGS["HIGH STRIATION MAX"])
    # reqs.append(isinstance(SETTINGS["LOW SPEED MIN"], (int, float)))
    
    if not all(reqs):
        print("Invalid settings.\n")
        exit(0)
 
def add_settings_from_file(path):

    global SETTINGS

    with open(path) as infile: text = infile.read()
    
    lines = [line.strip() for line in text.split("\n") if len(line.strip()) > 0]
    
    for line in lines:
    
        if "#" in line: line = line[:line.index("#")]
    
        splat = re.split("\t+", line)
        
        if len(splat) == 2:
        
            key = splat[0]
            
            valstring = splat[1].strip().lower()
            
            if valstring == "true": value = True
            elif valstring == "false": value = False
            
            else:
            
                try: value = int(valstring)
                except ValueError:
                    try: value = float(valstring)
                    except ValueError: value = valstring
                
            SETTINGS[key] = value

def load_settings():

    global SETTINGS
    
    absolute_art_dir = os.path.dirname(os.path.realpath(__file__))
    
    add_settings_from_file(os.path.join(absolute_art_dir, CONSTANTS_FILENAME))
    add_settings_from_file(os.path.join(absolute_art_dir, SETTINGS_FILENAME))
    
    SETTINGS["CACHE DIRECTORY"] = os.path.join(absolute_art_dir, *re.split("[/\\\\]", SETTINGS["CACHE DIRECTORY"]))
    SETTINGS["IMAGE DIRECTORY"] = os.path.join(absolute_art_dir, *re.split("[/\\\\]", SETTINGS["IMAGE DIRECTORY"]))
    SETTINGS["PTYCH DIRECTORY"] = os.path.join(absolute_art_dir, *re.split("[/\\\\]", SETTINGS["PTYCH DIRECTORY"]))
    SETTINGS["VERSION DIRECTORY"] = os.path.join(absolute_art_dir, *re.split("[/\\\\]", SETTINGS["VERSION DIRECTORY"]))
    SETTINGS["ARTIST DIRECTORY"] = os.path.join(absolute_art_dir, *re.split("[/\\\\]", SETTINGS["ARTIST DIRECTORY"]))
    SETTINGS["RERENDER DIRECTORY"] = os.path.join(absolute_art_dir, *re.split("[/\\\\]", SETTINGS["RERENDER DIRECTORY"]))
    SETTINGS["GRAPH DIRECTORY"] = os.path.join(absolute_art_dir, *re.split("[/\\\\]", SETTINGS["GRAPH DIRECTORY"]))
    
    SETTINGS["SEARCH SLEEP MS"] = SETTINGS["SEARCH SLOWING"] * SETTINGS["SEARCH SLOWING PERIOD"]
    
    validate_settings()
        
    set_render_slowing(SETTINGS["RENDER SLOWING"], SETTINGS["RENDER SLOWING PERIOD"])
  
    
    
### I/O: CACHE
    
def clear_cache(leave_picture):

    fnames = [fn for fn in os.listdir(SETTINGS["CACHE DIRECTORY"]) if os.path.isfile(os.path.join(SETTINGS["CACHE DIRECTORY"], fn))]
    
    if len(fnames) > 4:
        print("Cache folder has more files than expected. Quitting to avoid further corruption.")
        exit(0)
    
    for fname in fnames:
    
        if leave_picture and (fname.startswith("image") or fname.startswith("ptych")): continue
    
        os.remove(os.path.join(SETTINGS["CACHE DIRECTORY"], fname))
    
def update_cache_name(prefix, new_name):

    old_name = cached_filename(prefix)

    os.rename(os.path.join(SETTINGS["CACHE DIRECTORY"], old_name), os.path.join(SETTINGS["CACHE DIRECTORY"], prefix+" "+new_name))
    
def update_cache_full(image, ev):
    
    clear_cache(image == None)
    
    timestamp = datetime.datetime.now().strftime("%m%d%H%M%S")[::-1]
    
    if ev.ptych_name != None:
        image_fname = "image "+ev.ptych_name+".png"
        ptych_fname = "ptych "+ev.ptych_name+".json"
    else:
        image_fname = "image "+timestamp+".png"
        ptych_fname = "ptych "+timestamp+".json"
    
    if ev.version_name != None: version_fname = "version "+ev.version_name+".json"
    else: version_fname = "version "+timestamp+".json"
    
    if ev.artist_name != None: artist_fname = "artist "+ev.artist_name+".json"
    else: artist_fname = "artist "+timestamp+".json"
    
    cached_image_path = os.path.join(SETTINGS["CACHE DIRECTORY"], image_fname)
    
    if image != None:
    
        art_bmp.create(image, cached_image_path)
        
        if not AUTO.on:
            if ON_WINDOWS: subprocess.call(["start", "", cached_image_path], shell=True)
            else: subprocess.call(["xdg-open", cached_image_path])
        
        save_object(ev.ptych, os.path.join(SETTINGS["CACHE DIRECTORY"], ptych_fname))
    
    save_object(ev.version, os.path.join(SETTINGS["CACHE DIRECTORY"], version_fname))
    
    save_object(ev.artist, os.path.join(SETTINGS["CACHE DIRECTORY"], artist_fname))
    
    return timestamp, cached_image_path

def cached_filename(prefix):

    filenames = [fn for fn in os.listdir(SETTINGS["CACHE DIRECTORY"]) if os.path.isfile(os.path.join(SETTINGS["CACHE DIRECTORY"], fn))]
    
    matching_filenames = [fn for fn in filenames if fn.startswith(prefix+" ")]
    
    if len(matching_filenames) != 1: return None
        
    filename = matching_filenames[0]
    
    return filename

def cached_object(prefix):

    filename = cached_filename(prefix)
    
    if filename == None: return None, None

    short_name = re.fullmatch(re.escape(prefix)+"\\s+(.*)\\.\\w+", filename).group(1)
    
    obj = load_object(os.path.join(SETTINGS["CACHE DIRECTORY"], filename))
    
    return obj, short_name



### I/O: GENERAL

def directory_set(directory, strip_resolutions, extension):

    # filenames = [fn.strip() for fn in os.listdir(directory) if os.path.isfile(os.path.join(directory, fn))]
    
    filenames = []
    
    for subdirectory, subfolders, fnames in os.walk(directory):
    
        for fname in fnames:

            if os.path.isfile(os.path.join(subdirectory, fname)):
            
                filenames.append(fname)
    
    matching_names = set([])
    
    if strip_resolutions: regex = "(.+?)(\\s+\\d+)?\\s*"+re.escape(extension)
    else: regex = "(.+?)\\s*"+re.escape(extension)
    
    for fname in filenames:
        
        m = re.fullmatch(regex, fname, flags=re.IGNORECASE)
        
        if m: matching_names.add(m.group(1))
        
    return matching_names

def copy_file(src, dst):

    if os.path.isfile(dst) or os.path.isdir(dst):
        print("Cannot copy file to replace existing file or directory.\n")
        exit(0)
        
    shutil.copyfile(src, dst)
    
def rebuild_function(atom_as_dict):
            
    return ALL_ELEMENTS[atom_as_dict["element"]].func
    
def interpret_as_object(pseudo, parent_pseudo):

    if type(pseudo).__name__ == "dict":
    
        if "class" in pseudo:
            
            class_name = pseudo["class"]
        
            if class_name in globals():
            
                new_o = globals()[class_name]()
                
                for var_name in pseudo:
                    if var_name != "class":
                        setattr(new_o, var_name, interpret_as_object(pseudo[var_name], pseudo))
                        
                return new_o
                
            elif class_name == "function":
            
                new_o = rebuild_function(parent_pseudo)
                
                return new_o
            
            else:
            
                print("Couldn't interpret \"" + class_name + "\" as class type.")
                exit(0)
        
        else:
        
            # assume this must be a artist or a conceiver
        
            new_o = {}
        
            for key in pseudo:
            
                try: fixed_key = int(key)
                except ValueError: fixed_key = key
            
                new_o[fixed_key] = interpret_as_object(pseudo[key], pseudo)
                
            return new_o
            
            
    elif type(pseudo).__name__ == "list":
    
        return list(map(lambda x,y=parent_pseudo: interpret_as_object(x,y), pseudo))
            
    else:
    
        return pseudo
    
def load_object(path):

    if not os.path.isfile(path):
        print("Could not find file at \""+path+"\".")
        exit(0)

    with open(path) as infile: text = infile.read()

    return interpret_as_object(json.loads(text), None)
                 
def save_object(obj, path):

    dumped = json.dumps(obj, indent=4, sort_keys=False, cls=SimpleEncoder)

    with open(path, "w") as outf: outf.write(dumped)
                
def validate_data():

    global ARTISTS
    global VERSIONS
    
    ARTISTS = {}
    VERSIONS = {}
    
    for subdirectory, subfolders, fnames in os.walk(SETTINGS["ARTIST DIRECTORY"]):
    
        for fname in fnames:
        
            full_path = os.path.join(subdirectory, fname)

            if os.path.isfile(full_path):
            
                m = re.fullmatch("(.+?)\\s*\\.json", fname.strip(), flags=re.IGNORECASE)
    
                if m:
                
                    artist_name = m.group(1)
                    
                    if not is_valid_scathe_name(artist_name):
                        print("Invalid artist name in data folder: \""+artist_name+"\".\n")
                        exit(0)
                        
                    if artist_name in ARTISTS:
                        print("Multiple files with same artist name: \""+artist_name+"\".\n")
                        exit(0)
                        
                    ARTISTS[artist_name] = full_path
    
    for subdirectory, subfolders, fnames in os.walk(SETTINGS["VERSION DIRECTORY"]):
    
        for fname in fnames:
        
            full_path = os.path.join(subdirectory, fname)

            if os.path.isfile(full_path):
            
                m = re.fullmatch("(.+?)\\s*\\.json", fname.strip(), flags=re.IGNORECASE)
    
                if m:
                
                    version_pair = m.group(1)
                    
                    if not is_valid_version_pair(version_pair):
                        print("Invalid version name in data folder: \""+version_pair+"\".\n")
                        exit(0)
                        
                    if version_pair in VERSIONS:
                        print("Multiple files with same artist & version name: \""+version_pair+"\".\n")
                        exit(0)
                        
                    VERSIONS[version_pair] = full_path
      
      
      
### I/O: RETROFIT

def is_legit(cor):

    if cor.element_0 == "SIGMOID" and cor.param_index_0 == 0: return False
    if cor.element_1 == "SIGMOID" and cor.param_index_1 == 0: return False
    return True

    
    
def retrofit():

    for artist_name, artist_path in ARTISTS.items():
    
        if "core" not in artist_path: continue
    
        artist = load_object(artist_path)
        
        xs = []
        ys = []
        
        for tope in artist.values():
        
            if tope.element == "X": xs.append(int(tope.ID))
            if tope.element == "Y": ys.append(int(tope.ID))
            
        for tope in artist.values():
        
            if tope.element == "IT":
            
                chosens = [random.choice(xs), random.choice(ys)]
            
                for i in range(len(tope.options)):
                    tope.options[i] = chosens + tope.options[i]
            
                for i in range(len(tope.defaults)):
                    tope.defaults[i] = chosens + tope.defaults[i]
        
        save_object(artist, "mgmt/temp/artists/"+artist_name+".json")
        
    # for artver, version_path in VERSIONS.items():
    
        # version = load_object(version_path)
        
        # conceiver = version.conceiver
        
        # try:
        
            # for eldict in conceiver.values():
            
                # for element, concept in eldict.items():
                
                    # if element != "SIGMOID": continue
                    
                    # del concept.inverse_chance
                    
        # except:
        
            # continue
                
        ##
        
        # correlator = version.correlator
        
        # version.correlator = [cor for cor in version.correlator if is_legit(cor)]
        
        # for cor in correlator:
        
            # if cor.element_0 == "SIGMOID": cor.param_index_0 -= 1
            # if cor.element_1 == "SIGMOID": cor.param_index_1 -= 1
        
        # save_object(version, "mgmt/temp/versions/"+artver+".json")
        
    exit(0)
    
    
    
### COMMAND PROMPT: GENERATE

def command_image_specified(ev, command):
        
    artist_name, version_name = command.replace("_"," ").split()

    if artist_name in ARTISTS:
        artist_path = ARTISTS[artist_name]
    else:
        print("\nArtist does not exist.\n")
        return True, False, False, False
        
    version_pair = "_".join([artist_name, version_name])
    if version_pair in VERSIONS:
        version_path = VERSIONS[version_pair]
    else:
        print("\nVersion does not exist.\n")
        return True, False, False, False

    ev.artist = load_object(artist_path)
    ev.artist_name = artist_name
    ev.version = load_object(version_path)
    ev.version_name = version_name
    ev.ptych = ev.ptych_name = None

    return False, SETTINGS["DEFAULT RESOLUTION"], True, []
    
def command_image_again(ev):
        
    ev.artist, ev.artist_name = cached_object("artist")
    ev.version, ev.version_name = cached_object("version")
    ev.ptych = ev.ptych_name = None
    
    if ev.artist == None:
        print("\nCached artist not found.\n")
        
    if ev.version == None:
        print("\nCached version not found.\n")
        return True, None, None, None
    
    if ev.artist_name in ARTISTS:
    
        if ev.artist_name+"_"+ev.version_name in VERSIONS: save_options = []
        else:
            save_options = ["VERSION"]
            ev.version_name = None
        
    else:
        save_options = ["BOTH"]
        ev.artist_name = ev.version_name = None

    return False, SETTINGS["DEFAULT RESOLUTION"], True, save_options
    
def command_image_halfrandom(ev):
        
    ev.artist, ev.artist_name = cached_object("artist")
    
    if ev.artist == None:
        print("\nCached artist not found.\n")
        return True, None, None, None
        
    if ev.artist_name not in ARTISTS:
        print("\nLast used artist not saved.\n")
        return True, None, None, None
        
    usable_versions = [t for t in list(VERSIONS) if t.startswith(ev.artist_name+"_")]
    
    if len(usable_versions) == 0:
        print("\nNo versions saved for last used artist.\n")
        return True, None, None, None
        
    version_name = random.choice(usable_versions)
    version_path = VERSIONS[version_name]

    ev.version_name = version_name.split("_")[1]
    ev.version = load_object(version_path)
    
    ev.ptych = ev.ptych_name = None
    
    return False, SETTINGS["DEFAULT RESOLUTION"], True, []
    
def command_image_random(ev):

    version_pair = random.choice(list(VERSIONS))
    artist_name, version_name = version_pair.split("_")
    artist_path = ARTISTS[artist_name]
    version_path = VERSIONS[version_pair]

    ev.artist = load_object(artist_path)
    ev.artist_name = artist_name
    ev.version = load_object(version_path)
    ev.version_name = version_name
    ev.ptych = ev.ptych_name = None
    
    return False, SETTINGS["DEFAULT RESOLUTION"], True, []
    
def command_image_specified_multi(ev, command):

    global AUTO
            
    AUTO.on = True
    
    command_words = command.replace("_"," ").split()
    
    num_rounds = int(command_words[3])
    AUTO.commands = [" ".join(command_words[:2])] * num_rounds
    AUTO.index = -1
    AUTO.save_n = ["image"] * num_rounds
    
    print()
    
def command_image_random_multi(ev, command):

    global AUTO
            
    AUTO.on = True
    num_rounds = int(command.split()[2])
    AUTO.index = -1
    AUTO.commands = [".."] * num_rounds
    AUTO.save_n = ["image"] * num_rounds
    
    print()
    
    return True, False, False, False

def command_version_again(ev):
        
    ev.artist, ev.artist_name = cached_object("artist")
    
    if ev.artist == None:
        print("\nCached artist not found.\n")
        return True, False, False, False
        
    if ev.artist_name in ARTISTS: save_options = ["VERSION"]
    else:
        save_options = ["BOTH"]
        ev.artist_name = None
        
    # try: ev.version = find_version(ev.artist, False, ev.artist_name, None)
    # except KeyboardInterrupt:
        # print("\nSearch interrupted.\n")
        
    ev.version = find_version(ev.artist, False, ev.artist_name, None)
    if ev.version == True:
        print("\nSearch interrupted.\n")
        return True, False, False, False
    
    ev.version_name = None
    ev.ptych = ev.ptych_name = None
    
    return False, SETTINGS["DEFAULT RESOLUTION"], True, save_options
    
def command_version_named(ev, command):
        
    if command in ARTISTS: artist_path = ARTISTS[command]
    else:
        print("Artist does not exist.\n")
        return True, False, False, False

    ev.artist = load_object(artist_path)
    ev.artist_name = command
    
    # try: ev.version = find_version(ev.artist, False, ev.artist_name, None)
    # except KeyboardInterrupt:
        # print("\nSearch interrupted.\n")
        # return True, False, False, False
    ev.version = find_version(ev.artist, False, ev.artist_name, None)
    if ev.version == True:
        print("\nSearch interrupted.\n")
        return True, False, False, False
    
    ev.version_name = None
    ev.ptych = ev.ptych_name = None

    return False, SETTINGS["DEFAULT RESOLUTION"], True, ["VERSION"]
    
def command_version_random(ev, hungry_artists):
        
    version_found = False
    
    while not version_found:
        
        artist_name = random.choice(hungry_artists)
        artist_path = ARTISTS[artist_name]
        ev.artist = load_object(artist_path)
        ev.artist_name = artist_name
            
        ev.version = find_version(ev.artist, False, ev.artist_name, SETTINGS["NEW VERSION NUM VERSIONS"])
        
        if ev.version == None:
            print("\nTrying another artist.")
            continue
        
        elif ev.version == True:
            print("\nSearch interrupted.\n")
            return True, False, False, False
            
        else:
            version_found = True
            
        if AUTO.on: AUTO.save_n[AUTO.index] = "image "+ev.artist_name+"_"+AUTO.save_n[AUTO.index]
        
        ev.version_name = None
        ev.ptych = ev.ptych_name = None
        

    return False, SETTINGS["DEFAULT RESOLUTION"], True, ["VERSION"]
    
def command_version_random_multi(ev, command):

    global AUTO
            
    num_rounds = int(command.split()[2])
    
    AUTO.on = True
    AUTO.commands = ["!!"] * num_rounds
    AUTO.index = -1
    
    timestamp = datetime.datetime.now().strftime("%m%d%H%M%S")[::-1]
    
    AUTO.save_s = [str(i+1)+"v"+timestamp for i in range(num_rounds)]
    AUTO.save_n = [str(i+1)+"v"+timestamp for i in range(num_rounds)]
    
    print()
    
    return [s for s in ARTISTS if len([t for t in VERSIONS if t.startswith(s+"_")]) < SETTINGS["ARTIST SATIATION"]]
    
def command_version_named_multi(ev, command):

    global AUTO
            
    artist_name = command.split()[0]
    num_rounds = int(command.split()[2])
    
    AUTO.on = True
    AUTO.commands = [artist_name] * num_rounds
    AUTO.index = -1
    
    timestamp = datetime.datetime.now().strftime("%m%d%H%M%S")[::-1]
    
    AUTO.save_s = [str(i+1)+"v"+timestamp for i in range(num_rounds)]
    AUTO.save_n = [str(i+1)+"v"+timestamp for i in range(num_rounds)]
    
    print()
    
def command_artist_random(ev):
        
    # try: ev.artist, ev.version = find_artist_and_version()
    
    # except KeyboardInterrupt:
        # print("\nSearch interrupted.\n")
        # return True, False, False, False
    ev.artist, ev.version = find_artist_and_version()
    if ev.artist == True:
        print("\nSearch interrupted.\n")
        return True, False, False, False
    
    ev.artist_name = ev.version_name = None
    ev.ptych = ev.ptych_name = None

    return False, SETTINGS["DEFAULT RESOLUTION"], True, ["BOTH"]
    
def command_artist_random_multi(ev, command):

    global AUTO
            
    num_rounds = int(command.split()[2])
    
    AUTO.on = True
    AUTO.commands = ["??"] * num_rounds
    AUTO.index = -1
    
    timestamp = datetime.datetime.now().strftime("%m%d%H%M%S")[::-1]
    
    AUTO.save_s = [str(i+1)+"a"+timestamp+" alpha" for i in range(num_rounds)]
    AUTO.save_n = ["image "+str(i+1)+"a"+timestamp+"_alpha" for i in range(num_rounds)]
    
    print()
    
    return
        
def command_rerender_last(ev, command):
        
    ev.artist = ev.artist_name = None
    ev.version = ev.version_name = None
    ev.ptych, ev.ptych_name = cached_object("ptych")
    
    if ev.ptych == None:
        print("\nCached ptych not found.\n")
        return True, False, False, False
    
    return False, int(command.split()[1]), False, []
    
def command_rerender_named(ev, command):
        
    ptych_name, resolution_str = command.split()[1:]

    ptych_path = os.path.join(SETTINGS["PTYCH DIRECTORY"], ptych_name+".json")
    if not os.path.isfile(ptych_path):
        print("\nPtych does not exist.\n")
        return True, False, False, False

    ev.artist = ev.artist_name = None
    ev.version = ev.version_name = None
    ev.ptych = load_object(ptych_path)
    ev.ptych_name = ptych_name
    
    return False, int(resolution_str), False, []
    
def command_rerender_all(ev, command):

    global AUTO
        
    resolution = int(command.split()[2])
    
    ptych_names = directory_set(SETTINGS["RERENDER DIRECTORY"], True, ".png")
    
    AUTO.on = True
    AUTO.index = -1
    AUTO.commands = ["\" "+ptych_name+" "+str(resolution) for ptych_name in ptych_names]
            
    return
    
def prompt_generate():
    
    global AUTO
    
    ev = Everything()
    
    
    hungry_artists = [s for s in ARTISTS if len([t for t in VERSIONS if t.startswith(s+"_")]) < SETTINGS["ARTIST SATIATION"]]
    
    while True:
    
        
        if AUTO.on:
        
            AUTO.index += 1
            
            if AUTO.index < len(AUTO.commands):
                command = AUTO.commands[AUTO.index]
            
            else:
            
                AUTO.on = False
                
        else:
        
            hungry_artists = [s for s in ARTISTS if len([t for t in VERSIONS if t.startswith(s+"_")]) < SETTINGS["ARTIST SATIATION"]]
    
        print("~~~~~~~~~~~~")
                
        if AUTO.on:
            print("\nBulk command: "+command+" [#"+str(AUTO.index+1)+" of "+str(len(AUTO.commands))+"]")
        else:
            command = input("\nGenerate: ").strip()
    
        load_settings()
        
        validate_data()    
            
        if re.fullmatch("#+", command):
            print("\n~~~~~~~~~~~~\n\nGoodbye.\n\n")
            exit(0)
            
        # (blank)
        elif re.fullmatch("", command):
            cont, resolution, newing, save_options = command_image_again(ev)
            if cont: continue
            
        # .
        elif re.fullmatch("\\.", command):
            cont, resolution, newing, save_options = command_image_halfrandom(ev)
            if cont: continue
            
        # ..
        elif re.fullmatch("\\.\\.", command):
            cont, resolution, newing, save_options = command_image_random(ev)
            if cont: continue
            
        # .. * 123
        elif re.fullmatch("\\.\\. \\* \\d+", command):
            cont, resolution, newing, save_options = command_image_random_multi(ev, command)
            if cont: continue
            
        # ??
        elif re.fullmatch("\\?\\?", command):
            cont, resolution, newing, save_options = command_artist_random(ev)
            if cont: continue
            
        # !!
        elif re.fullmatch("!!", command):
            cont, resolution, newing, save_options = command_version_random(ev, hungry_artists)
            if cont: continue
            
        # !! * 123
        elif re.fullmatch("!! \\* \\d+", command):
            hungry_artists = command_version_random_multi(ev, command)
            continue
            
        # ?? * 123
        elif re.fullmatch("\\?\\? \\* \\d+", command):
            command_artist_random_multi(ev, command)
            continue
            
        # !
        elif re.fullmatch("!", command):
            cont, resolution, newing, save_options = command_version_again(ev)
            if cont: continue
            
        # abc def
        elif re.fullmatch("\\w+(\\s|_)+\\w+", command):
            cont, resolution, newing, save_options = command_image_specified(ev, command)
            if cont: continue
            
        # abc def * 123
        elif re.fullmatch("\\w+(\\s|_)+\\w+\\s+\\*\\s+\\d+", command):
            command_image_specified_multi(ev, command)
            continue
            
        # abc
        elif re.fullmatch("\\S+", command):
            cont, resolution, newing, save_options = command_version_named(ev, command)
            if cont: continue
            
        # abc * 123
        elif re.fullmatch("\\w+\\s+\\*\\s+\\d+", command):
            command_version_named_multi(ev, command)
            continue
            
        # " 123
        elif re.fullmatch("\"\\s+\\d+", command):
            cont, resolution, newing, save_options = command_rerender_last(ev, command)
            if cont: continue
            
        ## " " 123
        elif re.fullmatch("\"\\s+\"\\s+\\d+", command):
            command_rerender_all(ev, command)
            continue
            
        ## " abc 123
        elif re.fullmatch("\"\\s+\\S+\\s+\\d+", command):
            cont, resolution, newing, save_options = command_rerender_named(ev, command)
            if cont: continue
            
        else:
            print("Improper command.\n")
            continue
        
        if not newing:
            old_resolution = max(ev.ptych.canvas_width, ev.ptych.canvas_height)
            ev.ptych.canvas_width = int(ev.ptych.canvas_width*resolution/old_resolution)
            ev.ptych.canvas_height = int(ev.ptych.canvas_height*resolution/old_resolution)
            
        return ev, resolution, newing, save_options
  


### COMMAND PROMPT: OTHER

def prompt_save_scathe(save_options, artist_name):

    global AUTO

    if len(save_options) == 0: return None, None

    forbidden_artist_names = ARTISTS.keys()
    forbidden_version_names = VERSIONS.keys()

    option_dict = {"ARTIST": "artist", "VERSION": "version", "BOTH": "both"}
    
    text_options = list(map(option_dict.get, save_options))
    
    if not AUTO.on:
    
        if save_options[0] == "VERSION": print("\nEnter the <version name> to save the version.\n")
        elif save_options[0] == "BOTH": print("\nEnter the <artist name> <version name> to save the artist.\n")
    
        # if len(save_options) == 1: prompt = "\nSave "+text_options[0]+"?\n"
        # elif len(save_options) == 2: prompt = "\nSave "+text_options[0]+" or "+text_options[1]+"?\n"
        # else: prompt = "\nSave "+text_options[0]+", "+text_options[1]+", or "+text_optons[2]+"?\n"
        
        # print(prompt)
    
    
    while True:
            
        if AUTO.on:
            command = AUTO.save_s[AUTO.index]
                
        if not AUTO.on:
            command = input("Save: ").strip()
        
        ## "#"
        m = re.fullmatch("#+", command)
        if m:
            print("\n~~~~~~~~~~~~\n\nGoodbye.\n\n")
            exit(0)
        
        ## "-"
        m = re.fullmatch("-+|`+", command)
        if m: return None, None
            
            
        if "ARTIST" in save_options:
        
            m = re.fullmatch("\\s*(\\S*\\D\\S*)", command, flags=re.IGNORECASE)
            
            if m:
            
                save_name = m.group(1)
                
                if not is_valid_scathe_name(save_name):
                    print("\nArtist name must be a single word without _, !, <, >, :, \", /, \\, |, ?, or *.\n")
                    continue
            
                if save_name in forbidden_artist_names:
                    print("\nName already used for another artist.\n")
                    continue
                    
                else: return save_name, None
            
            
        if "VERSION" in save_options:
        
            m = re.fullmatch("\\s*(\\S+)\\s*", command, flags=re.IGNORECASE)
            
            
            
            if m:
            
                save_name = m.group(1)
                
                if not is_valid_scathe_name(save_name):
                    print("\nVersion name must be a single word without _, !, <, >, :, \", /, \\, |, ?, or *.\n")
                    continue
                    
                if artist_name+"_"+save_name in forbidden_version_names:
                    print("\nName already used for another version of this artist.\n")
                    continue
                    
                else: return None, artist_name+"_"+save_name
            
            
        if "BOTH" in save_options:
        
            m = re.fullmatch("\\s*(\\S+)\\s+(\\S+)\\s*", command, flags=re.IGNORECASE)
            
            if m:
            
                save_sname = m.group(1)
                save_thname = m.group(2)
                
                if not (is_valid_scathe_name(save_sname) and is_valid_scathe_name(save_thname)):
                    print("\nArtist and version names must be a single word without _, !, <, >, :, \", /, \\, |, ?, or *.\n")
                    continue
            
                if save_sname in forbidden_artist_names:
                    print("Name already used for another artist.\n")
                    continue
                    
                if save_sname+"_"+save_thname in forbidden_version_names:
                    print("\nName already used for another version of an artist.\n")
                    continue
                    
                else: return save_sname, save_sname+"_"+save_thname
            
            
        print("Improper command.\n")
     
def prompt_save_new_image(timestamp, old_resolution):

    global AUTO

    forbidden_image_names = directory_set(SETTINGS["IMAGE DIRECTORY"], True, ".png")|directory_set(SETTINGS["PTYCH DIRECTORY"], False, ".json")
    
    default_name = timestamp
    
    
    if not AUTO.on: print("Save the image?\n")


    while True:
        
        if AUTO.on:
            command = AUTO.save_n[AUTO.index]
            
        if not AUTO.on:
            command = input("Save: ").strip()
        
        ## "#"
        m = re.fullmatch("#+", command)
        if m:
            print("\n~~~~~~~~~~~~\n\nGoodbye.\n\n")
            exit(0)
        
        
        ## "-"
        m = re.fullmatch("-+", command)
        if m: return None, None, False
        
        
        ## "`"
        m = re.fullmatch("`+", command)
        if m: return None, None, True
        
        
        ## "image"
        m = re.fullmatch("image", command, flags=re.IGNORECASE)
        if m: return default_name, None, False
        
        
        ## "image XX 123"
        m = re.fullmatch("image\\s+(\\S+)\\s+(\\d+)", command, flags=re.IGNORECASE)
        
        if m:
        
            if m.group(1) in forbidden_image_names:
                print("\nName already used for another image/ptych.")
                continue
                
            else:
                if int(m.group(2)) == old_resolution: return os.path.join(SETTINGS["IMAGE DIRECTORY"], m.group(1)), None, False
                else: return m.group(1), int(m.group(2)), False
            
            
        ## "image XX" or "image 123"
        m = re.fullmatch("image\\s+(\\S+)", command, flags=re.IGNORECASE)
        
        if m:
        
            try: return default_name, int(m.group(1)), False
            
            except ValueError:
            
                if m.group(1) in forbidden_image_names:
                    print("\nName already used for another image/ptych.")
                    continue
                    
                else: return m.group(1), None, False
                
            
        print("Improper command.\n")
 
def save_scathe(ev, save_options):

    global ARTISTS
    global VERSIONS


    save_artist_name, save_version_name = prompt_save_scathe(save_options, ev.artist_name)
    
    # if save_artist_name != None or save_version_name != None: print()
    
    ## save artist
    if save_artist_name != None:
    
        save_artist_path = os.path.join(SETTINGS["ARTIST DIRECTORY"], save_artist_name+".json")
        
        save_object(ev.artist, save_artist_path)
        
        ARTISTS[save_artist_name] = save_artist_path
        
        print("Saved artist as "+save_artist_name+".json.")
        
        update_cache_name("artist", save_artist_name+".json")
        
        if SETTINGS["EXPORT ARTIST GRAPHS"]:
            with open(os.path.join(SETTINGS["GRAPH DIRECTORY"], save_artist_name+".tgf"), "w") as outf: outf.write(artist_to_graph(ev.artist))
            print("Saved artist graph as "+save_artist_name+".tgf.")
        
        
    ## save version
    if save_version_name != None:
    
        save_version_path = os.path.join(SETTINGS["VERSION DIRECTORY"], save_version_name+".json")
        
        save_object(ev.version, save_version_path)
        
        VERSIONS[save_version_name] = save_version_path
 
        print("Saved version as "+save_version_name+".json.")
        
        update_cache_name("version", save_version_name.split("_")[1]+".json")
        
def save_new_image(ptych, resolution, timestamp, cached_image_path):
            
            
    save_image_name, new_resolution, save_nothing = prompt_save_new_image(timestamp, resolution)
    
    
    if save_image_name != None:
    
        save_image_path = os.path.join(SETTINGS["IMAGE DIRECTORY"], save_image_name+".png")
    
        ## do save, use the render just performed
        if new_resolution == None:
        
            copy_file(cached_image_path, save_image_path)
            
        ## do save, after re-rendering at another resolution
        elif new_resolution != None:
        
            new_ptych = copy.deepcopy(ptych)
            new_ptych.canvas_width = int(new_ptych.canvas_width*new_resolution/resolution)
            new_ptych.canvas_height = int(new_ptych.canvas_height*new_resolution/resolution)
            
            art_bmp.create(render_ptych(new_ptych), save_image_path)
            
        print("Saved image as "+save_image_name+".png.")
        
        save_object(ptych, os.path.join(SETTINGS["PTYCH DIRECTORY"], save_image_name+".json"))
        print("Saved ptych as "+save_image_name+".json.")
        
        if SETTINGS["EXPORT IMAGE FORMULAS"]:
            with open(os.path.join(SETTINGS["GRAPH DIRECTORY"], save_image_name+".txt"), "w") as outf: outf.write(ptych_to_text(ptych))
            print("Saved image formula as "+save_image_name+".txt.")
            
        if SETTINGS["EXPORT IMAGE GRAPHS"]:
            with open(os.path.join(SETTINGS["GRAPH DIRECTORY"], save_image_name+".tgf"), "w") as outf: outf.write(ptych_to_graph(ptych))
            print("Saved image graph as "+save_image_name+".tgf.")
            
        update_cache_name("image", save_image_name+".png")
        update_cache_name("ptych", save_image_name+".json")
        
    return save_nothing
              
def save_old_image(ev, resolution, timestamp, cached_image_path):
    
    if ev.ptych_name == None: save_image_name = timestamp
    else: save_image_name = ev.ptych_name
    
    ptych_filename = save_image_name+".json"
    image_filename = save_image_name+" "+str(resolution)+".png"
    image_cache_filename = save_image_name+".png"
    
    copy_file(cached_image_path, os.path.join(SETTINGS["IMAGE DIRECTORY"], image_filename))
    
    save_object(ev.ptych, os.path.join(SETTINGS["PTYCH DIRECTORY"], ptych_filename))
    
    update_cache_name("image", image_cache_filename)
    
    print("Saved image as "+image_filename+".")
    
def announce_panels(artist_name, version_name, num_panels):

    sname = artist_name if artist_name != None else "(unnamed artist)"
    thname = version_name if version_name != None else "(unnamed version)"
    
    word = "panels" if num_panels > 1 else "panel"
    
    print("\nCommissioning ~ "+sname+" ~ "+thname+" ~ for "+str(num_panels)+" "+word+".")
     
def main_loop():

    global SEEDER
    global AUTO

    while True:
    
        SEEDER.painting_index = None

        ev, resolution, newing, save_options = prompt_generate()
        
        if ev.ptych == None:
        
            canvas_width, canvas_height = random.sample((resolution, int(resolution / random.uniform(1, 2))), k = 2)
            
            num_panels = choose_num_paintings()
            
            generate_seeds(num_panels)
        
            ev.ptych = Ptych(canvas_width, canvas_height, arrange_ptych(num_panels, canvas_width, canvas_height))
            ev.ptych.info = {"artist": ev.artist_name, "version": ev.version_name}
            
            controlled_artist = apply_controller(ev.version.controller, ev.artist)
            
            announce_panels(ev.artist_name, ev.version_name, num_panels)
            
            for SEEDER.painting_index in range(num_panels):
                
                SEEDER.seed_ptyndex = 0
                
                palettes = generate_palettes(ev.version.prism, ev.version.permuter, ev.version.pasteller, ev.version.puddle, ev.version.procession)
                borders = generate_borders(palettes[0].base_colors)
                glow = generate_glow(palettes[0].base_colors)
                
                thicket, roots = grow_trim_thicket(controlled_artist, ev.version)
                # thicket, roots = manual_thicket()
                
                whirl = Whirl(palettes, thicket, roots, apply_calmer(ev.version.calmer))
                
                painting = Painting(whirl, glow, borders)
                
                synchronize_borders(painting, num_panels, ev.ptych)
                
                ev.ptych.paintings.append(painting)
            
        image = render_ptych(ev.ptych)
        if image == None: print("Rendering interrupted.")
                
        timestamp, cached_image_path = update_cache_full(image, ev)
                
        if newing:
            save_nothing = image != None and save_new_image(ev.ptych, resolution, timestamp, cached_image_path)
            if not save_nothing: save_scathe(ev, save_options)
        
        elif image != None:
            save_old_image(ev, resolution, timestamp, cached_image_path)
            
        print("")

def main():

    print("\n\nWelcome.\n")

    initialize_elements()

    load_settings()
    
    validate_data()
    
    # retrofit()
    
    main_loop()



### GO

try:

    try:
    
        main()
        
        # import cProfile
        # cProfile.run("main()")
    
    except Exception as e:
        time.sleep(0.01)
        print("\nart.py crashed:\n\n  "+repr(e)+"\n")
        traceback.print_tb(e.__traceback__)
        input("\nPress enter to quit.")
        print("")
        
except KeyboardInterrupt as e:
    print("\n\nart.py forced to stop by KeyboardInterrupt.")
    input("\nPress enter to quit.")
    print("")