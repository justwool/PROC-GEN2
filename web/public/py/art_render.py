import time # calculate rendering speed
from sys import exit # nuitka seems to need this

from art_class import *


### GLOBALS

CANVAS_WIDTH = None
CANVAS_HEIGHT = None
RESOLUTION = None

SLOW_MS = None
SLOW_PERIOD = None



### SET GLOBALS

def set_canvas_dimensions(width, height):

    global CANVAS_WIDTH
    global CANVAS_HEIGHT
    global RESOLUTION
    
    CANVAS_WIDTH = width
    CANVAS_HEIGHT = height
    RESOLUTION = max(CANVAS_WIDTH, CANVAS_HEIGHT)
    
def set_render_slowing(slowing, period):

    global SLOW_MS
    global SLOW_PERIOD
    
    SLOW_MS = slowing*period/1000
    SLOW_PERIOD = period
    
    
    
### RENDER
    
def quick_place(height, quicks):

    high_index = 0

    for i in range(len(quicks)):
    
        if quicks[i] > height:
                high_index = i
                break
    
    if high_index == 0:
    
            low_index = -1 % len(quicks)
    
            if height <= quicks[0]:
                amt = (height + (1-quicks[-1])) / (quicks[0] + (1 - quicks[-1]))
                
            else:
                amt = (height - quicks[-1]) / (quicks[0] + (1 - quicks[-1]))
    
    else:
                
        low_index = high_index-1
    
        amt = yti(quicks[low_index], quicks[high_index], height)

    return low_index, high_index, amt
           
def color_from_height(height, pality, gradience, palettes):

    cindex = (pality*len(palettes)) % len(palettes)
    first_pindex = int(cindex)
    second_pindex = math.ceil(cindex) % len(palettes)
    palette_amt = cindex % 1

    first_low_index, first_high_index, first_amt = quick_place(cosoften(height), palettes[first_pindex].quicks)
    second_low_index, second_high_index, second_amt = quick_place(cosoften(height), palettes[second_pindex].quicks)
    
    first_color = HSV_blend(palettes[first_pindex].colors[first_low_index], palettes[first_pindex].colors[first_high_index], restrict(sigmoid(derestrict(first_amt), 0, gradience)))
    second_color = HSV_blend(palettes[second_pindex].colors[second_low_index], palettes[second_pindex].colors[second_high_index], restrict(sigmoid(derestrict(second_amt), 0, gradience)))
    
    return HSV_blend(first_color, second_color, palette_amt)
    
def whirl_pixel(x, y, whirl, tm):
    
    heights = calculate_up(x, y, tm, whirl.roots, {})
    
    for i in (0, 1, 2, 3, 4, 5, 6, 7): heights[i] = unit_bound(heights[i])
    for i in (0, 4, 5, 6, 7): heights[i] = (heights[i]+1)/2
    
    pality = heights[4] * whirl.calm.pal
    
    gradience = heights[5] * whirl.calm.gradience_weight + whirl.calm.gradience_lift
    
    hsv = color_from_height(heights[0], pality, gradience, whirl.palettes)
    
    hsv[0] += heights[1] * 360 * whirl.calm.hue
    hsv[1] += heights[2] * whirl.calm.sat
    hsv[2] += heights[3] * whirl.calm.val
    
    if heights[7] >= whirl.calm.unarm and whirl.calm.unarm != 1:
        low_index, high_index, amt = quick_place(heights[6], whirl.palettes[0].quicks)
        target = HSV_blend(whirl.palettes[0].colors[low_index], whirl.palettes[0].colors[high_index], amt)
        aim = ((heights[7] - whirl.calm.unarm)/(1 - whirl.calm.unarm)) ** whirl.calm.unaim
        hsv = HSV_blend(hsv, target, aim)
    
    return (hsv[0] % 360, cosoften(hsv[1]), cosoften(hsv[2]))
    
def render_whirl(canvas, whirl):
        
    x_min = 0
    x_max = CANVAS_WIDTH-1
    y_min = 0
    y_max = CANVAS_HEIGHT-1
    
    num_pixels_drawn = 0
    
    update_pixels = [int((i/100) * CANVAS_WIDTH * CANVAS_HEIGHT) - 1 for i in range(1, 101)]
    
    tm = Thicketmaster(whirl.thicket, list(range(len(whirl.thicket))))
    
    for y in range(y_min, y_max+1):
                
            for x in range(x_min, x_max+1):
            
                if SLOW_MS >= 15 and num_pixels_drawn % SLOW_PERIOD == 0: time.sleep(SLOW_MS/1000)
                
                if num_pixels_drawn in update_pixels:
                    print(str(1 * (update_pixels.index(num_pixels_drawn)+1))+"%", end="\r")
                    
                canvas[y][x] = HSV_to_RGB(whirl_pixel(x/RESOLUTION, y/RESOLUTION, whirl, tm))
                
                num_pixels_drawn += 1
                
            if check_interrupt(): return True
                
    return False
              
def render_glow(canvas, glow):
    
    if glow.height == 0: return
    
    normal_width = CANVAS_WIDTH/RESOLUTION
    normal_height = CANVAS_HEIGHT/RESOLUTION
    
    x_midpoint = normal_width/2
    y_midpoint = normal_height/2
    
    long_radius = (2 ** 0.5)/2
    short_radius = 1/2 - glow.height

    for x in range(0, CANVAS_WIDTH):
        for y in range(0, CANVAS_HEIGHT):
        
            xr = x/RESOLUTION
            yr = y/RESOLUTION
        
            if glow.is_circular:
            
                radius = max(hypot(xr - x_midpoint, yr - y_midpoint), short_radius)
                
                canvas[y][x] = RGB_blend(canvas[y][x], glow.color, yti(short_radius, long_radius, radius))
            
            else:
        
                wall_dist = min(xr, yr, normal_width-xr, normal_height-yr)
                
                if wall_dist <= (glow.height):
                
                    canvas[y][x] = RGB_blend(canvas[y][x], glow.color, 1 - (wall_dist/glow.height))
    
def render_borders(canvas, borders):

    for border in borders:

        for side in ["NORTH", "SOUTH", "EAST", "WEST"]:

            if side == "NORTH":
                
                for x in range(0, CANVAS_WIDTH):
                
                    height = (border.height + border.amplitude * tame_pow((math.sin(2*math.pi*(x/RESOLUTION)/(border.period))), border.exponent)) * RESOLUTION
                    
                    dheight = int(height)
                    antialias = height % 1
                    
                    for y in range(0, dheight): canvas[y][x] = border.color
                    canvas[dheight][x] = RGB_blend(canvas[dheight][x], border.color, antialias)

            if side == "SOUTH":
                
                for x in range(0, CANVAS_WIDTH):
                
                    height = (border.height + border.amplitude * tame_pow((math.sin(2*math.pi*(x/RESOLUTION)/(border.period))), border.exponent)) * RESOLUTION
                    
                    dheight = int(height)
                    antialias = height % 1
                
                    for y in range(CANVAS_HEIGHT-1, CANVAS_HEIGHT-1-dheight, -1): canvas[y][x] = border.color
                    canvas[CANVAS_HEIGHT-1-dheight][x] = RGB_blend(canvas[CANVAS_HEIGHT-1-dheight][x], border.color, antialias)

            if side == "EAST":
                
                for y in range(0, CANVAS_HEIGHT):
                
                    height = (border.height + border.amplitude * tame_pow((math.sin(2*math.pi*(y/RESOLUTION)/(border.period))), border.exponent)) * RESOLUTION
                    
                    dheight = int(height)
                    antialias = height % 1
                
                    for x in range(CANVAS_WIDTH-1, CANVAS_WIDTH-1-dheight, -1): canvas[y][x] = border.color
                    canvas[y][CANVAS_WIDTH-1-dheight] = RGB_blend(canvas[y][CANVAS_WIDTH-1-dheight], border.color, antialias)

            if side == "WEST":
                
                for y in range(0, CANVAS_HEIGHT):
                
                    height = (border.height + border.amplitude * tame_pow((math.sin(2*math.pi*(y/RESOLUTION)/(border.period))), border.exponent)) * RESOLUTION
                    
                    dheight = int(height)
                    antialias = height % 1
                
                    for x in range(0, dheight): canvas[y][x] = border.color
                    canvas[y][dheight] = RGB_blend(canvas[y][dheight], border.color, antialias)
    
    
    
### PTYCH

def new_canvas(width, height):
        
    return [[(SETTINGS["GAP R"], SETTINGS["GAP G"], SETTINGS["GAP B"]) for x in range(width)] for y in range(height)]
      
def stitch_canvases(canvases, layout):

    gap_width = int(RESOLUTION * SETTINGS["GAP FRACTION"])
    
    megacanvas_width = ((CANVAS_WIDTH + gap_width) * max(layout)) + gap_width
    megacanvas_height = ((CANVAS_HEIGHT + gap_width) * len(layout)) + gap_width
    
    megacanvas = new_canvas(megacanvas_width, megacanvas_height)
    
    canvas_index = 0
    
    for y in range(len(layout)):
    
        for x in range(layout[y]):
        
            local_gap_width = int((megacanvas_width - ((layout[y] * (CANVAS_WIDTH + gap_width)) - gap_width)) / 2)
        
            w_start = local_gap_width + ((CANVAS_WIDTH + gap_width) * x)
            z_start = gap_width + ((CANVAS_HEIGHT + gap_width) * y)
            
            w_end = w_start + CANVAS_WIDTH-1
            z_end = z_start + CANVAS_HEIGHT-1
            
            for z in range(z_start, z_end+1):
            
                megacanvas[z][w_start: w_end+1] = canvases[canvas_index][z - z_start]
        
            canvas_index += 1
            
    return megacanvas
                
def render_ptych(ptych):

    start_time = time.time()
    
    if sum(ptych.layout) != len(ptych.paintings):
        print("Layout is for "+str(sum(ptych.layout))+" paintings but ptych has "+str(len(ptych.paintings))+" paintings.")
        exit(0)
    
    set_canvas_dimensions(ptych.canvas_width, ptych.canvas_height)

    canvases = []

    for i in range(len(ptych.paintings)):

        print("\nRendering panel #" + str(i+1) + " of " + str(len(ptych.paintings)) + ".")
        
        painting = ptych.paintings[i]
        
        fill_descendants(painting.whirl.thicket)
        
        canvas = new_canvas(CANVAS_WIDTH, CANVAS_HEIGHT)

        interrupted = render_whirl(canvas, painting.whirl)
        
        if interrupted: return None
        
        # render_glow(canvas, painting.glow)
        
        # render_borders(canvas, painting.borders)

        canvases.append(canvas)

    final_canvas = stitch_canvases(canvases, ptych.layout)
    
    finish_time = time.time()
    
    megapixels = CANVAS_WIDTH * CANVAS_HEIGHT * len(canvases)/1000000
    
    seconds = finish_time - start_time
    
    print("\n\nRendered " + str(round(megapixels, 1)) + " megapixels in " + str(round(seconds, 1)) + " seconds ("+str(round(megapixels*60/seconds, 1))+" mp/min).")
    
    return final_canvas