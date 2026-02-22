import os
from PIL import Image

def bmp_pixel(pixel):
    r, g, b = pixel
    return (b).to_bytes(1,"little")+(g).to_bytes(1,"little")+(r).to_bytes(1,"little")

def write_BMP_header(outf,h,w,r):
  outf.write(b'BM')
  outf.write((54+(h*(w*3+r))).to_bytes(4,"little"))
  outf.write(b'\x00\x00')
  outf.write(b'\x00\x00')
  outf.write((54).to_bytes(4,"little"))
  
def write_DIB_header(outf,h,w,r):
  outf.write((40).to_bytes(4,"little"))
  outf.write((w).to_bytes(4,"little"))
  outf.write((h).to_bytes(4,"little"))
  outf.write((1).to_bytes(2,"little"))
  outf.write((24).to_bytes(2,"little"))
  outf.write((0).to_bytes(4,"little"))
  outf.write((h*(w*3+r)).to_bytes(4,"little"))
  outf.write((2835).to_bytes(4,"little"))
  outf.write((2835).to_bytes(4,"little"))
  outf.write((0).to_bytes(4,"little"))
  outf.write((0).to_bytes(4,"little"))
  
def write_array(outf,pixels,h,r):
  for i in range(h-1,-1,-1):
    for j in pixels[i]:
      outf.write(bmp_pixel(j))
    outf.write(b'\x00'*r)
  
def create(pixels, title):
  height = len(pixels)
  width = len(pixels[0])
  row_padding = (4-((width*3)%4))%4
  print("Converting to PNG.\n")
  with open("~tmp.bmp","wb") as outfile:
    write_BMP_header(outfile,height,width,row_padding)
    write_DIB_header(outfile,height,width,row_padding)
    write_array(outfile,pixels,height,row_padding)
  with Image.open("~tmp.bmp") as bmp:
    bmp.save(title, "PNG")
  os.remove("~tmp.bmp")