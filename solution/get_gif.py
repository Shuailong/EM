#!/usr/bin/env python
# encoding: utf-8

"""
get_gif.py
 
Created by Shuailong on 2016-03-28.

Util to generate a gif from pngs.

"""

from images2gif import writeGif
from PIL import Image
import os

def main():
    file_names = ['../output_image/'+fn for fn in os.listdir('../output_image/') if fn.startswith('iteration')]
    file_names.sort(key=lambda x: int(x[len('../output_image/iteration'):-len('.png')]))
    images = [Image.open(fn) for fn in file_names]
    writeGif("../output_images/movie.gif", images, duration=0.25)

if __name__ == '__main__':
    main()