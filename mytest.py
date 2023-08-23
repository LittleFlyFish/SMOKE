import os
import csv
import logging
import random
import numpy as np
from PIL import Image


img = Image.open('/home/soe/Documents/000608.png')

img = img.transform(
       (1280, 540),
       method=Image.AFFINE,
       resample=Image.BILINEAR,
       )

