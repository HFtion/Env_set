
import os
import sys
sys.path.append(os.getcwd())

print(os.getcwd())

print(__name__)
print(__file__)

print(os.path.dirname(__file__))
print(os.path.basename(__file__))

from glob import glob

source = './t1/'
files = os.listdir(source)
f1 = glob(source+'[0-9]?.x')



