import sys
from cfg import cfg

if len(sys.argv) < 2:
    print ("Usage: need arguments <filename>")
    sys.exit(1)
    
filename = sys.argv[1]


kernel = cfg(filename)  # get the code block flow from PTX

kernel.print()

# Get from Hardware Manual
max_wSM = 64   # Hardare limit for kepler+  it is 64

# Get from Device Query
S_zw = 32   # Warp size to schedule a set of threads

ILP = 1 # assume or instruction are dependent. If independent, ILP could be higher.

TLP = max_wSM   # For simpler model