import sys
from cfg import cfg

if len(sys.argv) < 2:
    print ("Usage: need arguments <filename>")
    sys.exit(1)
    
filename = sys.argv[1]

kernel = cfg(filename)

kernel.print()