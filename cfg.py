class BasicBlock:
    name = ""
    listInstComp = []
    listInstMem = []
    dictInstComp = dict()
    dictInstMem = dict()
    dictReg = dict()
    def __init__(self, name):
        self.name = name
        pass
    def print(self):
        print("NAME:", self.name)
        
        print("Registers used per thread:")
        for key in sorted(self.dictReg):
            print(f"  {key}: {self.dictReg[key]}")
        
        print("Computation Instructions per thread:")
        for key in sorted(self.dictInstComp):
            print(f"  {key}: {self.dictInstComp[key]}")
            
        print("Memory Instructions per thread:")
        for key in sorted(self.dictInstMem):
            print(f"  {key}: {self.dictInstMem[key]}")

class Kernel:
    version = ""
    target = ""
    address_size = 0
    listBasicBlock = []
    
    def __init__(self):
        pass
    
    def print(self):
        print("version =", self.version)
        print("target =", self.target)
        print("address_size =", self.address_size)
        print("Function blocks nums = {}".format(len(self.listBasicBlock)))
        print()
        for i,b in enumerate(self.listBasicBlock):
            print("#", i+1)
            b.print()
    
    

def cfg(filename):
    with open(filename, 'r') as file:
        kn = Kernel()
        bb = None
        
        for line in file:
            l = line.strip()
            if l == "": continue
            if l.startswith("//"): continue
            ls = l.split()
            
            inst = ls[0]
            
            if inst == ".version":
                kn.version = ls[1]
                continue
            if inst == ".target":
                kn.target = ls[1]
                continue
            if inst == ".address_size":
                kn.address_size = ls[1]
                continue
            if inst == ".visible":  # function start
                bb = BasicBlock(ls[2][:-1])
                continue
            if inst == "}":  # function end
                kn.listBasicBlock.append(bb)
                bb = None
                continue
                
            if bb is None: continue
            # following code are counting instructions inside a function block
            
            if inst == ")" or inst == "{": continue
            if inst == ".param": continue
            if inst == ".reg":  # register allocation for single function single thread
                temp = ls[2]
                numIdx = temp.find("<") + 1
                numEnd = temp.find(">")
                num = int(temp[numIdx:numEnd])
                bb.dictReg[ls[1][1:]] = num
                continue
            if inst[0] == "@":  # branch
                continue
            if inst[0] == "$":  # label
                continue
            
            if inst == "ret;":  # return
                continue
            if inst.startswith("ld") or inst.startswith("st"): # load & store
                bb.listInstMem.append(ls[0])
                if not bb.dictInstMem.get(inst):
                    bb.dictInstMem[inst] = 0
                bb.dictInstMem[inst] += 1
                continue
            
            bb.listInstComp.append(ls[0])
            if not bb.dictInstComp.get(inst):
                bb.dictInstComp[inst] = 0
            bb.dictInstComp[inst] += 1
            
    return kn
            
