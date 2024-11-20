class BasicBlock:
    def __init__(self):
        self.name = ""
        self.isLoop = False
        self.ifJumpToBlock = ""
        self.listInstComp = []
        self.listInstMem = []
        self.dictInstComp = dict()
        self.dictInstMem = dict()
        pass
    def print(self):
        print("NAME:", self.name)
        
        # print(self.listInstComp)
        
        print("Computation Inst:")
        for key in sorted(self.dictInstComp):
            print(f"  {key}: {self.dictInstComp[key]}")
            
        print("Memory Inst:")
        for key in sorted(self.dictInstMem):
            print(f"  {key}: {self.dictInstMem[key]}")
        # if self.ifJumpToBlock!="":
        #     print("Can jump to:", self.ifJumpToBlock)
        if self.isLoop:
            print("IS LOOP!")
        
    def isEmpty(self):
        if self.name == "" and len(self.listInstComp) + len(self.listInstMem) == 0:
            return True
        return False

class Kernel:
    
    def __init__(self):
        self.version = ""
        self.target = ""
        self.address_size = 0
        self.dictReg = dict()
        self.listBasicBlock = []
        pass
    
    def print(self):
        print("version =", self.version)
        print("target =", self.target)
        print("address_size =", self.address_size)
        print("Function blocks nums = {}".format(len(self.listBasicBlock)))
        print("Registers used per thread:")
        for key in sorted(self.dictReg):
            print(f"  {key}: {self.dictReg[key]}")
        print()
        for i,b in enumerate(self.listBasicBlock):
            print("#", i)
            if b is not None:
                b.print()
                print()

    
    

def cfg(filename):
    with open(filename, 'r') as file:
        kn = Kernel()
        bb = None
        state = "beforeReg"
        
        for line in file:
            l = line.strip()
            
            if state == "beforeReg":
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
                    continue
                    
                if inst == ".param": continue
                if inst == ")": continue
                if inst[0] == "{": continue
                if inst == ".reg":  # register allocation for single function single thread
                    state = "afterReg"
                    temp = ls[2]
                    numIdx = temp.find("<") + 1
                    numEnd = temp.find(">")
                    num = int(temp[numIdx:numEnd])
                    kn.dictReg[ls[1][1:]] = num
                    continue
            
            else: 
                
                if l == "": 
                    if bb is not None and not bb.isEmpty():
                        kn.listBasicBlock.append(bb)
                    bb = BasicBlock()
                    continue
                
                ls = l.split()
                
                inst = ls[0]
                
                if inst == ".reg":  # register allocation for single function single thread
                    state = "afterReg"
                    temp = ls[2]
                    numIdx = temp.find("<") + 1
                    numEnd = temp.find(">")
                    num = int(temp[numIdx:numEnd])
                    kn.dictReg[ls[1][1:]] = num
                    continue
                
                if inst[0] == ".":  # not an instruction
                    continue
            
                if inst[0] == "@":  # branch
                    bb.ifJumpToBlock = ls[2][:-1]
                    if bb.ifJumpToBlock == bb.name: # Check loop
                        bb.isLoop = True
                    continue
                
                if inst[0] == "}":  # End
                    return kn
                
                if inst[0] == "$":  # label, new block
                    bb.name=inst[:-1]
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
            
    return 

if __name__ == "__main__":
    
    demoFilePath = "./ptx/vecmulf.ptx"
    
    try:
        kn = cfg(demoFilePath)
        
        kn.print()
    except:
        print(demoFilePath, "not found!")
            
