import random
import numpy as np
from numba import njit
import math
from tpg.utils import flip
import pdb

"""
A program that is executed to help obtain the bid for a learner.
"""
class Program:

    """
    bits for:
    mode   op    dest       src
    11      1111   11111...   11111111111...
    Mode:  2 bit, whether to use register, input or memory.
    Op: Max 4 bits, one of 9 operations (add, sub, mult, div, cos, log,
        exp, neg, write).
    Dest: At-least # of bits to store # of registers. The register to place the
        result of the instruction into.
    Src: At-least # of bits to store size of input. The index to take from
        input, or a register depending on Mode.
    """
    instructionLengths   = [2,4,3,23]


    idCount = 0 # unique id of each program

    def __init__(self, instructions=None, maxProgramLength=128):
        if instructions is not None: # copy from existing
            self.instructions = list(instructions)
        else: # create random new
            self.instructions = [self.createInstruction() for _ in
                            range(random.randint(1, maxProgramLength))]
        self.instructionList = getOperationList()
        self.id = Program.idCount
        Program.idCount += 1


        self.update()


    """
    Executes the program which returns a single final value.
    """
    def execute(self, state, rootMem, regs, modes, ops, dsts, srcs):

        regSize = len(regs)
        # append rootMem to end of inpt
        inpt = state.copy()
        inpt = list(inpt)
        mem = rootMem.flatten()
        inptLen = len(inpt)
        memSize = len(mem)
        # iterate through instructions
        for i in range(len(modes)):
            # first get source
            if modes[i] == 0:
                src = regs[srcs[i]%regSize]
            elif modes[i] == 1:
                src = inpt[srcs[i]%inptLen]
            elif modes[i] == 2:
                src = mem[srcs[i]%memSize]
            else:
                raise Exception('Mode not set correctly')

            # do operation
            op = ops[i]
            x = regs[dsts[i]]
            y = src
            dest = dsts[i]%regSize
            # run instructions
            self.instructionList[op](regs, dest, x, y, rootMem)
            # put new mem into inpt
            for i, memVal in enumerate(reversed(mem)):
                inpt[-i] = memVal


            if math.isnan(regs[dest]):
                regs[dest] = 0
            elif regs[dest] == np.inf:
                regs[dest] = np.finfo(np.float64).max
            elif regs[dest] == np.NINF:
                regs[dest] = np.finfo(np.float64).min

    """
    Takes instructions and converts them into np arrays for easier more
    efficient execution.
    """
    def update(self):
        totalLen = sum(Program.instructionLengths)
        instsData = np.array([
            [
                getIntSegment(inst, 0, Program.instructionLengths[0], totalLen),
                getIntSegment(inst, Program.instructionLengths[0],
                        Program.instructionLengths[1], totalLen),
                getIntSegment(inst, sum(Program.instructionLengths[:2]),
                        Program.instructionLengths[2], totalLen),
                getIntSegment(inst, sum(Program.instructionLengths[:3]),
                        Program.instructionLengths[3], totalLen)
            ]
            for inst in self.instructions])

        self.modes = np.array(instsData[:,0], dtype = np.int8)
        self.operations = np.array(instsData[:,1], dtype = np.int8)
        self.destinations = np.array(instsData[:,2], dtype = np.int8)
        self.sources = np.array(instsData[:,3], dtype = np.int32)


    """
    Mutates the program, by performing some operations on the instructions. If
    inpts, and outs (parallel) not None, then mutates until this program is
    distinct. If update then calls update when done.
    """
    def mutate(self, pMutRep, pDelInst, pAddInst, pSwpInst, pMutInst,
                regSize, uniqueProgThresh, inputs=None, outputs=None, update=True,
                maxMuts=100):
        if inputs is not None and outputs is not None:
            # mutate until distinct from others
            unique = False
            while not unique:
                if maxMuts <= 0:
                    break # too much
                maxMuts -= 1

                unique = True # assume unique until shown not
                self.mutateInstructions(pDelInst, pAddInst, pSwpInst, pMutInst)
                self.update()
                # check unique on all inputs from all learners outputs
                # input and outputs of i'th learner
                for i, lrnrInputs in enumerate(inputs):
                    lrnrOutputs = outputs[i]

                    for j, input in enumerate(lrnrInputs):
                        output = lrnrOutputs[j]
                        regs = np.zeros(regSize)
                        Program.execute(input, regs, self.modes, self.operations,
                                        self.destinations, self.sources)
                        myOut = regs[0]
                        if abs(output-myOut) < uniqueProgThresh:
                            unique = False
                            break

                    if unique == False:
                        break
        else:
            # mutations repeatedly, random amount
            mutated = False
            while not mutated or flip(pMutRep):
                self.mutateInstructions(pDelInst, pAddInst, pSwpInst, pMutInst)
                mutated = True

        if update:
            self.update()

    """
    Potentially modifies the instructions in a few ways.
    """
    def mutateInstructions(self, pDel, pAdd, pSwp, pMut):
        changed = False

        while not changed:
            # maybe delete instruction
            if len(self.instructions) > 1 and flip(pDel):
                del self.instructions[random.randint(0, len(self.instructions)-1)]
                changed = True

            # maybe mutate an instruction (flip a bit)
            if flip(pMut):
                idx = random.randint(0, len(self.instructions)-1)
                num = self.instructions[idx]
                totalLen = sum(Program.instructionLengths)
                while True:
                    bit = random.randint(0, totalLen-1)
                    newInst = bitFlip(num, bit, totalLen)
                    # check that new op int is valid, else choose bit again
                    if  self.checkInstructionValid(newInst):
                        self.instructions[idx] = newInst
                        break

                changed = True

            # maybe swap two instructions
            if len(self.instructions) > 1 and flip(pSwp):
                # indices to swap
                idx1, idx2 = random.sample(range(len(self.instructions)), 2)
                # do swap
                tmp = self.instructions[idx1]
                self.instructions[idx1] = self.instructions[idx2]
                self.instructions[idx2] = tmp
                changed = True

            # maybe add instruction
            if flip(pAdd):
                newInst = self.createInstruction()
                self.instructions.insert(
                            random.randint(0, len(self.instructions)-1),
                           newInst )
                changed = True

    """
    Creates a new randomly generated instruction
    """
    def createInstruction(self):
        maxInst = 2**sum(Program.instructionLengths)-1

        while True:
            newInst = random.randint(0, maxInst)
            # check op bits are valid
            if self.checkInstructionValid(newInst):
                return newInst

    def checkInstructionValid(self, instruction):
        totalLen = sum(Program.instructionLengths)
        opsCorrect = getIntSegment(instruction, Program.instructionLengths[0],Program.instructionLengths[1], totalLen) < len(getOperationList())
        modeCorrect = getIntSegment(instruction, 0,Program.instructionLengths[0], totalLen) < 3
        return opsCorrect and modeCorrect

"""
Takes an int and returns another int made of some bits of the original.
"""
def getIntSegment(num, bitStart, bitLen, totalLen):
    binStr = format(num, 'b').zfill(totalLen)
    return int(binStr[bitStart:bitStart+bitLen], 2)

"""
Flip a bit in the provided int.
"""
def bitFlip(num, bit, totalLen):
    binStr = format(num, 'b').zfill(totalLen)

    if binStr[bit] == '0':
        newNum = int(binStr[:bit] + '1' + binStr[bit+1:], 2)
    else:
        newNum = int(binStr[:bit] + '0' + binStr[bit+1:], 2)

    return newNum

def getOperationList():
    return [instAdd, instAdd, instMul, instDiv, instCos, instLog, instExp, instCond, instWrite ]

@njit
def instAdd(regs, dest, x, y, mem):
    regs[dest] = x+y

@njit
def instAdd(regs, dest, x, y, mem):
    regs[dest] = x-y

@njit
def instMul(regs, dest, x, y, mem):
    regs[dest] = x*y

@njit
def instDiv(regs, dest, x, y, mem):
    if y != 0:
        regs[dest] = x/y

@njit
def instCos(regs, dest, x, y, mem):
    regs[dest] = math.cos(y)

@njit
def instLog(regs, dest, x, y, mem):
    if y > 0:
        regs[dest] = math.log(y)

@njit
def instExp(regs, dest, x, y, mem):
    regs[dest] = math.exp(y)

@njit
def instCond(regs, dest, x, y, mem):
    if x < y:
        regs[dest] = x*(-1)

@njit
def instWrite(regs, dest, x, y, mem):
    mid = mem.shape[0] // 2
    for offset in range(0,mid):
        pWrite = 0.25 - (0.01*offset)**2
        for registryIndex in range(len(regs)):
            if random.random() <= pWrite:
                mem[mid + offset][registryIndex] = regs[registryIndex]
            if random.random() <= pWrite:
                mem[mid - offset][registryIndex] = regs[registryIndex]

def updateInpt(rootMem, inpt):
    for i, memVal in enumerate(reversed(inpt)):
        rootMem[-i] = memVal
