from tpg.utils import flip
from tpg.learner import Learner
import random
import numpy as np
import pdb
"""
The main building block of TPG. Each team has multiple learning which decide the
action to take in the graph.
"""
class Team:

    idCount = 0

    def __init__(self, memDim):
        self.learners = []
        self.outcomes = {} # scores at various tasks
        self.fitness = None
        self.numLearnersReferencing = 0 # number of learners that reference this
        self.id = Team.idCount
        Team.idCount += 1
        self.mem = np.zeros(memDim)

    """
    Returns an action to use based on the current state.
    """
    def act(self, state, rootMem = [],  visited=set()):

        # If this is a root team, then use it's own memory
        if len(rootMem) == 0 and self.numLearnersReferencing == 0:
            rootMem = self.mem
        elif len(rootMem) == 0 and self.numLearnersReferencing != 0 :
            pdb.set_trace()
            raise Exception('Non-root team not supplied with root mem')
        elif len(rootMem) != 0 and self.numLearnersReferencing == 0:
            pdb.set_trace()
            raise Exception('root team supplied with root mem')


        visited.add(self) # track visited teams

        topLearner = max([lrnr for lrnr in self.learners
                if lrnr.isActionAtomic() or lrnr.action not in visited],
            key=lambda lrnr: lrnr.bid(state, rootMem))

        return topLearner.getAction(state, rootMem, visited=visited)

    """
    Adds learner to the team and updates number of references to that program.
    """
    def addLearner(self, learner=None):
        program = learner.program
        # don't add duplicate program
        if any([lrnr.program == program for lrnr in self.learners]):
            return False

        self.learners.append(learner)
        learner.numTeamsReferencing += 1

        return True

    """
    Removes learner from the team and updates number of references to that program.
    """
    def removeLearner(self, learner):
        if learner in self.learners:
            learner.numTeamsReferencing -= 1
            self.learners.remove(learner)

    """
    Bulk removes learners from teams.
    """
    def removeLearners(self):
        for lrnr in list(self.learners):
            self.removeLearner(lrnr)

    """
    Number of learners with atomic actions on this team.
    """
    def numAtomicActions(self):
        num = 0
        for lrnr in self.learners:
            if lrnr.isActionAtomic():
                num += 1

        return num

    """
    Mutates the learner set of this team.
    """
    def mutate(self, pDelLrn, pAddLrn, pMutLrn, allLearners,
                pMutProg, pMutAct, pActAtom, atomics, allTeams,
                pDelInst, pAddInst, pSwpInst, pMutInst,
                multiActs, pSwapMultiAct, pChangeMultiAct,
                uniqueProgThresh, inputs=None, outputs=None, update=True):

        # delete some learners
        p = pDelLrn
        while flip(p) and len(self.learners) > 2: # must have >= 2 learners
            p *= pDelLrn # decrease next chance

            # choose non-atomic learners if only one atomic remaining
            learner = random.choice([l for l in self.learners
                                     if not l.isActionAtomic()
                                        or self.numAtomicActions() > 1])
            self.removeLearner(learner)

        # add some learners
        p = pAddLrn
        while flip(p):
            p *= pAddLrn # decrease next chance
            #import pdb; pdb.set_trace()
            learner = random.choice([l for l in allLearners
                                     if l not in self.learners and
                                        l.action is not self])
            self.addLearner(learner)

        # give chance to mutate all learners
        oLearners = list(self.learners)
        for learner in oLearners:
            if flip(pMutLrn):
                if self.numAtomicActions() == 1 and learner.isActionAtomic():
                    pActAtom0 = 1 # action must be kept atomic if only one
                else:
                    pActAtom0 = pActAtom

                # must remove then re-add fresh mutated learner
                self.removeLearner(learner)
                newLearner = Learner(learner=learner)
                newLearner.mutate(
                        pMutProg, pMutAct, pActAtom0, atomics, self, allTeams,
                        pDelInst, pAddInst, pSwpInst, pMutInst,
                        multiActs, pSwapMultiAct, pChangeMultiAct,
                        uniqueProgThresh, inputs=inputs, outputs=outputs, update=update)
                self.addLearner(newLearner)
