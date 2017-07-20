import pickle
import numpy as np
from AI_RV import AI_RV

C = 3
N = 6
I = 1000
cbw = []

import main
from main import World

wrd = World(C, N)
for _ in range(I) : wrd.news(); wrd.i += 1
# with open('pre_world.pkl', 'rb') as infile :
#     wrd = pickle.load(infile)
#     assert (type(wrd) is main.World)
#     assert ((wrd.C == C) and (wrd.N == N))
    
# List-to-matrix conversion
# to feed to neural network
class Matricize :
    def __init__(self, C, N, b, B, Q) :
        self.C = C
        self.N = N
        self.b = b
        self.B = B
        self.Q = Q
        self.QQ = self.cQ()
        self.BB = self.cB()
    
    # Position relative to the current bus position
    # (the result is in the interval [0, N-1])
    def circ(self, x) :
        return (x - self.b) % self.N
    
    def cQ(self) :
        # Represent the people waiting in Q in a table QQ
        # Source = location rel to bus position 
        # Target = destination rel to bus position
        # Both in the interval [0, N-1]
        # Current bus location shifted to 0
        # Note: this erases the arrival order in Q.
        # The source-target matrix is defined as:
        # QQ[s][t] is the number of people 
        # waiting at s and aiming to go to t
        # (where 0 is the current bus position)
        QQ = np.zeros((self.N, self.N))
        for (s, q) in enumerate(self.Q):
            for t in q:
                QQ[self.circ(s)][self.circ(t)] += 1
        return QQ

    def cB(self) :
        # Represent the people on the bus as a vector BB
        # Compute the target vector of passengers on the bus:
        # BB[n] = target of n-th passenger
        # (where 0 is the current bus position)
        # Note: 0 is not a valid passenger destination
        BB = np.zeros((self.C,)) 
        for (n, t) in enumerate(self.B):
            BB[n] = self.circ(t)
        return BB

    def cm(self, m) :
        # m is index into Q[b] indicating
        # the person to board the bus.
        # NOTE: people are boarded one by one.
        MM = np.zeros(self.N)
        MM[self.circ(self.Q[self.b][m])] = 1
        return MM

    def cs(self, s) :
        # Direction of travel as a class:
        #   s =  0  =>  ss[0] = 1
        #   s =  1  =>  ss[1] = 1
        #   s = -1  =>  ss[2] = 1
        ss = np.zeros(3)
        ss[s % 3] = 1
        return ss


class ExperienceBuffer :
    def __init__(self, wrd) :
        self.wrd = wrd
        self.X = {'S': [], 'A': [], 'R': []}
    
    # Record state
    def RecState(self, b, B, Q) :
        self.X['S'].append({'b': b, 'B': B, 'Q': Q})
        return b, B, Q

    def _RecAction_Board(self, m) :
        # Finished boarding / Forward = False
        self.X['A'].append({'f': False, 'm': m})
    
    def _RecAction_Drive(self, s) :
        # Finished boarding / Forward = True
        self.X['A'].append({'f': True, 's': s})

    # Record action
    def RecAction(self, M, s) :
        # Board people ONE BY ONE
        for m in sorted(M, reverse=True) : 
            self._RecAction_Board(m)
            self.RecReward(0)
            self.wrd.board1(m)
            self.RecState(*self.wrd.look())

        # Record the intention to move bus
        self._RecAction_Drive(s)

        return [], s
    
        # Record reward
    def RecReward(self, r) :
        self.X['R'].append(r)
        return r
                
XB = ExperienceBuffer(wrd)    
nav_teacher = AI_RV(C,N)

# Main loop
wrd.rewind()
while (wrd.i < I) :
    #wrd.Q[wrd.b].sort()
    
    # Get teacher's response R = (M, s)
    R = nav_teacher.step(*XB.RecState(*wrd.look()))
    # NOTE: XB.RecAction returns ([], s) and modifies wrd appropriately
    wrd.move(*XB.RecAction(*R))

    w0 = wrd.get_w()
    cbw.append(w0)
    
    # Reward = decrease in number of people waiting
    XB.RecReward(wrd.get_w() - w0)

X = []
Y = []
spre = 0
                
for i in range(len(XB.X['S'])) :
    (S, A, R) = (XB.X['S'][i], XB.X['A'][i], XB.X['R'][i])
                
    M = Matricize(C, N, S['b'], S['B'], S['Q'])
        
    X.append([spre] + list(M.BB) + list(M.QQ.flatten()))
            
    # Moving forward?
    if (A['f']): 
        #print(C.cs(A['s']))
        #print(type(C.cs(A['s'])))
        Y.append(np.hstack((M.cs(A['s']),np.zeros(N))))
        spre = np.argmax(M.cs(A['s']))
        if spre == 2:
            spre = -1
        
    # Not moving forward?
    else :
        Y.append(np.hstack((np.zeros(C),M.cm(A['m']))))
            
            
X = np.asarray(X)
Y = np.asarray(Y)

with open('pre_xb.pkl', 'wb') as output:
    pickle.dump(XB, output, pickle.HIGHEST_PROTOCOL) 
                    
with open("pre_Xfeed.dat", "wb") as fu:   #Pickling
    pickle.dump(X, fu)
            
with open("pre_Yfeed.dat", "wb") as fy:   #Pickling
    pickle.dump(Y, fy)

with open("pre_cbw.dat", "wb") as fw:   #Pickling
    pickle.dump(cbw, fw)

with open('pre_world.pkl', 'wb') as output:
    pickle.dump(wrd, output, pickle.HIGHEST_PROTOCOL)

with open("out_acc.dat", "wb") as f : 
    pickle.dump(dict([]), f)

with open("out_loss.dat", "wb") as f : 
    pickle.dump(dict([]), f)
    
with open("out_vacc.dat", "wb") as f : 
    pickle.dump(dict([]), f)
    
with open("out_vloss.dat", "wb") as f : 
    pickle.dump(dict([]), f)
    
with open("out_Epps.dat", "wb") as f :
    pickle.dump(dict([]), f)

