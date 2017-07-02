#!/usr/bin/env python3

# Author: RA
#         incl parts by RV


C = 3
N = 6
I = 1000


import tensorflow as tf

# Disable / Restore sys.stderr(..)
import sys, os
def stop_stderr(): sys.stderr = open(os.devnull, 'w')
def open_stderr(): sys.stderr = sys.__stderr__

stop_stderr()
from keras import backend as keras_backend
open_stderr()


import main
import random
import matplotlib.pyplot as plt
import numpy as np

from keras.models       import Sequential, load_model
from keras.optimizers   import Adam
from keras import regularizers
from keras.layers       import Dense, Dropout, Merge

from main import mean

from AI_RV import AI_RV
from AI_CB import AI_CB

import pickle


# Initialisation des donnés

with open('pre_world.pkl', 'rb') as infile :
    wrd = pickle.load(infile)
    assert (type(wrd) is main.World)
    assert ((wrd.C == C) and (wrd.N == N))

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


class AI_OFF:
    """
    AI class
    """
    name = "ReinLear"
    
    def __init__(self, C, N):
        self.C = C
        self.N = N
        self.f_model = None
        self.model = None
        self.hist = None
        self.init_model()
    
    def init_model(self) :
        try :
            self.model = load_model('out_nn.h5')
            print("ReinLear load model OK")
        except :
            print("ReinLear building model")
            # model = Sequential()
            # merged = Merge([self.s_model,self.m_model], mode='concat')
            # model.add(merged)
            # model.add(Dense(self.N + self.C, kernel_initializer='normal', activation='softmax'))
            # self.model = model
            model = Sequential()
            model.add(Dense(64, input_dim=(self.C + self.N**2 + 1), 
                                activation='relu', kernel_initializer='normal', 
                                kernel_regularizer=regularizers.l2(0.00001)))
            model.add(Dense(128, activation='relu', kernel_initializer='normal', 
                                 kernel_regularizer=regularizers.l2(0.00001)))
            # model.add(Dense(254, activation='relu', kernel_initializer='normal', 
            #                     kernel_regularizer=regularizers.l2(0.00000)))
            model.add(Dense(128, activation='relu', kernel_initializer='normal', 
                                kernel_regularizer=regularizers.l2(0.00001)))
            model.add(Dense(64, activation='relu', kernel_initializer='normal', 
                                kernel_regularizer=regularizers.l2(0.00001)))
            # model.add(Dense(128, activation='relu', kernel_initializer='uniform', 
            #                     kernel_regularizer=regularizers.l2(0.000001)))
            # model.add(Dense(128, activation='relu', kernel_initializer='uniform', 
            #                     kernel_regularizer=regularizers.l2(0.000001)))
            model.add(Dense(self.N + self.C, kernel_initializer='normal', activation='softmax'))
            self.model = model                     

    def step(self, b, B, Q, spre):
        """
        Note: b, B, Q are OK to modify
        """
        
        M0 = list(range(len(Q[b])))
        M = []
        s = 0
        while True:
            # Current state
            C = Matricize(self.C, self.N, b, B, Q)
            X = np.asarray([[spre] + list(C.BB) + list(C.QQ.flatten())])
            
            # Suggestion by NN: what to do?
            t = np.argmax(self.model.predict(X))
            
            # We are moving
            if t < 3 :
               s = [0, 1, -1] [t]
               break

            # Cannot board any more people: move forward
            if (len(B) >= self.C) : break
            
            # We are boarding
            t = (t + b - 3) % self.N
            # Check if the suggestion is valid
            if (not t in Q[b]) : 
                print("Oops")
                break
            else :
                print("OK")
            m = Q[b].index(t)
                
            # Collect original index
            M.append(M0.pop(m))
            # Remove from the (virtual) queue, and board onto (virtual) bus
            B.append(Q[b].pop(m))
        #print(b, B0, B, s)
        
        # Print the stations
        #Qb = [q[:] for q in Q]; Qb[b].append('b'); print(B, Qb, s)
        
        return M, s
    
    
    def curriculum(self, wrd, nav_teacher, XB) :
        self.train_RL(XB)
    
    def train_RL(self, XB) :
        mod = self.model
        
        with open('pre_cbw.txt', 'rb') as infile :
            cbw = pickle.load(infile)
            
        with open("pre_Xfeed.txt", "rb") as ft:   # Unpickling
            X = pickle.load(ft)
    
        with open("pre_Yfeed.txt", "rb") as fr:   # Unpickling
            Y = pickle.load(fr)
        
        with open("out_acc.txt", "rb") as fp:   # Unpickling
            acc = pickle.load(fp)
    
        with open("out_vacc.txt", "rb") as fo:   # Unpickling
            vacc = pickle.load(fo)
            
        with open("out_loss.txt", "rb") as fp:   # Unpickling
            loss = pickle.load(fp)
    
        with open("out_vloss.txt", "rb") as fo:   # Unpickling
            vloss = pickle.load(fo)
                    
        with open("out_Epps.txt", "rb") as fe:
            epps = pickle.load(fe)
                    
        for j in range(10):
            
            nepoch = j*10
            
            ai = Profiler(wrd,AI_OFF(3,6))
            
            mod.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
            
            
            self.hist = mod.fit(X, Y, epochs=10, batch_size=20, verbose=2, validation_split=0)
            
            
            acc[nepoch] = self.hist.history['acc'][0]
            loss[nepoch] = self.hist.history['loss'][0]
            # vacc[nepoch - 10] = self.hist.history['val_acc'][0]
            # vloss[nepoch - 10] = self.hist.history['val_loss'][0]
            epps[nepoch] = ai.w
            
            plt.plot(range(I), cbw[0:I], 'r-')
            c = str(round(mean(cbw[0:I])))
            plt.xlabel('Iterations')
            plt.ylabel('People waiting per station')
            # plt.text(10, 3, r'$\mu = ' + c +'$')
            plt.show()
            
            plt.plot(range(I),ai.W)
            c=str(round(ai.w,2))
            plt.xlabel('Iterations')
            plt.ylabel('People waiting per station')
            # plt.text(20, 4, r'$\mu = ' + c +'$')
            plt.show()
            
            mod.save('out_nn.h5')
            
        with open("out_acc.txt", "wb") as fp:   #Pickling
            pickle.dump(acc, fp)
            
        with open("out_vacc.txt", "wb") as fo:   #Pickling
            pickle.dump(vacc, fo)
            
        with open("out_loss.txt", "wb") as fp:   #Pickling
            pickle.dump(loss, fp)
            
        with open("out_vloss.txt", "wb") as fo:   #Pickling
            pickle.dump(vloss, fo)
                    
        with open("out_Epps.txt", "wb") as fe:   #Pickling
            pickle.dump(epps, fe)
            
        
        # mod.save('out_nn.h5')

        

class Profiler:
    """
    Runs the systems with a particular strategy "nav".
    """
    
    # Number of iterations (time steps)
    # This will be I ~ 1e6
    
    def __init__(self, wrd, nav):
        self.I = I
        # W[i] = average number of people waiting at time i
        self.W = []
        # w = average over time
        self.w = None
        self.l = [[],[]]
        
        assert (0 < self.I <= 1e9)
        
        wrd.rewind()
        assert (wrd.i == 0)
        spre = 0
        
        # Main loop
        while wrd.i < self.I:
            M,s = (nav.step(*wrd.look(),spre))
            self.l[0].append(M)
            self.l[1].append(s)
            wrd.move(*nav.step(*wrd.look(),spre))
            spre = s
            #print(s)
            self.W.append(wrd.get_w())

        assert len(self.W)
        self.w = mean(self.W)


def main_entry_train():
 
    random.seed(-1)
    nav_teacher = AI_RV(C, N)
    nav_learner = AI_OFF(C, N)
    nav_learner.curriculum(wrd, nav_teacher, XB)

    #report = Trainer(wrd, nav)

from timeit import timeit

if (__name__ == "__main__"):
    #tf_sess = tf.Session()
    #keras_backend.set_session(tf_sess)

    #t = timeit(main_entry, number=1)
    t = timeit(main_entry_train, number=1)
    print("Time:", t, "(sec)")
    
# Stats

# RV learning

# 


def drawacc():
    with open('out_acc.txt', 'rb') as infile :
        acc = pickle.load(infile)
    list0 = sorted(acc.items())
    x,y = zip(*list0)
    plt.plot(x, y, 'o--')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    c = str(round(y[-1],4))
    plt.text(700,0.3, ' last value = ' + c )
    plt.show() 
    
def drawloss():
    with open('out_loss.txt', 'rb') as infile :
        loss = pickle.load(infile)
    list0 = sorted(loss.items())
    x,y = zip(*list0)
    plt.plot(x, y, 'o--')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    c = str(round(y[-1],4))
    plt.text(700,0.5, ' last value = ' + c )
    plt.show()
    
def drawvacc():
    with open('out_vacc.txt', 'rb') as infile :
        vacc = pickle.load(infile)
    list0 = sorted(vacc.items())
    x,y = zip(*list0)
    plt.plot(x, y, 'o--')
    plt.xlabel('Epochs')
    plt.ylabel('Validation Accuracy')
    c = str(round(y[-1],4))
    plt.text(700,0.9, ' last value = ' + c )
    plt.show()
    
def drawvloss():
    with open('out_vloss.txt', 'rb') as infile :
        vloss = pickle.load(infile)
    list0 = sorted(vloss.items())
    x,y = zip(*list0)
    plt.plot(x, y, 'o--')
    plt.xlabel('Epochs')
    plt.ylabel('Validation Loss')
    c = str(round(y[-1],4))
    plt.text(700,0.6, ' last value = ' + c )
    plt.show()
    
def drawepps():
    with open('out_Epps.txt', 'rb') as infile :
        epps = pickle.load(infile) 
    list0 = sorted(epps.items())
    x,y = zip(*list0)
    plt.plot(x, y, 'o--')
    plt.xlabel('Epochs')
    plt.ylabel('E[pps]')
    c = str(round(y[-1],4))
    plt.text(700,75, ' last value = ' + c )
    d = str(round(min(y),4))
    plt.text(700,65, ' min value = ' + d )
    plt.show()
    
    

