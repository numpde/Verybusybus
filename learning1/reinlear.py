#!/usr/bin/env python3

# Author: RA
#         incl parts by RV

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

import pickle

C = 3
N = 6
# 
# with open('world.pkl', 'rb') as infile :
#     wrd = pickle.load(infile)
#     #assert (type(wrd) is main.World)
#     assert ((wrd.C == C) and (wrd.N == N))

class AI_GREEDY1:
    """
    'Modestly greedy' strategy
    """
    name = "Modestly greedy'"
    
    def __init__(self, C, N):
        self.C = C
        self.N = N
        self.s = +1
    
    def step(self, b, B, Q):
        """
        Calculates one step.
        """
        # Number of passengers to board
        n = min(len(Q[b]), self.C - len(B))
        # Passenger selection from Q[b]
        M = list(range(n))
        
        # No passengers? 
        if (not B) and (not M):
            return [], +1
        
        # Next passenger's destination
        if len(B):
            t = B[0]
        else:
            t = Q[b][M[0]]
        
        # Destination relative to the current position
        t = self.N - 2 * ((t - b + self.N) % self.N)
        
        # Move towards that destination (modify default direction)
        self.s = (+1) if (t > 0) else (-1)
        
        return M, self.s

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


class AI_MY:
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
        # try :
        #     self.f_model = load_model('NN0-f.h5')
        #     print("ReinLear load f_model OK")
        # except :
        #     print("ReinLear building f_model")
        #     f_model = Sequential()
        #     f_model.add(Dense(64, input_dim=(self.C + self.N**2), 
        #                         activation='relu', kernel_initializer='uniform', 
        #                         kernel_regularizer=regularizers.l2(0.00001)))
        #     f_model.add(Dense(128, 
        #                         activation='relu', kernel_initializer='uniform', 
        #                         kernel_regularizer=regularizers.l2(0.00001)))
        #     f_model.add(Dense(64, 
        #                         activation='relu', kernel_initializer='uniform', 
        #                         kernel_regularizer=regularizers.l2(0.00001)))
        #     f_model.add(Dense(32, 
        #                         activation='relu', kernel_initializer='uniform', 
        #                         kernel_regularizer=regularizers.l2(0.00001)))
        #     f_model.add(Dense(16, 
        #                         activation='relu', kernel_initializer='uniform', 
        #                         kernel_regularizer=regularizers.l2(0.00001)))
        #     f_model.add(Dense(32,  
        #                         activation='relu', kernel_initializer='uniform', 
        #                         kernel_regularizer=regularizers.l2(0.01)))
        #     f_model.add(Dense(16, 
        #                         activation='relu', kernel_initializer='uniform', 
        #                         kernel_regularizer=regularizers.l2(0.01)))
        #     f_model.add(Dense(8, 
        #                         activation='relu', kernel_initializer='uniform', 
        #                         kernel_regularizer=regularizers.l2(0.01)))
        #     f_model.add(Dense(1, kernel_initializer='normal', activation='sigmoid'))
        #     self.f_model = f_model

        #   try :
        #     self.s_model = load_model('NN0-s.h5')
        #     print("ReinLear load s_model OK")
        # except:
        #     print("ReinLear building s_model")
        #     s_model = Sequential()
        #     s_model.add(Dense(70, input_dim=(self.C + self.N**2), 
        #                         activation='relu', kernel_initializer='uniform', 
        #                         kernel_regularizer=regularizers.l2(0.00001)))
        #     s_model.add(Dense(70, activation='relu', kernel_initializer='uniform', 
        #                         kernel_regularizer=regularizers.l2(0.00001)))
        #     s_model.add(Dense(70, activation='relu', kernel_initializer='uniform', 
        #                         kernel_regularizer=regularizers.l2(0.00001)))
        #     s_model.add(Dense(70, activation='relu', kernel_initializer='uniform', 
        #                         kernel_regularizer=regularizers.l2(0.00001)))
        #     '''s_model.add(Dense(55, activation='relu', kernel_initializer='uniform', 
        #                         kernel_regularizer=regularizers.l2(0.000001)))
        #     s_model.add(Dense(55, activation='relu', kernel_initializer='uniform', 
        #                         kernel_regularizer=regularizers.l2(0.000001)))
        #     s_model.add(Dense(55, activation='relu', kernel_initializer='uniform', 
        #                         kernel_regularizer=regularizers.l2(0.000001)))
        #     s_model.add(Dense(55, activation='relu', kernel_initializer='uniform', 
        #                         kernel_regularizer=regularizers.l2(0.000001)))'''
        #     s_model.add(Dense(3, kernel_initializer='normal', activation='softmax'))
        #     self.s_model = s_model
        #     
        # try :
        #     self.m_model = load_model('NN0-m.h5')
        #     print("ReinLear load m_model OK")
        # except :
        #     print("ReinLear building m_model")
        #     m_model = Sequential()
        #     m_model.add(self.s_model) 
        #     m_model.add(Dense(64, input_dim=(self.C + self.N**2), 
        #                         activation='relu', kernel_initializer='uniform', 
        #                         kernel_regularizer=regularizers.l2(0.00001)))
        #     m_model.add(Dense(128, activation='relu', kernel_initializer='uniform', 
        #                         kernel_regularizer=regularizers.l2(0.00001)))
        #     m_model.add(Dense(256, activation='relu', kernel_initializer='uniform', 
        #                         kernel_regularizer=regularizers.l2(0.00001)))
        #     m_model.add(Dense(128, activation='relu', kernel_initializer='uniform', 
        #                         kernel_regularizer=regularizers.l2(0.00001)))
        #     m_model.add(Dense(64, activation='relu', kernel_initializer='uniform', 
        #                         kernel_regularizer=regularizers.l2(0.00001)))
        #     m_model.add(Dense(50, activation='relu', kernel_initializer='uniform', 
        #                         kernel_regularizer=regularizers.l2(0.00000)))
        #     m_model.add(Dense(50, activation='relu', kernel_initializer='uniform', 
        #                         kernel_regularizer=regularizers.l2(0.00000)))
        #     m_model.add(Dense(self.N, kernel_initializer='glorot_normal', activation='softmax'))
        #     self.m_model = m_model
        try :
            self.model = load_model('NN0.h5')
            print("ReinLear load model OK")
        except :
            print("ReinLear building model")
            # model = Sequential()
            # merged = Merge([self.s_model,self.m_model], mode='concat')
            # model.add(merged)
            # model.add(Dense(self.N + self.C, kernel_initializer='normal', activation='softmax'))
            # self.model = model
            model = Sequential()
            model.add(Dense(64, input_dim=(self.C + self.N**2), 
                                activation='relu', kernel_initializer='normal', 
                                kernel_regularizer=regularizers.l2(0.00000)))
            model.add(Dense(128, activation='relu', kernel_initializer='normal', 
                                kernel_regularizer=regularizers.l2(0.00000)))
            # model.add(Dense(254, activation='relu', kernel_initializer='normal', 
            #                     kernel_regularizer=regularizers.l2(0.00000)))
            model.add(Dense(128, activation='relu', kernel_initializer='normal', 
                                kernel_regularizer=regularizers.l2(0.00000)))
            model.add(Dense(64, activation='relu', kernel_initializer='normal', 
                                kernel_regularizer=regularizers.l2(0.00000)))
            # model.add(Dense(128, activation='relu', kernel_initializer='uniform', 
            #                     kernel_regularizer=regularizers.l2(0.000001)))
            # model.add(Dense(128, activation='relu', kernel_initializer='uniform', 
            #                     kernel_regularizer=regularizers.l2(0.000001)))
            model.add(Dense(self.N + self.C, kernel_initializer='normal', activation='softmax'))
            self.model = model                     

    def step(self, b, B, Q):
        """
        Note: b, B, Q are OK to modify
        """
        
        Q[b].sort()
        M0 = list(range(len(Q[b])))
        M = []
        s = 0
        while True:
            # Current state
            C = Matricize(self.C, self.N, b, B, Q)
            X = np.asarray([list(C.BB) + list(C.QQ.flatten())])
            
            # Suggestion by NN: move forward?
            # f = self.f_model.predict(X)
            
            # DEBUG
            #(M1, _) = self.greedy.step(b, B, Q)
            #print((len(M1) > 0), " should agree with ", (f < 0.5))
            
            #print("f = {}".format(f))
            
            # f_model suggests to move forward
            #if (f > 0.5) : break
            
            # Get index of the next person to board
            
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
                
            # Suggestion by AI_GREEDY
            #(M1, _) = self.greedy.step(b, B, Q)
            # AI_GREEDY does not want to board anyone
            #if (not M1) : break
            #m = M1[0] # With greedy, always m==0
                
            #print(b, Q[b], m)
            #print(np.round(self.m_model.predict(X), decimals=2), C.QQ[0], t, m1)
                
            # Collect original index
            M.append(M0.pop(m))
            # Remove from the (virtual) queue, and board onto (virtual) bus
            B.append(Q[b].pop(m))
        
        # Suggestion by AI_GREEDY
        # Ask greedy where to go with the new B, Q
        #(_, s) = self.greedy.step(b, B, Q)
        
        #print(b, B0, B, s)
        
        # Print the stations
        #Qb = [q[:] for q in Q]; Qb[b].append('b'); print(B, Qb, s)
        
        return M, s
    
    def lesson(self, wrd_look, nav_teacher) :
        #print(wrd_look, R)
        pass
    
    def curriculum(self, wrd, nav_teacher, XB) :
        # self.train_f(XB)
        # self.train_s(XB)
        # self.train_m(XB)
        self.train_RL(XB)
    
    def train_f(self, XB) :
        # Construct a neural network "f_model"
        # that decides whether to aquire passengers
        # or to allow the bus to move
        
        model = self.f_model
        
        X = []
        Y = []
        print(XB.X)
        
        for i in range(len(XB.X['S'])) :
            (S, A, R) = (XB.X['S'][i], XB.X['A'][i], XB.X['R'][i])
            
            C = Matricize(self.C, self.N, S['b'], S['B'], S['Q'])
            
            X.append(list(C.BB) + list(C.QQ.flatten()))
            Y.append(0 + A['f'])
            #print(X)
            #print(Y)
            
            
        
        X = np.asarray(X)
        Y = np.asarray(Y)
    
        #print(X.shape)
        #print(Y.shape)
        
        #for i in range(len(X)) : print(X[i], Y[i])
        
        assert (model is not None)
        model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
        res = model.fit(X, Y, epochs=1, batch_size=20, verbose=2, validation_split=0.1)
        
        #model.save('./NN0-f.h5')
        
        
    def train_s(self, XB) :
        model = self.s_model
        
        X = []
        Y = []
        
        for i in range(len(XB.X['S'])) :
            (S, A, R) = (XB.X['S'][i], XB.X['A'][i], XB.X['R'][i])
            ## Not moving forward?
            if (not A['f']) : continue
            
            C = Matricize(self.C, self.N, S['b'], S['B'], S['Q'])
            
            #print(list(C.BB) + list(C.QQ.flatten()))
            X.append(list(C.BB) + list(C.QQ.flatten()))
            Y.append(C.cs(A['s']))
        
        X = np.asarray(X)
        Y = np.asarray(Y)
        #print(X)
        #print(Y)
        
        #print(X[0:1])
        
        #print(Y[0:1])
        #print(model.predict(X[0:1]))
        
        # for i in range(len(X)) : print(X[i], Y[i])
        
        assert (model is not None)
        ## categorical_crossentropy / mse / ... (https://keras.io/objectives/) 
        #model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
        #res = model.fit(X, Y, epochs=50, batch_size=20, verbose=2, validation_split=0.1)

        #model.save('./NN0-s.h5')
        
    def train_m(self, XB) :
        model = self.m_model
        
        X = []
        Y = []
        for i in range(len(XB.X['S'])) :
            (S, A, R) = (XB.X['S'][i], XB.X['A'][i], XB.X['R'][i])
            ## Moving forward?
            if (A['f']) : continue
            
            C = Matricize(self.C, self.N, S['b'], S['B'], S['Q'])
            
            X.append(list(C.BB) + list(C.QQ.flatten()))
            Y.append(C.cm(A['m']))
            
            
        #print(type(X))
        #print(type(Y))
        
        X = np.asarray(X)
        Y = np.asarray(Y)
        #print(Y.shape)
        
        
        # print(X[0:1])
        # print(Y[0:1])
        # print(model.predict(X[0:1]))
        
        for i in range(len(X)) : print(X[i], Y[i])
        
        assert (model is not None)
        ## Optimizer (https://keras.io/optimizers/)
        ##opt = Adam(lr=0.001, beta_1=0.99, beta_2=0.999, epsilon=1e-08, decay=0.0)
        ## loss: categorical_crossentropy / mse / ... (https://keras.io/objectives/)
        #model.compile(optimizer='sgd', loss='categorical_crossentropy', metrics=['accuracy'])
        #model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
        #res = model.fit(X, Y, epochs=1, batch_size=20, verbose=2, validation_split=0.1)
        
        #model.save('./NN0-m.h5')
    
    def train_RL(self, XB) :
        mod = self.model
         
        X = []
        Y = []
        
        with open("test.txt", "rb") as fp:   # Unpickling
            acc = pickle.load(fp)
    
        with open("test2.txt", "rb") as f0:   # Unpickling
            vacc = pickle.load(f0)
        
        for i in range(len(XB.X['S'])) :
            (S, A, R) = (XB.X['S'][i], XB.X['A'][i], XB.X['R'][i])
            
            C = Matricize(self.C, self.N, S['b'], S['B'], S['Q'])
            
            X.append(list(C.BB) + list(C.QQ.flatten()))
            
            # Moving forward?
            if (A['f']) : 
                #print(C.cs(A['s']))
                #print(type(C.cs(A['s'])))
                Y.append(np.concatenate((C.cs(A['s']),np.zeros(self.N))))
        
            # Not moving forward?
            else: 
                Y.append(np.concatenate((np.zeros(self.C),C.cm(A['m']))))
        
        
        X = np.asarray(X)
        Y = np.asarray(Y)
        
        mod.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
                                
        self.hist = mod.fit(X, Y, epochs=10, batch_size=20, verbose=2, validation_split=0.1)
        # print(self.hist.history.keys())
        acc = acc + self.hist.history['acc']
        vacc = vacc + self.hist.history['val_acc']
        
        with open("test.txt", "wb") as fp:   #Pickling
            pickle.dump(acc, fp)
            
        with open("test2.txt", "wb") as f0:   #Pickling
            pickle.dump(vacc, f0)
        
        mod.save('./NN0.h5')

class School :
    def __init__(self, wrd, nav_teacher) :
        self.wrd = wrd
        self.nav_teacher = nav_teacher

    def teach(self, nav_learner, I) :
        assert (0 < I <= 1e6)
        wrd = self.wrd
        XB = ExperienceBuffer(wrd)
        
        # Main loop
        wrd.rewind()
        while (wrd.i < I) :
            wrd.Q[wrd.b].sort()
            w0 = wrd.get_w()
            # Let the learner ask the teacher
            nav_learner.lesson(wrd.look(), self.nav_teacher)
            # Get teacher's response R = (M, s)
            R = self.nav_teacher.step(*XB.RecState(*wrd.look()))
            # NOTE: XB.RecAction returns ([], s) and modifies wrd appropriately
            wrd.move(*XB.RecAction(*R))
            # Reward = decrease in number of people waiting
            XB.RecReward(wrd.get_w() - w0)

        assert (len(XB.X['S']) == len(XB.X['A']))
        assert (len(XB.X['A']) == len(XB.X['R']))
        
        #for a in XB.X['A'] : print(a)
        
        #for _ in range(10) : 
        nav_learner.curriculum(wrd, self.nav_teacher, XB)
        
        #f_model = self.train_f_model(wrd, XB)
        #s_model = self.train_s_model(wrd, XB)
        #m_model = self.train_m_model(wrd, XB)
        
        # return {'NN-f': f_model, 'NN-s': s_model, 'NN-M': m_model}
        

class Profiler:
    """
    Runs the systems with a particular strategy "nav".
    """
    
    # Number of iterations (time steps)
    # This will be I ~ 1e6
    I = 100
    
    def __init__(self, wrd, nav):
        # W[i] = average number of people waiting at time i
        self.W = []
        # w = average over time
        self.w = None
        self.l = [[],[]]
        
        assert (0 < self.I <= 1e9)
        
        wrd.rewind()
        assert (wrd.i == 0)
        
        # Main loop
        while wrd.i < self.I:
            M,s = (nav.step(*wrd.look()))
            self.l[0].append(M)
            self.l[1].append(s)
            wrd.move(*nav.step(*wrd.look()))
            #print(s)
            self.W.append(wrd.get_w())

        assert len(self.W)
        self.w = mean(self.W)
        

def main_entry():
    C = 3
    N = 6
    
    random.seed(-1)
    # wrd = main.World(C, N)
    nav = AI_MY(C, N)
    
    print("Profiling")
    report = Profiler(wrd, nav)
    print("Profiling done")
    

def main_entry_train():
    C = 3
    N = 6
    
    for I in [100]:
        random.seed(-1)
        # wrd = main.World(C, N)
        nav_teacher = AI_RV(C, N)
        nav_learner = AI_MY(C, N)
        school = School(wrd, nav_teacher)
        school.teach(nav_learner, I)

    #report = Trainer(wrd, nav)

from timeit import timeit

if (__name__ == "__main__"):
    #tf_sess = tf.Session()
    #keras_backend.set_session(tf_sess)

    #t = timeit(main_entry, number=1)
    t = timeit(main_entry_train, number=1)
    print("Time:", t, "(sec)")
    
##Stats
# 
# import matplotlib.pyplot as plt
# 
# I = 1000
# 
# 
# 
# plt.plot(range(I), pro.W)
# c=str(pro.w)
# plt.xlabel('Iterations')
# plt.ylabel('People waiting per station')
# plt.text(100, 30, r'$\mu = ' + c +'$')
# plt.show()
# 
# plt.plot(range(I), ai.W)
# c=str(ai.w)
# plt.xlabel('Iterations')
# plt.ylabel('People waiting per station')
# plt.text(100, 4, r'$\mu = ' + c +'$')
# plt.show()
# 
# plt.plot(range(I), rui.W)
# c=str(rui.w)
# plt.xlabel('Iterations')
# plt.ylabel('People waiting per station')
# plt.text(100, 2, r'$\mu = ' + c +'$')
# plt.show()

## CB learning


# # 
# ai = Profiler(wrd,AI_MY(3,6))
# 
# plt.plot(range(I),ai.W)
# c=str(ai.w)
# plt.xlabel('Iterations')
# plt.ylabel('People waiting per station')
# # plt.text(100, 100, r'$\mu = ' + c +'$')
# plt.show()


# cb = Profiler(wrd,AI_CB(3,6))
# 
# plt.plot(range(I),cb.W)
# c=str(cb.w)
# plt.xlabel('Iterations')
# plt.ylabel('People waiting per station')
# plt.text(200, 50, r'$\mu = ' + c +'$')
# plt.show()
