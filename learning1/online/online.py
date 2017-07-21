#!/usr/bin/env python3

# Author: Roman Andreev and Alejandro Caicedo
#         incl parts by RV

import tensorflow as tf

# Disable / Restore sys.stderr(..)
import sys, os
def stop_stderr(): sys.stderr = open(os.devnull, 'w')
def open_stderr(): sys.stderr = sys.__stderr__

stop_stderr()
from keras import backend as keras_backend
open_stderr()

import statistics as st
import main
import random
import matplotlib.pyplot as plt
import numpy as np

from keras.models       import Sequential, load_model
from keras.optimizers   import Adam
from keras import regularizers
from keras.layers       import Dense, Dropout, Merge

from main import mean
from main import World

from AI_RV import AI_RV

import pickle

I = 1000
C = 3
N = 6

def acc_and_loss(NN, X, Y) :
    loss = []
    acc  = []
    for (p, q) in zip(Y, NN.predict(X)) :
        loss += [ -sum(a*np.log(b) for (a, b) in zip(p, q) if (b != 0)) ]
        acc  += [ np.argmax(p) == np.argmax(q) ]
    return (np.mean(acc), np.mean(loss))

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


class AI_ON:
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
            self.model = load_model('NN1.h5')
            print("ReinLear load model OK")
        except :
            print("ReinLear building model")
            model = Sequential()
            model.add(Dense(128, input_dim=(self.C + self.N**2+1), 
                                activation='relu', kernel_initializer='uniform', 
                                kernel_regularizer=regularizers.l2(0.000000001)))
            model.add(Dense(128, activation='relu', kernel_initializer='uniform', 
                                kernel_regularizer=regularizers.l2(0.000000001)))
            model.add(Dense(128, activation='relu', kernel_initializer='uniform', 
                                kernel_regularizer=regularizers.l2(0.000000001)))
            model.add(Dense(128, activation='relu', kernel_initializer='uniform', 
                                kernel_regularizer=regularizers.l2(0.000000001)))
            model.add(Dense(128, activation='relu', kernel_initializer='uniform', 
                                kernel_regularizer=regularizers.l2(0.000000001)))
            # model.add(Dense(128, activation='relu', kernel_initializer='uniform', 
            #                     kernel_regularizer=regularizers.l2(0.000001)))
            # model.add(Dense(128, activation='relu', kernel_initializer='uniform', 
            #                     kernel_regularizer=regularizers.l2(0.000001)))
            model.add(Dense(self.N + self.C, kernel_initializer='normal', activation='softmax'))

            model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
            self.model = model

    def step(self, b, B, Q, spre = 0):
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
            # print(self.model.predict(X))
            # We are moving
            if t < 3 :
               s = [0, 1, -1] [t]
               break

            # Cannot board any more people: move forward
            if (len(B) >= self.C) : 
                # Debug
                s = 1
                if spre != 0:
                    s = spre
                # print('paila')
                break
            
            # We are boarding
            t = (t + b - 3) % self.N
            # Check if the suggestion is valid
            if (not t in Q[b]) : 
                # print("Oops")
                break
            # else :
                # print("OK")
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
        # mod = self.model
        
        with open("out_Epps.dat", "rb") as fe:
            epps = pickle.load(fe)
        
        with open("out_acc.dat", "rb") as fp:   # Unpickling
            acc = pickle.load(fp)
    
        with open("out_vacc.dat", "rb") as fo:   # Unpickling
            vacc = pickle.load(fo)
            
        with open("out_loss.dat", "rb") as fp:   # Unpickling
            loss = pickle.load(fp)
    
        with open("out_vloss.dat", "rb") as fo:   # Unpickling
            vloss = pickle.load(fo)

        EPOCH_STEP = 10

        for j in range(10000) :
            print("TRAINING ROUND {}".format(j))

            mod = self.model
            nepoch = (j+1)*EPOCH_STEP
            X = []
            Y = []
            spre = 0

            for i in range(len(XB.X['S'])) :
                (S, A, R) = (XB.X['S'][i], XB.X['A'][i], XB.X['R'][i])

                M = Matricize(C, N, S['b'], S['B'], S['Q'])

                X.append([spre] + list(M.BB) + list(M.QQ.flatten()))

                # Moving forward?
                if A['f'] : 
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
            
            
            if (j == 0) :
                (acc0, loss0) = acc_and_loss(mod, X, Y) #before epoch 0
                ai = Profiler(World(C, N), self)
                epps[0] = ai.w
                acc[0] = acc0
                loss[0] = loss0
                
            self.hist = mod.fit(X, Y, epochs=EPOCH_STEP, batch_size=20, verbose=2, validation_split=0.1)
            (accn, lossn) = acc_and_loss(mod, X, Y) # after training
            
            ai = Profiler(World(C, N), self)
            
            acc[nepoch] = accn
            loss[nepoch] = lossn
            vacc[nepoch - EPOCH_STEP] = self.hist.history['val_acc'][0]
            vloss[nepoch - EPOCH_STEP] = self.hist.history['val_loss'][0]
            epps[nepoch] = ai.w
            # plot(ACC.keys(), ACC.values(), 'o--')
            # acc.append(self.hist.history['acc'][-1])
            # loss.append(self.hist.history['loss'][-1])
            # vacc.append(self.hist.history['val_acc'][-1])
            # vloss.append(self.hist.history['val_loss'][-1])
            
            mod.save('NN1.h5')
            
        with open("out_Epps.dat", "wb") as fe:   #Pickling
            pickle.dump(epps, fe)    
            
        with open("out_acc.dat", "wb") as fp:   #Pickling
            pickle.dump(acc, fp)
            
        with open("out_vacc.dat", "wb") as fo:   #Pickling
            pickle.dump(vacc, fo)
            
        with open("out_loss.dat", "wb") as fp:   #Pickling
            pickle.dump(loss, fp)
            
        with open("out_vloss.dat", "wb") as fo:   #Pickling
            pickle.dump(vloss, fo)
            
            
        mod.save('NN1.h5')

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
            # Get teacher's response R = (M, s)
            R = self.nav_teacher.step(*XB.RecState(*wrd.look()))
            # NOTE: XB.RecAction returns ([], s) and modifies wrd appropriately
            wrd.move(*XB.RecAction(*R))
            # Reward = decrease in number of people waiting
            XB.RecReward(wrd.get_w() - w0)

        assert (len(XB.X['S']) == len(XB.X['A']))
        assert (len(XB.X['A']) == len(XB.X['R']))
        
        nav_learner.curriculum(wrd, self.nav_teacher, XB)
        

class Profiler:
    """
    Runs the systems with a particular strategy "nav".
    """
    
    # Number of iterations (time steps)
    # This will be I ~ 1e6
    # I = I
    
    def __init__(self, wrd, nav):
        self.I = I
        # W[i] = average number of people waiting at time i
        self.W = []
        # w = average over time
        self.w = None
        # self.l = [[],[]]
        
        assert (0 < self.I <= 1e9)
        
        wrd.rewind()
        assert (wrd.i == 0)
        spre = 0
        
        # Main loop
        while wrd.i < self.I:
            try:
                M,s = (nav.step(*wrd.look(),spre))
                wrd.move(*nav.step(*wrd.look(),spre))
            except:
                M,s = (nav.step(*wrd.look()))
                wrd.move(*nav.step(*wrd.look()))
            # self.l[0].append(M)
            # self.l[1].append(s)
            spre = s
            #print(s)
            self.W.append(wrd.get_w())

        assert len(self.W)
        self.w = mean(self.W)


def main_entry_train():
    # C = 3
    # N = 6
    # I = 1000
    
    random.seed(-1)
    wrd = main.World(C, N)
    nav_teacher = AI_RV(C, N)
    nav_learner = AI_ON(C, N)
    school = School(wrd, nav_teacher)
    school.teach(nav_learner, I)


def show_save_close(filename) :
    plt.show()
    plt.savefig(filename + '.eps')
    plt.savefig(filename + '.png')
    plt.close()


def drawex():
    rui = Profiler(World(C,N), AI_RV(C, N))
    plt.plot(range(len(rui.W)),rui.W)
    c=str(round(ai.w,2))
    plt.xlabel('Iterations')
    plt.ylabel('People waiting per station')
    # plt.text(20, 4, r'$\mu = ' + c +'$')
    show_save_close('outfig_ex')


def drawacc():
    with open('out_acc.dat', 'rb') as infile :
        acc = pickle.load(infile)
    list0 = sorted(acc.items())
    x,y = zip(*list0)
    plt.plot(x, y, 'o--')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    c = str(round(y[-1],4))
    plt.text(700,0.3, ' last value = ' + c )
    show_save_close('outfig_acc')
    
def drawlogacc():
    with open('out_acc.dat', 'rb') as infile :
        acc = pickle.load(infile)
    list0 = sorted(acc.items())
    x,y = zip(*list0)
    ylog = [-np.log(1-v) for v in y]
    plt.plot(x, ylog, 'o--')
    plt.xlabel('Epochs')
    plt.ylabel('-log(1 - Accuracy)')
    c = str(round(ylog[-1],4))
    plt.text(700,0.3, ' last value = ' + c )
    show_save_close('outfig_logacc')
    
def drawloss():
    with open('out_loss.dat', 'rb') as infile :
        loss = pickle.load(infile)
    list0 = sorted(loss.items())
    x,y = zip(*list0)
    plt.plot(x, y, 'o--')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    c = str(round(y[-1],4))
    plt.text(700,0.5, ' last value = ' + c )
    show_save_close('outfig_loss')
    
def drawvacc():
    with open('out_vacc.dat', 'rb') as infile :
        vacc = pickle.load(infile)
    list0 = sorted(vacc.items())
    x,y = zip(*list0)
    plt.plot(x, y, 'o--')
    plt.xlabel('Epochs')
    plt.ylabel('Validation Accuracy')
    c = str(round(y[-1],4))
    plt.text(700,0.9, ' last value = ' + c )
    show_save_close('outfig_vacc')
    
def drawvloss():
    with open('out_vloss.dat', 'rb') as infile :
        vloss = pickle.load(infile)
    list0 = sorted(vloss.items())
    x,y = zip(*list0)
    plt.plot(x, y, 'o--')
    plt.xlabel('Epochs')
    plt.ylabel('Validation Loss')
    c = str(round(y[-1],4))
    plt.text(700,0.6, ' last value = ' + c )
    show_save_close('outfig_vloss')
    
def drawepps():
    with open('out_Epps.dat', 'rb') as infile :
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
    show_save_close('outfig_epps')
        
def testai():
    with open("pre_ai.dat", "rb") as fe:
        epps = pickle.load(fe)
    c=str(round(st.mean(epps),2))
    d=str(round(st.pstdev(epps),2))
    e=str(len(epps))
    plt.clf()
    plt.xlabel('E[pps]')
    plt.ylabel('Frequency')
    plt.text(30, 0.25, r'$\mu = ' + c + ', \ \sigma = ' + d + ', n = ' + e + '$')
    plt.hist(epps, normed=1, bins = 16)
    show_save_close('outfig_ai')
    
def testaizoom():
    with open("pre_ai.dat", "rb") as fe:
        epps = pickle.load(fe)
    c=str(round(st.mean(epps),2))
    d=str(round(st.pstdev(epps),2))
    e=str(len(epps))
    plt.clf()
    plt.xlabel('E[pps]')
    plt.ylabel('Frequency')
    plt.text(30, 0.25, r'$\mu = ' + c + ', \ \sigma = ' + d + ', n = ' + e + '$')
    plt.hist(epps, normed=1, bins = 16, range=[1,2])
    show_save_close('outfig_zoomai')
    
def prepai(n):
    try:
        with open("pre_ai.dat", "rb") as fe:
            epps = pickle.load(fe)
    except:
        epps = []

    ai = AI_ON(C, N)
    for k in range (n):
        print("{}/{}".format((k+1),n))
        pr = Profiler(World(C, N), ai)
        epps.append(pr.w)

    with open("pre_ai.dat", "wb") as fw:   #Pickling
        pickle.dump(epps, fw)

def plot_all():
    plt.ion()
    plt.clf()
    drawacc()
    plt.clf()
    drawlogacc()
    plt.clf()
    drawloss()
    plt.clf()
    drawvacc()
    plt.clf()
    drawvloss()
    plt.clf()
    drawepps()

from timeit import timeit
   
if (__name__ == "__main__"):
    #tf_sess = tf.Session()
    #keras_backend.set_session(tf_sess)

    t = timeit(main_entry_train, number=1)
    print("Time:", t, "(sec)")
    
    plot_all()
    
    prepai(512)
    testai()
    testaizoom()
