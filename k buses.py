#!/usr/bin/env python3

"""
Template by verybusybus.wordpress.com
Author: Alejandro Caicedo
TODO: SAVE THIS FILE TO YOUR MACHINE
      WRITE YOUR NAME(S) HERE
"""

import statistics as st
import matplotlib.pyplot as plt
from random import randint
from numpy import mean

class AI_KCLOCK:
    """
    'Always go in the same direction' strategy
    """
    name = "KClock"

    def __init__(self, C, N, K):
        self.C = C
        self.N = N
        self.K = K

    def step(self, b, B, Q):
        """
        Calculates one step.
        """
        # Number of passengers to board
        # n = [0 for _ in range(self.K)]
        M = [[] for _ in range(self.K)]
        
        # for i in range(self.K):
        #     acc = 0
        #     for j in range(i):
        #         if b[j] != b[i]:continue
        #         acc += n[j]
        #     n[i] = min((len(Q[b[i]]) - acc), self.C - len(B[i]))
        #     M[i] = list(range(acc, acc + n[i]))
        # Passenger selection from Q[b]:
        # Take passengers number 0, 1, .., n-1
        M[0] = [i for i in range(len(Q[b[0]])) if (((Q[b[0]][i] - b[0])%(self.N)) <= (int((self.N)/2)))]
        M[0] = M[0][:(self.C - len(B[0]))]
        
        M[1] = [i for i in range(len(Q[b[1]])) if (((Q[b[1]][i] - b[1])%(self.N)) > (int((self.N)/2)))]
        M[1] = M[1][:(self.C - len(B[1]))]
        
        # print(M)
        # print(b)
        # print(Q[b[0]])
        # print(Q[b[1]])
        # print(B)
        
        # Always go in one direction
        s = [ (-1)**i for i in range(self.K)]

        return M, s        
        

class World:
    """
    Simulates the system step by step.
    Do not change this class.
    """
    def __init__(self, C, N, K):
        self.C = C         # Bus capacity
        self.N = N         # Number of stations
        self.K = K         # Number of buses
        self.b = None      # Buses position [list]
        self.B = None      # Bus passengers' destinations [list of list]
        self.Q = None      # Queues at stations [list of list]
        self.i = None      # Iteration number (i.e. time)
        self.NEWS = [None] # World trajectory record [list of tuple/None]
        self.rewind()

    def rewind(self):
        """
        Rewinds the world.
        """
        self.b = [int((i*(self.N))/(self.K)) for i in range(self.K)]
        self.B = [[] for _ in range(self.K)]
        self.Q = [[] for _ in range(self.N)]
        self.i = 0

    def news(self):
        """
        Creates news if necessary.
        Returns:
            (a, b): a person arrives at "a" with destination "b".
        """
        # Create news if necessary
        while len(self.NEWS) <= self.i:
            # New person arrives at "a" with destination "b"
            a = randint(0, self.N - 1)
            b = (a + randint(1, self.N - 1)) % self.N
            self.NEWS.append((a, b))
        assert 0 <= self.i < len(self.NEWS)
        return self.NEWS[self.i]

    def look(self):
        """
        Returns a copy of (b, B, Q).
        """
        return self.b[:],[h[:] for h in self.B], [q[:] for q in self.Q]

    def board1(self, m, i,q,remov):
        '''
        Board one passenger
        m is an element of M, see move(...)
        '''
        #for i in range(self.K-1,-1,-1):
        self.B[i].append(self.Q[self.b[i]][m])
        # self.Q[self.b[i]].pop(m)
        remov[q].append(m)
    
    def move(self, M, s):
        """
        Performs the move indicated by an AI.
        Args:
            M (:obj: `list of list of int´): is a list of indices M = [[i11, i12, .., i1m],..,[ik1,ik2,...,ikr]]
                into the list Q[b] indicating that the people Q[b][i] will board
                the bus j (in the order defined by M).
                Set M = [] if no one boards the bus.
                Note the constraints:
                    len(B) + len(M[j]) <= Capacity C,
                and
                    0 <= i < len(Q[b]) for each i in M.
            s (int): is either +1, -1, or 0, indicating the direction of travel
                of the bus (the next station is (b + s) % N).
        """
        # Check consistency from time to time
        if randint(0, 100) == 0:
            self.check_consistency(self.C, self.N, self.b, self.B, self.Q, M, s)

        # Passengers mount (in the given order)
        # and are removed from the queue
        remov = [[] for _ in range(self.N)]
        for i in (range(self.K)):
            # if len(self.B[i]) == self.C: break
            for m in M[i]:
                self.board1(m,i,self.b[i],remov)
            # print(remov) 
        for k in range(self.N):       
            self.Q[k] = [self.Q[k][j] for j in range(len(self.Q[k])) if j not in remov[k]]
        # Advance time
        self.i += 1

        # Advance buses
        self.b = [((self.b[i] + (self.N + s[i])) % self.N) for i in range(self.K)]

        # Passengers unmount
        self.B = [[p for p in self.B[i] if p != self.b[i]] for i in range(self.K)]

        # Advance time
        # self.i += 1

        assert self.news() is not None
        # New person arrives at "a" with destination "b"
        a, b = self.news()
        # Queue in the new person
        self.Q[a].append(b)

    def get_w(self):
        """
        Returns:
            Number of people waiting in queue, averaged over the stations.
        """
        return mean([len(q) for q in self.Q])

    @staticmethod
    def check_consistency(C, N, b, B, Q, M, s):
        """
        Checks consistency of the input.
        """
        pass
        # 0.
        # C is an integer >= 1
        # N is an integer >= 2

        #assert isinstance(C, int) and (C >= 1)
        #assert isinstance(N, int) and (N >= 2)

        #is_station = lambda n: isinstance(n, int) and (0 <= n < N)

        # 1.
        # b is an integer 0 <= b < N denoting
        #   the current location of the bus.

        #for i in range(K):
        #    assert is_station(b[i])

        # 2.
        # B is a list [n1, n2, ..] of
        #   the destinations of the passengers
        #   currently on the bus
        #   (not exceeding the capacity), i.e.
        #   nk is the destination of passenger k.
        #   The order is that of boarding
        #   (provided by this function: see M).
        #   No destination is the current position.

        #assert isinstance(B, list)
        #assert all(is_station(n) for n in B)
        #assert all((n != b) for n in B)

        # 3.
        # Q is a list of N lists, where
        #   Q[n] = [t1, t2, ..] is the list of
        #   people currently waiting at station n
        #   with destinations t1, t2, ..
        #   No destination equals the location,
        #   i.e. (t != n) for any t in Q[n].

        #assert isinstance(Q, list)
        #assert len(Q) == N
        #assert all(isinstance(q, list) for q in Q)
        #assert all(all(is_station(t) for t in q) for q in Q)
        #assert all(all((t != n) for t in q) for n, q in enumerate(Q))

        # 4.
        # M is a list of indices M = [i1, i2, .., im]
        #   into the list Q[b] indicating that
        #   the people Q[b][i] will board the bus
        #   (in the order defined by M).
        #   Set M = [] if no one boards the bus.
        #   Note the constraints:
        #     len(B) + len(M) <= Capacity C,
        #   and
        #     0 <= i < len(Q[b]) for each i in M.

        #assert isinstance(M, list)
        #assert all(isinstance(i, int) for i in M)
        #assert all((0 <= i < len(Q[b])) for i in M)
        #assert len(B) + len(M) <= C

        # 5.
        # s is either +1, -1, or 0, indicating
        #   the direction of travel of the bus
        #   (the next station is (b + s) % N).

        #assert isinstance(s, int)
        #assert (s in [-1, 0, 1])

class Profiler:
    """
    Runs the systems with a particular strategy "nav".
    """

    # Number of iterations (time steps)
    # This will be I ~ 1e6
    I = 1000000

    def __init__(self, wrd, nav):
        # W[i] = average number of people waiting at time i
        self.W = []
        # w = average over time
        self.w = None

        assert 0 < self.I <= 1e9

        wrd.rewind()
        assert wrd.i == 0

        # Main loop
        while wrd.i < self.I:
            wrd.move(*nav.step(*wrd.look()))
            self.W.append(wrd.get_w())

        assert len(self.W)
        self.w = mean(self.W)

# Helper function
def get_name(nav):
    """
    Args:
        nav (:obj: AI_*): the Strategy nav.
    Returns:
        nav.name (str): the name of a nav or "Unknown".
    """
    try:
        return nav.name
    except:
        return "Unknown"

def show_save_close(filename) :
    plt.show()
    plt.savefig(filename + '.eps')
    plt.savefig(filename + '.png')
    plt.close()

if __name__ == "__main__":
    
    I = 1000000    
    kcl5=Profiler(World(5,20,2),AI_KCLOCK(5,20,2))    
    kcl=Profiler(World(10,20,2),AI_KCLOCK(10,20,2))    
    
    plt.ion()
    
    plt.plot(range(I),kcl.W)
    c=str(kcl.w)
    plt.xlabel('Iterations')
    plt.ylabel('People waiting per station')
    plt.text(100000, 1, r'$\mu = ' + c + '$')
    show_save_close('2clock_evol')
    
    plt.clf()
    c=str(round(kcl.w,2))
    d=str(round(st.pstdev(kcl.W),2))
    plt.xlabel('People waiting per station')
    plt.ylabel('Frequency')
    plt.text(0.7, 4, r'$\mu = ' + c + ', \ \sigma =' + d + '$')
    plt.hist(kcl.W, normed=1)
    show_save_close('2clock_histo')

    plt.clf()
    plt.plot(range(I),kcl5.W)
    c=str(kcl5.w)
    plt.xlabel('Iterations')
    plt.ylabel('People waiting per station')
    plt.text(100000, 1.4, r'$\mu = ' + c + '$')
    show_save_close('2clock5_evol')
    
    plt.clf()
    c=str(round(kcl5.w,2))
    d=str(round(st.pstdev(kcl5.W),2))
    plt.xlabel('People waiting per station')
    plt.ylabel('Frequency')
    plt.text(0.9, 3, r'$\mu = ' + c + ', \ \sigma =' + d + '$')
    plt.hist(kcl5.W, normed=1)
    show_save_close('2clock5_histo')
    