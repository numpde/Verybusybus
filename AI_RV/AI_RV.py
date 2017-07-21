"""
Author of AI_RV: Rui Viana
"""
from random import randint
from numpy import mean
from collections import namedtuple
import matplotlib.pyplot as plt
import statistics as st

#
Carry_Option = namedtuple("Carry_Option","from_idx to_idx score")
Passenger = namedtuple("Passenger", "from_st to_st qidx carry_options chosen_option")
#
class AI_RV:
    name = "Rui's AI"
    def __init__(self, C, N):
        self.C = C
        self.N = N
        self.lastDir = 1
        self.path_len = int(self.N * 1.25)
        disc = 1.02
        self.drop_ws = []
        for idx in range(self.path_len):
            self.drop_ws.append(1.0-(1.0-pow(disc,-idx))/(1.0-pow(disc,-self.path_len+1)))

    def evaluate_path(self, B, Q, path):
        occupations = [0]*self.path_len # how many people in the bus at the time we leave the i'th station
        drops = [0]*self.path_len # how many people we drop into the i'th station
     
        for passenger in B: # account for all the passengers already in the bus
            for pidx,p in enumerate(path):
                if p == passenger:
                    drops[pidx] += 1
                    break
                occupations[pidx] += 1
 
        # make a list of all the queued passengers, and when we'd pick them up and drop them off
        passengers = []
        for from_st, stQ in enumerate(Q):
            visit_pathidxs = [idx for idx, st in enumerate(path) if st==from_st]  # find out when the path visits this station
            for qidx, to_st in enumerate(stQ):
                carry_options = []
                for from_idx in visit_pathidxs:
                    after_path = path[from_idx:]
                    # after pick up the path must pass by the drop off station
                    try: to_idx = after_path.index(to_st)
                    except ValueError: break  # if not found, then no point in checking latter visits, so break out
                    carry_options.append(Carry_Option(from_idx=from_idx, to_idx=from_idx+to_idx, score=to_idx*self.path_len+from_idx))
 
                if len(carry_options)==0: continue
                carry_options.sort(key=lambda x: x.score)
                passengers.append(Passenger(from_st=from_st, to_st=to_st, qidx=qidx, carry_options=carry_options, chosen_option=[None]))
 
        passengers.sort(key=lambda x: x.carry_options[0].score) # sort based on how long they'll stay on the bus
 
         # add passengers one at a time until the bus is full, or we run out of passengers to add
        picked = []
        for passenger in passengers:
            for co in passenger.carry_options:
                if not all([occupations[x] < self.C for x in range(co.from_idx, co.to_idx)]): continue
                # we're able to carry this passenger
                for x in range(co.from_idx, co.to_idx): occupations[x] += 1
                passenger.chosen_option[0] = co
                picked.append(passenger)
                break

        # for passengers that were picked, can we pick them up even earlier? if so, do that
        for passenger in picked:
            for co in passenger.carry_options:
                if co.from_idx >= passenger.chosen_option[0].from_idx: continue
                if not all([occupations[x] < self.C for x in range(co.from_idx, passenger.chosen_option[0].from_idx)]): continue
                for x in range(co.from_idx, passenger.chosen_option[0].from_idx): occupations[x] += 1
                passenger.chosen_option[0] = co

        # build the drops and M arrays
        M = []
        for passenger in picked:
            drops[passenger.chosen_option[0].to_idx] += 1
            if passenger.chosen_option[0].from_idx == 0: M.append(passenger.qidx)

        score = 0.0
        for idx,d in enumerate(drops): score += self.drop_ws[idx] * d
        return score, sorted(M)
 
    def step(self, b, B, Q):
        options = []
        for direction in [self.lastDir, self.lastDir*(-1)]:
            path = [(b+x*direction+self.N*10)%self.N for x in range(self.path_len)]
            path_score, path_M = self.evaluate_path(B, Q, path)
            if direction == self.lastDir: path_score += 0.2
            options.append([path, path_score, path_M])
 
            max_pivot_dist = int(0.35*self.N) if direction == self.lastDir else 1
            for pivot_dist in range(1,max_pivot_dist+1):
                path2 = [(b+x*direction+self.N*10)%self.N for x in range(pivot_dist+1)]
                path2 += list(reversed(path2[:-1]))
                path2 += [(b-x*direction+self.N*10)%self.N for x in range(1,self.path_len-len(path2)+1)]
                path_score, path_M = self.evaluate_path(B, Q, path2)
                if direction == self.lastDir: path_score -= 0.08
                options.append([path2, path_score, path_M])
     
        options.sort(reverse = True, key = lambda x: x[1])
        top_path = options[0][0]
        self.lastDir = (top_path[1] - top_path[0] + 1 + self.N )%self.N - 1
        return options[0][2], self.lastDir

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
        
        # tw = total waiting time per person (list)
        # bw = bus waiting time per person (list)
        # sw = station waiting time per person (list)
        # _abs = absolute excess waiting time 
        # _rel = relative excess waiting time
        
        self.sw = []
        self.sabs = []
        self.srel = []
        self.tw = []
        self.tabs = []
        self.trel = []
        self.bw = []
        self.babs = []
        self.brel = []

        assert 0 < self.I <= 1e9

        wrd.rewind()
        assert wrd.i == 0

        # Main loop
        while wrd.i < self.I:
            wrd.move(*nav.step(*wrd.look()))
            self.W.append(wrd.get_w())
            #print(wrd.i)

        assert len(self.W)
        for p in wrd.P :
            (a, b, i, m, d) = p
            if (d is None) : continue
            #dab = distance from a to b
            dab = min(abs(b-a),wrd.N-b+a,wrd.N-a+b)
            self.sw.append(m-i)
            self.sabs.append((m-i)-dab)
            self.srel.append((m-i)/dab)
            self.tw.append(d-i)
            self.tabs.append((d-i)-dab)
            self.trel.append((d-i)/dab)
            self.bw.append(d-m)
            self.babs.append((d-m)-dab)
            self.brel.append((d-m)/dab)
        self.w = mean(self.W)

class World:
    """
    Simulates the system step by step.
    Do not change this class.
    """
    def __init__(self, C, N):
        self.C = C         # Bus capacity
        self.N = N         # Number of stations
        self.b = None      # Bus position
        self.B = None      # Bus passengers' destinations [list]
        self.Q = None      # Queues at stations [list of list]
        self.H = None      # how long wait at station (each passenger classed by station) [list of list]
        self.T = None      # List that says at what iteration ppl got in the bus
        self.P = None       # people
        self.i = None      # Iteration number (i.e. time)
        self.NEWS = [None] # World trajectory record [list of tuple/None]
        self.rewind()

    def rewind(self):
        """
        Rewinds the world.
        """
        self.b = 0
        self.B = []
        self.Q = [[] for _ in range(self.N)]
        self.H = [[] for _ in range(self.N)]
        self.T = []
        self.P = []
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
        return self.b, self.B[:], [q[:] for q in self.Q]

    def board1(self, m):
        '''
        Board one passenger
        m is an element of M, see move(...)
        '''
        
        self.P[self.H[self.b][m]][3] = self.i
        self.B.append(self.Q[self.b][m])
        self.T.append(self.H[self.b][m])
        self.Q[self.b].pop(m)
        self.H[self.b].pop(m)
        
    def move(self, M, s):
        """
        Performs the move indicated by an AI.

        Args:
            M (:obj: `list` of int): is a list of indices M = [i1, i2, .., im]
                into the list Q[b] indicating that the people Q[b][i] will board
                the bus (in the order defined by M).
                Set M = [] if no one boards the bus.
                Note the constraints:
                    len(B) + len(M) <= Capacity C,
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
        for m in sorted(M, reverse=True):
            self.board1(m)
            
        # Advance time
        self.i += 1

        # Advance bus
        self.b = (self.b + (self.N + s)) % self.N

        # Passengers unmount
        self.B = [p for p in self.B if p != self.b]
        # Record passenger arrival time
        for i in self.T:
            if (self.P[i][1] == self.b):
                self.P[i][4] = self.i
        self.T = [i for i in self.T if self.P[i][1] != self.b]

        # Advance time
        #self.i += 1

        assert self.news() is not None
        # New person arrives at "a" with destination "b"
        a, b = self.news()
        # Queue in the new person
        self.Q[a].append(b)
        self.H[a].append(len(self.P))
        self.P.append([a, b, self.i, None, None])
        
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

        # 0.
        # C is an integer >= 1
        # N is an integer >= 2

        assert isinstance(C, int) and (C >= 1)
        assert isinstance(N, int) and (N >= 2)

        is_station = lambda n: isinstance(n, int) and (0 <= n < N)

        # 1.
        # b is an integer 0 <= b < N denoting
        #   the current location of the bus.

        assert is_station(b)

        # 2.
        # B is a list [n1, n2, ..] of
        #   the destinations of the passengers
        #   currently on the bus
        #   (not exceeding the capacity), i.e.
        #   nk is the destination of passenger k.
        #   The order is that of boarding
        #   (provided by this function: see M).
        #   No destination is the current position.

        assert isinstance(B, list)
        assert all(is_station(n) for n in B)
        assert all((n != b) for n in B)

        # 3.
        # Q is a list of N lists, where
        #   Q[n] = [t1, t2, ..] is the list of
        #   people currently waiting at station n
        #   with destinations t1, t2, ..
        #   No destination equals the location,
        #   i.e. (t != n) for any t in Q[n].

        assert isinstance(Q, list)
        assert len(Q) == N
        assert all(isinstance(q, list) for q in Q)
        assert all(all(is_station(t) for t in q) for q in Q)
        assert all(all((t != n) for t in q) for n, q in enumerate(Q))

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

        assert isinstance(M, list)
        assert all(isinstance(i, int) for i in M)
        assert all((0 <= i < len(Q[b])) for i in M)
        assert len(B) + len(M) <= C

        # 5.
        # s is either +1, -1, or 0, indicating
        #   the direction of travel of the bus
        #   (the next station is (b + s) % N).

        assert isinstance(s, int)
        assert (s in [-1, 0, 1])

        
def show_save_close(filename) :
    plt.show()
    plt.savefig(filename + '.eps')
    plt.savefig(filename + '.png')
    plt.close()

rui = Profiler(World(10,20),AI_RV(10,20))

I = 1000000

plt.ion()
        
plt.plot(range(I),rui.W)
c=str(round(rui.w,2))
plt.xlabel('Iterations')
plt.ylabel('People waiting per station')
plt.text(100000, 1.8, r'$\mu = ' + c +'$')
show_save_close('rui_evol')

plt.clf()
c=str(round(rui.w,2))
d=str(round(st.pstdev(rui.W),2))
plt.xlabel('People waiting per station')
plt.ylabel('Frequency')
plt.text(0.2, 2, r'$\mu = ' + c + ', \ \sigma = ' + d +'$')
plt.hist(rui.W, normed=1, bins = 17)
show_save_close('rui_hist')

#Station waiting time

plt.clf()
c=str(round(st.mean(rui.sw),2))
d=str(round(st.pstdev(rui.sw),2))
plt.xlabel('On station total waiting time')
plt.ylabel('Frequency')
plt.text(80, 0.03, r'$\mu = ' + c + ', \ \sigma = ' + d +'$')
plt.hist(rui.sw, normed=1, bins = 20)
show_save_close('rui_sw')

plt.clf()
c=str(round(st.mean(rui.sw),2))
d=str(round(st.pstdev(rui.sw),2))
plt.xlabel('On station total waiting time')
plt.ylabel('Frequency')
plt.text(50, 0.03, r'$\mu = ' + c + ', \ \sigma = ' + d +'$')
plt.hist(rui.sw, normed=1, bins = 14,range = [0,75])
show_save_close('rui_sw_zoom')

plt.clf()
c=str(round(st.mean(rui.srel),2))
d=str(round(st.pstdev(rui.srel),2))
plt.xlabel('On station relatif exceed waiting time')
plt.ylabel('Frequency')
plt.text(80, 0.08, r'$\mu = ' + c + ', \ \sigma = ' + d +'$')
plt.hist(rui.srel, normed=1, bins = 20)
show_save_close('rui_srel')

plt.clf()
c=str(round(st.mean(rui.srel),2))
d=str(round(st.pstdev(rui.srel),2))
plt.xlabel('On station relatif exceed waiting time')
plt.ylabel('Frequency')
plt.text(18, 0.175, r'$\mu = ' + c + ', \ \sigma = ' + d +'$')
plt.hist(rui.srel, normed=1, range=[0,25], bins = 14)
show_save_close('rui_srel_zoom')

plt.clf()
c=str(round(st.mean(rui.sabs),2))
d=str(round(st.pstdev(rui.sabs),2))
plt.xlabel('On station absolute exceed waiting time')
plt.ylabel('Frequency')
plt.text(80, 0.03, r'$\mu = ' + c + ', \ \sigma = ' + d +'$')
plt.hist(rui.sabs, normed=1, bins = 20)
show_save_close('rui_sabs')

plt.clf()
c=str(round(st.mean(rui.sabs),2))
d=str(round(st.pstdev(rui.sabs),2))
plt.xlabel('On station absolute exceed waiting time')
plt.ylabel('Frequency')
plt.text(40, 0.03, r'$\mu = ' + c + ', \ \sigma = ' + d +'$')
plt.hist(rui.sabs, normed=1, range=[-10,75], bins = 14)
show_save_close('rui_sabs_zoom')

#Bus waiting time

plt.clf()
c=str(round(st.mean(rui.bw),2))
d=str(round(st.pstdev(rui.bw),2))
plt.xlabel('On bus total waiting time')
plt.ylabel('Frequency')
plt.text(40, 0.08, r'$\mu = ' + c + ', \ \sigma = ' + d +'$')
plt.hist(rui.bw, normed=1, bins = 20)
show_save_close('rui_bw')

plt.clf()
c=str(round(st.mean(rui.bw),2))
d=str(round(st.pstdev(rui.bw),2))
plt.xlabel('On bus total waiting time')
plt.ylabel('Frequency')
plt.text(20, 0.065, r'$\mu = ' + c + ', \ \sigma = ' + d +'$')
plt.hist(rui.bw, normed=1, bins = 14,range = [0,30])
show_save_close('rui_bw_zoom')

plt.clf()
c=str(round(st.mean(rui.brel),2))
d=str(round(st.pstdev(rui.brel),2))
plt.xlabel('On bus relatif exceed waiting time')
plt.ylabel('Frequency')
plt.text(40, 0.3, r'$\mu = ' + c + ', \ \sigma = ' + d +'$')
plt.hist(rui.brel, normed=1, bins = 20)
show_save_close('rui_brel')

plt.clf()
c=str(round(st.mean(rui.brel),2))
d=str(round(st.pstdev(rui.brel),2))
plt.xlabel('On bus relatif exceed waiting time')
plt.ylabel('Frequency')
plt.text(3.5, 1.2, r'$\mu = ' + c + ', \ \sigma = ' + d +'$')
plt.hist(rui.brel, normed=1, range=[1,5], bins = 8)
show_save_close('rui_brel_zoom')

plt.clf()
c=str(round(st.mean(rui.babs),2))
d=str(round(st.pstdev(rui.babs),2))
plt.xlabel('On bus absolute exceed waiting time')
plt.ylabel('Frequency')
plt.text(40, 0.150, r'$\mu = ' + c + ', \ \sigma = ' + d +'$')
plt.hist(rui.babs, normed=1, bins = 15)
show_save_close('rui_babs')

plt.clf()
c=str(round(st.mean(rui.babs),2))
d=str(round(st.pstdev(rui.babs),2))
plt.xlabel('On bus absolute exceed waiting time')
plt.ylabel('Frequency')
plt.text(10, 0.25, r'$\mu = ' + c + ', \ \sigma = ' + d +'$')
plt.hist(rui.babs, normed=1, range=[0,14], bins = 7)
show_save_close('rui_babs_zoom')

#Total waiting time

plt.clf()
c=str(round(st.mean(rui.tw),2))
d=str(round(st.pstdev(rui.tw),2))
plt.xlabel('Total waiting time')
plt.ylabel('Frequency')
plt.text(125, 0.025, r'$\mu = ' + c + ', \ \sigma = ' + d +'$')
plt.hist(rui.tw, normed=1, bins = 16)
show_save_close('rui_tw')

plt.clf()
c=str(round(st.mean(rui.tw),2))
d=str(round(st.pstdev(rui.tw),2))
plt.xlabel('Total waiting time')
plt.ylabel('Frequency')
plt.text(35, 0.03, r'$\mu = ' + c + ', \ \sigma = ' + d +'$')
plt.hist(rui.tw, normed=1, bins = 10,range = [0,50])
show_save_close('rui_tw_zoom')

plt.clf()
c=str(round(st.mean(rui.trel),2))
d=str(round(st.pstdev(rui.trel),2))
plt.xlabel('Relatif exceed waiting time')
plt.ylabel('Frequency')
plt.text(100, 0.07, r'$\mu = ' + c + ', \ \sigma = ' + d +'$')
plt.hist(rui.trel, normed=1, bins = 14)
show_save_close('rui_trel')

plt.clf()
c=str(round(st.mean(rui.trel),2))
d=str(round(st.pstdev(rui.trel),2))
plt.xlabel('Relatif exceed waiting time')
plt.ylabel('Frequency')
plt.text(15, 0.12, r'$\mu = ' + c + ', \ \sigma = ' + d +'$')
plt.hist(rui.trel, normed=1, range=[1,21], bins = 10)
show_save_close('rui_trel_zoom')

plt.clf()
c=str(round(st.mean(rui.tabs),2))
d=str(round(st.pstdev(rui.tabs),2))
plt.xlabel('Absolute exceed waiting time')
plt.ylabel('Frequency')
plt.text(125, 0.020, r'$\mu = ' + c + ', \ \sigma = ' + d +'$')
plt.hist(rui.tabs, normed=1, bins = 17)
show_save_close('rui_tabs')

plt.clf()
c=str(round(st.mean(rui.tabs),2))
d=str(round(st.pstdev(rui.tabs),2))
plt.xlabel('Absolute exceed waiting time')
plt.ylabel('Frequency')
plt.text(50, 0.025, r'$\mu = ' + c + ', \ \sigma = ' + d +'$')
plt.hist(rui.tabs, normed=1, range=[0,70], bins = 14)
show_save_close('rui_tabs_zoom')