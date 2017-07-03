"""
Author of AI_RV: Rui Viana
"""
from random import randint
from numpy import mean
from collections import namedtuple
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
