import numpy as np
import random
from functions import rand_argmin, rand_argmax, kl_inv_approx
from kullback_leibler import klucbBern
import math

       
# ET: proposed algorithm for model B
class KLStopping():
    def __init__(self, horizon, epsilon_satis_regret, M, N, s, T_0):
        self.N = N
        self.M = M
        self.epsilon_satis_regret = epsilon_satis_regret
        self.horizon = horizon
        self.s = s
        self.T_0 = T_0
        self.u_current = 0
        self.item_selected = 0
        self.item_prev = 0 #previously chosen item
        self.prev_reward =0 # previous reward
        self.Xi_u = [0 for i in range(N)] 
        self.Xi_current = []
        self.mu_1_minus_eps_est = 0 #estimate of mu_1_minus_eps/2
        
        self.t = 1 # time
        
        self.S = [0 for i in range(self.N)] # item set S
        self.V =  [0 for i in range(self.N)] # item set V
        self.item_cnt = [0 for i in range(N)] # number of observations for each item 
        self.item_reward_sum = [0 for i in range(N)] # sum of rewards collected
        
        self.create_V = 0 
        
        self.testing_timings = set() # list of numbers for the testings
        

    def choose_item(self, state):
        
        # expand the state
        (self.u_current, self.Xi_current, self.prev_reward) = state
        self.Xi_u = self.Xi_current[:, self.u_current] 
        if sum(self.Xi_u) == self.N:
            print('impossible!')
            
        
        assert len(self.Xi_u) == self.N
        
        if self.t ==1:
            # 1. item sampling
            self.random_S_selection()
            
        # update the reward sum information
        self.update_reward_sum()            

        if self.t >= 1 and self.t <= self.T_0:
            # 2. exploration
            self.explorations()

        else: 
            if self.create_V ==0:
                self.create_V = 1 
                # compute V, mu_{1 - eps/2}, timings, 
                self.create_V_from_S()
                print('exploration finished.')
            
            # 3. exploitation
            self.exploitation()

        # increment time
        self.t = self.t + 1
                
        #assert self.Xi_current[self.item_selected, self.u_current] == 0
        
        # record previously selected item
        self.item_prev = self.item_selected

        
        return self.item_selected
    

    def random_S_selection(self):
        indices = random.sample(range(self.N), self.s)
        for index in indices:
            self.S[index] = 1
            
    def update_reward_sum(self):
        self.item_reward_sum[self.item_prev] = self.item_reward_sum[self.item_prev] + self.prev_reward
    
    def explorations(self):
        #remove the previously recommended items (Xi_u = 1) for the current user from the candidate
        #item_cnt_temp: element that is previously recommended items or not in S (S = 0) is made M
        item_cnt_temp = np.array(self.Xi_u == 1) *self.M + np.array(np.array(self.S) == 0) * self.M + self.item_cnt
        item_cnt_temp = np.minimum(item_cnt_temp, self.M)
        
        # If some of item_cnt_temp is not M <-> if there exists an item in I_0_minus that can be recommended to curent user
        if sum(item_cnt_temp) < self.M*self.N:
            
            # select item from S with the smallest recommended number
            self.item_selected = rand_argmin(item_cnt_temp)
            assert self.Xi_current[self.item_selected, self.u_current] == 0
            
        else:
            # select recommendable item from outside of S
            self.item_selected = rand_argmin(self.Xi_u)
            
            assert self.Xi_current[self.item_selected, self.u_current] == 0
            print('random sampling')
            
            
        #update the counter for the selected item
        self.item_cnt[self.item_selected] = self.item_cnt[self.item_selected] + 1
        
        
    def create_V_from_S(self):
        # compute V and hat{mu_{1 - eps/2}}, and the timings for the hypothesis test
        
        # compute hat{mu_{1 - eps/2}}, as a eps/2 |S| th element in emp_average 
        emp_average = self.emprical_average()
        emp_average = np.nan_to_num(emp_average, nan=-1)
        emp_average_descend = sorted(emp_average, reverse=True)
        self.mu_1_minus_eps_est = emp_average_descend[int(np.floor(self.epsilon_satis_regret / 2 * self.s)) - 1]
        
        # compute V
        for i in range(self.N):
            if self.S[i] == 1:
                if emp_average[i] >= self.mu_1_minus_eps_est:
                    self.V[i] = 1
                    
        # compute the timings for the hypothesis testing
        ell = 1
        while int(2**ell * math.log(math.log(2**np.e * self.M**2 ,2),2))  < self.M:
            self.testing_timings.add(int(2**ell * math.log(math.log(2**np.e * self.M**2 ,2),2)))
            ell = ell + 1
        
        # reset reward observation history
        for i in range(self.N):
            if self.S[i] == 1:
                if self.V[i] == 1:
                    self.item_cnt[i] = 0
                    self.item_reward_sum[i] = 0
                else:
                    self.item_cnt[i] = 1 # so that the item will not be not selected again
                    self.item_reward_sum[i] = 0
            else:
                    self.item_cnt[i] = 0
                    self.item_reward_sum[i] = 0                
                
        print('mu_1_minus_eps_est=', end="")
        print(self.mu_1_minus_eps_est)
        print('sum(S)=', end="")
        print(sum(self.S))                    
        print('sum(V)=', end="")
        print(sum(self.V))
        print('len(testing_timings) = ', end="")
        print(len(self.testing_timings))
        
            
    def emprical_average(self):        
        return np.divide(self.item_reward_sum, self.item_cnt)
    
    def exploitation(self):
        
        # do a hypothesis testing for the previous item
        self.hypothesis_test()
        
        # compute recommendable items from V to the current user 
        activeitems = np.multiply(self.Xi_u==0, np.array(self.V)==1)
        
        # if there are some items recommendable, recommend least recently recommended among V
        if sum(activeitems) > 0:
            
            # make the element that are not in V or activeitems to be larger than + self.M
            cnt_modified = np.array(self.item_cnt)  + self.M*  (np.array(activeitems) == 0)  + self.M*   (np.array(self.V)==0)
            self.item_selected = rand_argmin(cnt_modified)
        
        # if there are none from V, recommend item that is never recommended before and add to V
        else: 
            cnt_modified = np.array(self.item_cnt) + self.M* (np.array(self.Xi_u) == 1) 
            self.item_selected = rand_argmin(cnt_modified)
            self.V[self.item_selected] = 1
            
        #update the counter for the selected item
        self.item_cnt[self.item_selected] = self.item_cnt[self.item_selected] + 1
            
        
    def hypothesis_test(self):
        # test for every ell
        if self.item_cnt[self.item_prev] in self.testing_timings:
            timings_list  =  list(self.testing_timings)
            timings_list  =  sorted(timings_list)
            ell = [i for (i,j) in enumerate(timings_list) if j == self.item_cnt[self.item_prev]]
            # as list index start from 0
            ell = ell[0] + 1 
            
            rho_ell = kl_inv_approx(self.mu_1_minus_eps_est , 2**(- ell))
            empirical_averages = self.emprical_average()
            if empirical_averages[self.item_prev] <= rho_ell:
                # remove the item from the set V
                self.V[self.item_prev] = 0
            
            
# BKLUCB algorithm with sampling : 
class BKLUCB():
    def __init__(self, horizon, epsilon_satis_regret, M, N, s, T_0):
        self.N = N
        self.M = M
        self.horizon = horizon
        self.s = s
        
        self.t = 1
        self.KLUCB_indexes = np.zeros(self.N)
        self.item_reward_sum = [0 for i in range(N)] # sum of observed rewards for each item 
        self.item_cnt = [0 for i in range(N)] # number of observations for each item 
        self.item_emp_avg_rewards = [0 for i in range(N)]
        self.item_selected = 0
        self.item_prev = 0
        self.I_0 = [0 for i in range(N)] # subset of items that will be sampled
        self.I_0_indexes = [] # index i s.t. I_0[i] =1
        
        #KLUCB indexes initialization (as inf)
        for k in range(self.N):
            self.KLUCB_indexes[k] = np.inf
        
    def update_KLUCB_index(self):
        #f_t = max(np.log(self.t) + 3 * np.log(np.log(self.t)), np.log(3) + 3 * np.log(np.log(3)))
        f_t = max(np.log(self.t -1) + 3 * np.log(np.log(self.t - 1)), np.log(3) + 3 * np.log(np.log(3)))
        
        for i in self.I_0_indexes:
            if self.item_cnt[i] > 0:
                f_t_over_cnt = f_t / self.item_cnt[i]
                self.KLUCB_indexes[i] = klucbBern(self.item_emp_avg_rewards[i], f_t_over_cnt)
            else:
                self.KLUCB_indexes[i] = np.inf
            
        
    def update_emprical_average(self): 
         self.item_emp_avg_rewards = np.divide(self.item_reward_sum, self.item_cnt)
    
    def update_reward_sum(self):
        self.item_reward_sum[self.item_prev] = self.item_reward_sum[self.item_prev] + self.prev_reward
    
    def choose_item(self, state):
        (self.u_current, self.Xi_current, self.prev_reward) = state
        Xi_u = self.Xi_current[:, self.u_current]
        assert len(Xi_u) == self.N
        
        if self.t ==1:
            #select T/m log T items
            self.random_I_0_selection()
        
        #update the reward sum
        self.update_reward_sum()
        
        #update emirical average
        self.update_emprical_average()
        
        #update KLUCB index
        self.update_KLUCB_index()
        KLUCB_index_u = self.KLUCB_indexes
        
        # remove items that are not recommendable to the user are removed
        for i in range(self.N):
            if Xi_u[i] == 1:
                KLUCB_index_u[i] = 0

        # choose item based on modified KLUCB indexes
        if sum(KLUCB_index_u)>0:
            
            # select item in I_0 with largest KLUCB index
            self.item_selected = rand_argmax(KLUCB_index_u)
        else:
            
            #random item selections
            self.item_selected = rand_argmin(Xi_u)
        
        #update the counter
        self.item_cnt[self.item_selected] = self.item_cnt[self.item_selected] + 1
        
        # record previously used item
        self.item_prev = self.item_selected
        
        self.t = self.t + 1
        
        return self.item_selected
        
    # random T/m log T item selection
    def random_I_0_selection(self):
        s = self.s
        print('s=', end="")
        print(s)
        indices = random.sample(range(self.N), s)
        for index in indices:
            self.I_0[index] = 1          

        self.I_0_indexes = indices #[i for i, x in enumerate(self.I_0) if x == 1]

        indices_comp = set(range(self.N)) - set(indices)

        # items that are not in I_0: set KLUCB index as 0
        for index in indices_comp:
            self.KLUCB_indexes[index] = 0

        
