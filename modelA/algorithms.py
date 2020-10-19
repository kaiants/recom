import numpy as np
import random
from functions import rand_argmin, rand_argmax
from kullback_leibler import klucbBern
import copy

        
# BKLUCB algorithm with sampling 
class BKLUCB():
    def __init__(self, horizon, s, T_0, epsilon_alg, M, N, K, L, coef_of_hyothesis_testing):
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
            
# BKLUCB algorithm without sampling 
class KLUCB():
    def __init__(self, horizon, s, T_0, epsilon_alg, M, N, K, L, coef_of_hyothesis_testing):
        self.N = N
        self.M = M
        self.horizon = horizon
        
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

        for i in range(self.N):
            if self.item_cnt[i] > 0:
                f_t_over_cnt = f_t / self.item_cnt[i]
                self.KLUCB_indexes[i] = klucbBern(self.item_emp_avg_rewards[i], f_t_over_cnt)
            else:
                self.KLUCB_indexes[i] = np.inf
                
                
    def choose_item(self, state):
        (self.u_current, self.Xi_current, self.prev_reward) = state
        Xi_u = self.Xi_current[:, self.u_current]
        assert len(Xi_u) == self.N
        
        #update the reward sum
        self.update_reward_sum()
        
        #update empirical average
        self.update_emprical_average()
        
        #update KLUCB index
        self.update_KLUCB_index()
        KLUCB_index_u = self.KLUCB_indexes
        
        # remove items that are not recommendable to the user are removed
        for i in range(self.N):
            if Xi_u[i] == 1:
                KLUCB_index_u[i] = 0

        # choose item based on modified KLUCB indexes
        self.item_selected = rand_argmax(KLUCB_index_u)
        
        #update the counter
        self.item_cnt[self.item_selected] = self.item_cnt[self.item_selected] + 1
        
        # record previously used item
        self.item_prev = self.item_selected
        
        # increment time
        self.t = self.t + 1
        
        return self.item_selected
        
    def update_emprical_average(self): 
         self.item_emp_avg_rewards = np.divide(self.item_reward_sum, self.item_cnt)
    
    def update_reward_sum(self):
        self.item_reward_sum[self.item_prev] = self.item_reward_sum[self.item_prev] + self.prev_reward
            

class UniformNoUserClusters():
    def __init__(self, horizon, s, T_0, epsilon_alg, M, N, K, L):
        self.N = N
        
    #uniform sampling strategy without repetitions
    def choose_item(self, state):
        (u_current, Xi_current, _) = state
        Xi_u = Xi_current[:, u_current]
        assert len(Xi_u) == self.N
        
        #choose the item that is not recommended uniformly at random
        i = rand_argmin(Xi_u)
        assert Xi_current[i, u_current] == 0
        return i


# ECT: proposed algorithm for Model A
class ClusterFirstRecommendNext():
    def __init__(self, horizon, s, T_0, epsilon_alg, M, N, K, L, coef_of_hyothesis_testing):
        self.N = N
        self.M = M
        self.K = K
        self.coef_of_hyothesis_testing = coef_of_hyothesis_testing
        self.horizon = horizon
        self.t = 0 #algorithm's time counter
        self.s = int(np.floor(s))
        self.T_0 = int(np.floor(T_0))
        self.epsilon_alg = epsilon_alg
        self.I_0 = [0 for i in range(N)] # set of items that will be sampled initially: element is 1 if selected
        self.I_0_minus = [0 for i in range(N)] # set of items for which sufficient samples have not been obtained: element is 1 if not sufficient
        self.V = [0 for i in range(N)] 
        self.V_0 =  [0 for i in range(N)] 
        self.item_cnt = [0 for i in range(N)] # number of observations for each item 
        self.item_reward_sum = [0 for i in range(N)] # sum of rewards collected
        self.cluster_average = [0 for k in range(K)] # esimated average reward for each cluster
        self.Delta_0  = 0# minimum average gap between the clusters
        self.S_0 = 0 #maximum number of trials for the test
        self.reward_rec_flag = 0 # indicating to record the reward. 0: discard the reward information. 1: collect the reward information
        self.cluster_flag = 0 # 0: if not clustered yet, 1: if already clustered
        self.exploitaion_flag = 0 # 0: if exploitation phase is not finished yet, 1: if finished
        self.need_additional_sampling = 1 #0: if no additional sampling is needed, 1 if additional sampling is necessary
        self.item_selected = 0 #current item
        self.item_prev = 0 #previously chosen item
        self.prev_reward =0 # previous reward
        self.Xi_u = [0 for i in range(N)] 
        self.Xi_current = []
        self.u_current = 0
        self.LARGE_CONST = 30
        
        self.cnt_suboptimal_items = 0
        self.cnt_num_tests = 0
        
        if K==1:
            self.avr_rewards = np.zeros(N)
    
    def choose_item(self, state):
        
        # expand the state
        (self.u_current, self.Xi_current, self.prev_reward) = state
        self.Xi_u = self.Xi_current[:, self.u_current] 
        
        assert len(self.Xi_u) == self.N
        
        if self.t ==0:
            # 1. item sampling
            self.random_I_0_selection()
                    
        # if previous information should be used, update the reward sum
        if self.reward_rec_flag ==1:
            self.update_reward_sum()
            
        if sum(self.I_0_minus) > 0: # when there are remaining items that needs to be sampled more
            
            # 2. exploration
            self.explorations()
            
        else: # if sum(self.I_0_minus) == 0 meaning that all of the items in I_0 are sufficiently sampled
            
            # 3. clustering phase
            if self.cluster_flag ==0:
                
                # clustering
                self.do_clustering() 
                
                # clustering is performed only once
                self.cluster_flag =1
                
                # compute the minimum gap
                self.compute_delta_0()
                
                #initialize V, V_0
                self.initialize_V()
                
                                
                print('Delta_0=', end="")
                print(self.Delta_0)
                print('S_0=', end="")
                print(self.S_0)
                print('p1:', end="")
                print(self.cluster_average[0])
                print('p2:', end="")
                print(self.cluster_average[1])
                
                print('3. clustering done.')
            
            # indicate to record the reward
            self.reward_rec_flag = 1
            
            # 4. Exploitation 
            self.exploitation()
            if np.mod(self.t, 18000) ==0:
                print('sum(self.V):', end="")
                print(sum(self.V))
                print('sum(self.V_0):', end="")
                print(sum(self.V_0)) 


        # increment time
        self.t = self.t + 1
        
        if self.t == self.horizon-1:
            print('self.cnt_suboptimal_items=', end="")
            print(self.cnt_suboptimal_items)
            print('self.cnt_num_tests', end="")
            print(self.cnt_num_tests)
        
        
        # record previously used item
        self.item_prev = self.item_selected
        
        return self.item_selected
    
    def random_I_0_selection(self):
        indices = random.sample(range(self.N), self.s)
        for index in indices:
            self.I_0[index] = 1
                
        # make a independent copy of I_0
        self.I_0_minus = copy.deepcopy(self.I_0)
        
        
    def update_reward_sum(self):
        self.item_reward_sum[self.item_prev] = self.item_reward_sum[self.item_prev] + self.prev_reward
        
        
    def explorations(self):
        
        #remove the previously recommended items for the current user from the candidate
        #item_cnt_temp: element that is previously recommended items or not in I_0 is made M
        item_cnt_temp = np.array(self.Xi_u == 1) *self.M + np.array(np.array(self.I_0) == 0) * self.M + self.item_cnt
        item_cnt_temp = np.minimum(item_cnt_temp, self.M)
        

        # If some of item_cnt_temp is not M <-> if there exists an item in I_0_minus that can be recommended to curent user
        if sum(item_cnt_temp) < self.M*self.N:
            
            # select item from argmin item_cnt_temp
            self.item_selected = rand_argmin(item_cnt_temp)
            assert self.Xi_current[self.item_selected, self.u_current] == 0
            
            #update the counter
            self.item_cnt[self.item_selected] = self.item_cnt[self.item_selected] + 1
            
            #update I_0_minus
            self.I_0_minus = np.multiply(self.I_0, (np.array(self.item_cnt) <  self.T_0))
            
            # indicate to record the reward
            self.reward_rec_flag = 1
            
        else:
            # do a random sampling from a new item
            self.item_selected = rand_argmin(self.Xi_u)
            
            assert self.Xi_current[self.item_selected, self.u_current] == 0
            print('random sampling')
            
            # do not update the counter
            
            # indicate not to record the reward
            self.reward_rec_flag = 0
        
    def emprical_average(self):        
        return np.divide(self.item_reward_sum, self.item_cnt)
        
    def do_clustering(self):
        averages = self.emprical_average()
        averages = averages + (self.I_0 == 0) * self.LARGE_CONST
        
        # find neighborhoods
        Q = [0 for i in range(self.N)]
        for i in range(self.N):
            Q[i] = set()
            if self.I_0[i] == 1:
                for j in range(self.N):
                    if self.I_0[j] == 1:
                        if abs(averages[i] - averages[j]) <= self.epsilon_alg:
                            Q[i].add(j)
        
        M = set() 
        Qprev = set()
        
        #find centers of clusters
        for k in range(self.K):
            cardinalities = [0 for i in range(self.N)]
            for i in range(self.N):
                if self.I_0[i] == 1:
                    cardinalities[i] = len(Q[i] - Qprev)
                
            i = rand_argmax(np.array(cardinalities))
            M.add(i)
            Qprev = Qprev.union(Q[i])
        
        # initialize with negative value
        average_cluster_tmp = [- self.LARGE_CONST for i in range(self.N)]
        
        # put empirical averages of the elements of M
        for m in M:
            average_cluster_tmp[m] = averages[m]
        average_cluster_tmp = np.array(average_cluster_tmp)
        

        
        # reorder the cluster_average (in a discreasing order in k)
        for k in range(self.K):
            i  = rand_argmax(average_cluster_tmp)
            assert i in M
            
            self.cluster_average[k] = average_cluster_tmp[i]
            M = M - {i}
            average_cluster_tmp[i] = - self.LARGE_CONST
            
    def compute_delta_0(self):        
        #self.Delta_0 = self.cluster_average[0] - self.cluster_average[1] - 2 * self.epsilon_alg
        self.Delta_0 = self.cluster_average[0] - self.cluster_average[1] 
        self.S_0 = self.compute_S_0()
        
    def compute_S_0(self):
        return np.ceil(self.coef_of_hyothesis_testing / self.Delta_0**2)
        
    def exploitation(self):
        
        # update I_1
        self.update_V()
        
        # compute recommendable items from V to current user 
        activeitems = np.multiply(self.Xi_u==0, np.array(self.V)==1)
                
        if sum(activeitems) > 0:
            # compute emirical averages 
            averages = self.emprical_average()
            
            # keep only for the active items. (otherwise element is -1)
            for i in range(self.N):
                if activeitems[i] == 0:
                    averages[i] = -1
            
            # choose the best (empirical average) item in V
            self.item_selected = rand_argmax(averages)   
            assert self.Xi_current[self.item_selected, self.u_current] == 0
            
        else:
            # random sampling from V_0^c when there are no items from V that can be recommended to current user
            
            # Recompute activeitems, recommendable and in V_0^c
            activeitems =  np.multiply(self.Xi_u==0, np.array(self.V_0)==0)
            if sum(activeitems)==0:
                self.item_selected = rand_argmin(self.Xi_u)
            else:
                # select item unifromly at random from activeitems
                self.item_selected = rand_argmax(activeitems)
            
            # add the item to V and V_0
            self.V[self.item_selected] = 1
            self.V_0[self.item_selected] = 1
            if self.Xi_current[self.item_selected, self.u_current] == 1:
                from IPython.core.debugger import Pdb; Pdb().set_trace()
                

            
        # update the counter
        self.item_cnt[self.item_selected] = self.item_cnt[self.item_selected] + 1
        
        
        
    
    def initialize_V(self):
        
        self.V = copy.deepcopy(self.I_0)
        self.V_0 = copy.deepcopy(self.I_0)
    
    
    def update_V(self):
        averages = self.emprical_average()
        
        for i in range(self.N): 
            # for all the items that is in V
            if self.V[i] == 1:
                if i == self.item_selected:
                    if np.mod(self.item_cnt[i], self.S_0) == 0:
                        self.cnt_num_tests = self.cnt_num_tests + 1
                        if i > 399:
                            self.cnt_suboptimal_items = self.cnt_suboptimal_items + 1
                    
            
            # for all the items that is in V
            if self.V[i] == 1:
                
                # do a test in every S_0 periods
                if np.mod(self.item_cnt[i], self.S_0) == 0:
                   
                    # included in I_1 if rho_1 > p_1 - Delta/2
                    if averages[i] > max(self.cluster_average) - self.Delta_0 / 2:
                        pass
                    else:
                        # the item will not be used
                        self.V[i] = 0                      
                    
                    
    def additional_sampling(self):
        S_0 = np.ceil(1/ self.Delta_0**2)
        
        item_cnt_tmp = self.compute_cnt_for_active_items()
        
        # if item is changed, put the previous item into I_0
        if self.item_prev != self.item_selected:
            self.I_0[self.item_prev] = 1
        
        imax = rand_argmax(item_cnt_tmp)
        if self.Xi_current[imax, self.u_current] !=0:
             from IPython.core.debugger import Pdb; Pdb().set_trace()
            
        assert self.Xi_current[imax, self.u_current] == 0
        while(item_cnt_tmp[imax] == self.M):
            print('M reached. change item')
            self.I_0[imax] = 1
            item_cnt_tmp = self.item_cnt  - 2*self.M * self.I_0
            imax = rand_argmax(item_cnt_tmp)      
        
        averages = self.emprical_average()
        
        if item_cnt_tmp[imax] >= S_0 and abs(averages[imax] - max(self.cluster_average)) >= self.Delta_0:
            self.I_0[imax] = 1
            item_cnt_tmp = self.compute_cnt_for_active_items()
            imax = rand_argmax(item_cnt_tmp)   
        assert self.Xi_current[imax, self.u_current] == 0
        
        self.item_selected = imax
        
        # record the number of observations
        self.item_cnt[imax] = self.item_cnt[imax] + 1
        
        # indicate to record the reward
        self.reward_rec_flag = 1
        
    # exclude the items in I_0 from the counter
    def compute_cnt_for_active_items(self):
        return np.array(self.item_cnt) - 2*self.M * np.array(self.I_0) -  2*self.M * self.Xi_u
        
            
        
            
        
        
        
    
            
    
        
        
    
