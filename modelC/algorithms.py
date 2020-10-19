import numpy as np
import random
# import optspace
from functions import rand_argmin, rand_argmax, lowrank_approx, compute_second_eigenvector
from kullback_leibler import klucbBern
import copy

class ProblemParameters():
    def __init__(self):
        return 0
        
# algorithm for Model C
class BiClusterRecommend():
    def __init__(self, horizon, s, s_user, s_u_star, T_0, T_1, epsilon_alg, M, N, K, L, coef_of_hyothesis_testing):
        self.N = N
        self.M = M
        self.K = K
        self.horizon = horizon
        self.t = 0 #         self.t = T_0 #algorithm's time counter
        
        self.s = s
        self.s_user = s_user
        self.s_u_star = s_u_star
        self.T_0 = T_0 #duration of exploration for the item clustering
        self.T_1 = T_1
        
        self.ind_user_clust = 0

        self.item_cnt = [0 for i in range(N)] # number of observations for each item 
        self.item_reward_sum = [0 for i in range(N)] # sum of rewards collected

        self.item_selected = 0 #current item
        self.item_prev = 0 #previously chosen item
        self.prev_reward =0 # previous reward
        self.item_user_reward_matrix = np.zeros((self.N, self.M)) # store all of the reward history of item user combinations.
        self.Xi_u = [0 for i in range(N)] 
        self.Xi_current = []
        self.u_current = 0
        self.u_prev = 0
        self.LARGE_CONST = 30
        self.A_obs_cnt = 0
        
        self.S = [0 for i in range(self.N)] # 1 if item is in S, 0 otherwise. 
        self.S_to_N =[0 for i in range(self.s)] # mapping from index of S to the index of N
        self.N_to_S =[0 for i in range(self.N)] # mapping from index of N to the index of S, if not in S, set as self.N +1
        self.num_fromS_for_u = [0 for i in range(self.M)] # number sampled from S
        
        self.U_0 = [0 for i in range(self.M)] # 1 if item is in U_0, 0 otherwise.
        self.U_star = [0 for i in range(self.M)] # 1 if item is in U_star, 0 otherwise.
        
        self.two_feedbacks = np.zeros((self.M, 2)) # store first two feedbacks from the set of items in S for each user
        self.two_feedbacks_item_pairs = (self.N+1)*np.ones((self.M, 2)) # store first two item comes from S for each user
        self.cluster_estimate = [0 for i in range(self.s)] # item cluster estimate by the algorithm
        
        self.P_kl = np.zeros((2,2)) #\hat{p}(i, j) in the paper, for the item clustering
        self.P_kl_user = np.zeros((2,2)) # for the user clustering
        self.argmax_k = np.zeros(2)# for the user clustering
        
        self.rho_users = np.zeros((self.M, self.K))  # empirical averages of users.
        self.eps_users = 1 # user clustering error (updated only after the each time of the user clustering)
        
        self.cluster_flag = 0 #flag for the item clustering
        self.random_sampling_flag = 0 #flag for the random subset sampling of users and items
        
        self.p_q = 0 # estimate on pq
        self.p2_q2_over_2 = 0 # estimate on (p^2 + q^2)/2
        
        self.thre_estimate = 0 # estimate of (p+q)/2
        
        self.k_prev = [0 for i in range(self.M)] # previously used cluster for each user
        
        self.I_S = [0 for i in range(self.s)] # estimated cluster in S
        self.I =  [0 for i in range(self.N)] # estimated cluster in items
        
        self.epsilon_alg = epsilon_alg # thresholding for the spectral clustering

    
    def choose_item(self, state):
        
        # expand the state
        (self.u_current, self.Xi_current, self.prev_reward) = state
        self.Xi_u = self.Xi_current[:, self.u_current] 
        
        assert len(self.Xi_u) == self.N
        
        if self.random_sampling_flag ==0:
            # 1. item sampling, user sampling
            self.random_S_selection()
            self.random_U_0_selection()
            self.random_sampling_flag = 1
        
        # collect the first two feedbacks from the users.
        self.update_reward_sum()
        # compute user's empirical average
        self.compute_user_average() 
            
        # 2. random recommendation from the set S for first 10m users
        if self.t <= self.T_0:
            # pure exploration for the item clustering
            self.explorations()
        else:
            if self.cluster_flag ==0:
                # 3. item clustering 
                self.do_clustering2()
                self.cluster_flag = 1
                print('Item clustering')
            
            # user clustering 
            if  (self.t <= self.T_1) and int(self.T_0/self.M) + np.power(2, self.ind_user_clust) * self.M + 1 == self.t :
                self.ind_user_clust =  self.ind_user_clust + 1
                print('User clustering')
                print('user clustering index', end='')
                print(self.ind_user_clust)
                self.do_user_clustering() # do user clustering and record the estimated statistical parameters
            
            self.exploitation()
            

        # increment time
        self.t = self.t + 1
        
        # record previously used item and user
        self.item_prev = self.item_selected
        self.u_prev = self.u_current
        
        return self.item_selected
    
    def random_S_selection(self):
        indices = random.sample(range(self.N), self.s)
        for index in indices:
            self.S[index] = 1
            
        # compute self.N_to_S and self.S_to_N
        # 
        scnt = 0
        for i in range(self.N):
            if self.S[i] == 1:
                self.N_to_S[i] = scnt
                scnt = scnt + 1
            else:
                self.N_to_S[i] = self.N +1
        assert scnt == self.s
        scnt = 0
        for i in range(self.N):
            if self.S[i] == 1:
                self.S_to_N[scnt] = i
                scnt = scnt + 1
            else:
                pass
        assert scnt == self.s

    def random_U_0_selection(self):
        indices = random.sample(range(self.M), self.s_user)
        for index in indices:
            self.U_0[index] = 1
        #from IPython.core.debugger import Pdb; Pdb().set_trace()
        
        
        
    def update_reward_sum(self):
        if self.t > 0:
            # record reward
            self.item_reward_sum[self.item_prev] = self.item_reward_sum[self.item_prev] + self.prev_reward
            
            #record reward in the matrix
            self.item_user_reward_matrix[self.item_prev, self.u_prev] =  self.prev_reward
        
            # for the items in S
            if self.N_to_S[self.item_prev] != self.N +1:
                # collect first two feedback from S for each user 
                self.num_fromS_for_u[self.u_prev] = self.num_fromS_for_u[self.u_prev] + 1
                if self.num_fromS_for_u[self.u_prev] < 3:
                    self.two_feedbacks[self.u_prev, self.num_fromS_for_u[self.u_prev] - 1] = self.prev_reward
                    self.two_feedbacks_item_pairs[self.u_prev, self.num_fromS_for_u[self.u_prev] - 1] = self.item_prev
                    
            

                
    def explorations(self):
        
        #remove the previously recommended items for the current user from the candidate
        #candidates: element that is previously recommended items or not in S is made 0, othewise elements are 1 (recommendable)
        candidates = np.multiply(np.array(self.Xi_u == 0) ,np.array(np.array(self.S) == 1))       

        # If there are recommendable item from S,
        if sum(candidates) > 0:
            
            # select item from candidates, 
            self.item_selected = rand_argmax(candidates)
            assert self.Xi_current[self.item_selected, self.u_current] == 0
            
        else:
            # do a random sampling from a new item
            self.item_selected = rand_argmin(self.Xi_u)
            
            assert self.Xi_current[self.item_selected, self.u_current] == 0
            print('random sampling')
            
        #update the counter
        self.item_cnt[self.item_selected] = self.item_cnt[self.item_selected] + 1
        
        
    def emprical_average(self):
        return np.divide(self.item_reward_sum, self.item_cnt)
    
    # item clustering
    def do_clustering(self):
        A = self.generate_A() # adj matrix (s times s)
        p_tilde = 2*np.sum(A)/self.s /(self.s - 1)
        A_low = lowrank_approx(A, self.K) # rank-2 approximation
        
        r_t = [0 for ind in range(int(np.floor(np.log(self.s))) )]
        for ind in range(int(np.floor(np.log(self.s))) ):
    
            # find neighborhoods
            Q = [0 for i in range(self.s)]
            for i in range(self.s):
                Q[i] = set()
                for j in range(self.s):
                    if np.linalg.norm(A_low[i] - A_low[j])**2 <= (ind + 1) * p_tilde * self.epsilon_alg:
                        Q[i].add(j)
                        
            T = [0 for i in range(self.K)]
            xi = np.zeros((self.K, self.s))
            Qprev = set()
            for k in range(self.K):
                cardinalities = [0 for i in range(self.s)]
                for i in range(self.s):
                    cardinalities[i] = len(Q[i] -  Qprev)
                
                # compute the index v_k^\star
                v_k = rand_argmax(np.array(cardinalities))
                T[k] = Q[v_k] -  Qprev
                Qprev = Qprev.union(Q[v_k])
                for i in range(self.s):
                    if i in T[k]:
                        xi[k] = xi[k] + A_low[i]/len(T[k])
            
            # remaining items assignment
            if len(Qprev) != self.s:
                for v in set(range(self.s)) - Qprev:
                    distances = np.zeros(self.K)
                    for k in range(self.K):
                        distances[k] =  np.linalg.norm(A_low[v] - xi[k])**2

                    k_star = rand_argmax(distances)
                    T[k_star].add(v)

            #compute r_t
            for k in range(self.K):
                for i in range(self.s):
                    if i in T[k]:
                        r_t[ind]  = r_t[ind] + np.linalg.norm(A_low[v] - xi[k])**2
                        
        #end for ind...
        minind = rand_argmin(np.array(r_t))
        ind = minind # do a clustering with a smallerst error
        
        # do a clustering again with minind
        # find neighborhoods
        Q = [0 for i in range(self.s)]
        for i in range(self.s):
            Q[i] = set()
            for j in range(self.s):
                if np.linalg.norm(A_low[i] - A_low[j])**2 <= (ind + 1) * p_tilde * self.epsilon_alg:
                    Q[i].add(j)

        T = [0 for i in range(self.K)]
        xi = np.zeros((self.K, self.s))
        Qprev = set()
        for k in range(self.K):
            cardinalities = [0 for i in range(self.s)]
            for i in range(self.s):
                cardinalities[i] = len(Q[i] -  Qprev)
                
            # compute the index v_k^\star
            v_k = rand_argmax(np.array(cardinalities))
            T[k] = Q[v_k] -  Qprev
            Qprev = Qprev.union(Q[v_k])
            for i in range(self.s):
                if i in T[k]:
                    xi[k] = xi[k] + A_low[i]/len(T[k])
            
        
        # remaining items assignment
        if len(Qprev) != self.s:
            for v in set(range(self.s)) - Qprev:
                distances = np.zeros(self.K)
                for k in range(self.K):
                    distances[k] =  np.linalg.norm(A_low[v] - xi[k])**2

                k_star = rand_argmin(distances)
                T[k_star].add(v)
       
        for k in range(self.K):
            for i in T[k]:
                self.I_S[i] = k+1
        
        for i in range(self.N):
            if self.S[i] == 1:
                self.I[i] = self.I_S[self.N_to_S[i]]
                
        # for the debug
        err_num = 0
        for i in range(self.N):
            if i <= int(self.N/2 - 1):
                if self.I[i] == 2:
                    err_num = err_num + 1
            
            if i > int(self.N/2 - 1): 
                if self.I[i] == 1:
                    err_num = err_num + 1
            
            
        err_rate = min(err_num/self.s, 1 - err_num/self.s)
        print('err_rate after SC=', end="")
        print(err_rate)

            
        #         print('self.A_obs_cnt = ', end="")
        #         print(self.A_obs_cnt)
        #         print('ratio self.A_obs_cnt/self.s**2 = ', end="")
        #         print(self.A_obs_cnt/self.s**2)

        #estimation of \hat{p}(i, j)
        for i in range(self.K):
            for j in range(self.K):
                numerator = 0
                for v in T[i]:
                    for u in T[j]:
                        numerator = numerator + A[v, u]
                denominator = len(T[i]) * self.s
                self.P_kl[i, j]  = numerator/denominator
                
        

        
        
        # local improvement        
        S = [0 for i in range(self.K)]
        Sprev = [0 for i in range(self.K)]
        for k in range(self.K):
            Sprev[k] = T[k]
        
        for ind in range(int(np.floor(np.log(self.s))) ):
            for k in range(self.K):
                S[k] = set()
            
            for v in range(self.s):
                
                # computation of likelihood
                likelihoods = np.zeros(self.K)
                for i in range(self.K):
                    # sum up over all k
                    wegihtsum = 0
                    psum = 0
                    for k in range(self.K):
                        
                        weight_by_Avw = 0
                        
                        for w in Sprev[i]:
                            weight_by_Avw = weight_by_Avw + A[v, w] 
                        wegihtsum = wegihtsum + weight_by_Avw
                        psum = psum + self.P_kl[i, k]
                        likelihoods[i] = likelihoods[i] + weight_by_Avw * np.log(self.P_kl[i, k]) 
                    
                    # add the case of k = 0 (in the paper's notations)
                    likelihoods[i] = likelihoods[i] + (self.s - wegihtsum) * (1 - psum)
                    
                # maximum likelihood
                i_star = rand_argmax(likelihoods)
                S[i_star].add(v)
            
            #update Sprev
            for k in range(self.K):
                Sprev[k] = S[k]
        
        # (end for ind loop)

          
        for k in range(self.K):
            for i in S[k]:
                self.I_S[i] = k+1
        
        for i in range(self.N):
            if self.S[i] == 1:
                self.I[i] = self.I_S[self.N_to_S[i]]
                
        # for the debug (compuation of err rate)
        err_num = 0
        for i in range(self.N):
            if i <= int(self.N/2 - 1):
                if self.I[i] == 2:
                    err_num = err_num + 1
            
            if i > int(self.N/2 - 1): 
                if self.I[i] == 1:
                    err_num = err_num + 1
            
            
        err_rate2 = min(err_num/self.s, 1 - err_num/self.s)
        print('err_rate after SP=', end="")
        print(err_rate2)
        print('err_rate improvement = ', end="")
        print(err_rate - err_rate2)
        
    
    # for comparison of the userclustering part only. (give true item cluster indexes)
    def do_clustering2(self):
        for i in range(self.N):
            if i <= int(self.N/2 - 1):
                self.I[i] = 1
            
            if i > int(self.N/2 - 1): 
                self.I[i] = 2

                    
    def generate_A(self):
        A = np.zeros((self.s, self.s)) # observation matrix
        
        for u in range(self.M):
             if self.two_feedbacks_item_pairs[u, 1] != self.N + 1:
                    i, j = int(self.two_feedbacks_item_pairs[u, 0]), int(self.two_feedbacks_item_pairs[u, 1])
                    assert i <self.N+1
                    assert j < self.N +1
                    assert self.S[i] ==1
                    assert self.S[j] ==1
                 
                    i = self.N_to_S[i]
                    j = self.N_to_S[j]
                    self.A_obs_cnt = self.A_obs_cnt + 1
                    if self.two_feedbacks[u, 0] ==1 and self.two_feedbacks[u, 1] ==1:
                        
                        A[i, j]= A[i, j] + 1
                        A[j, i]= A[j, i] + 1
                        
        return A
        
        
        
    def exploitation(self):
        #round robbin recommendations
        if self.U_0[self.u_current] == 1 and self.t <= self.T_1:
            # do a recommendations in a round robbins manner
            if self.k_prev[self.u_current ] == 1:
                # recom from 2
                k = 2
                self.k_prev[self.u_current ] = 2
                recommendable_from_I_k = np.multiply(np.array(self.Xi_u) == 0, np.array(self.I) == k)
                if sum(recommendable_from_I_k) > 0:
                    self.item_selected = rand_argmax(recommendable_from_I_k)
                else:
                    self.item_selected = rand_argmin(self.Xi_u)
                    #print('random sampling (exploi)')
            else:
                # recom from 1
                k = 1
                self.k_prev[self.u_current ] = 1
                recommendable_from_I_k = np.multiply(np.array(self.Xi_u) == 0, np.array(self.I) == k)
                if sum(recommendable_from_I_k) > 0:
                    self.item_selected = rand_argmax(recommendable_from_I_k)
                else:
                    self.item_selected = rand_argmin(self.Xi_u)
                    #print('random sampling (exploi)')
        #end round robbin
        else: # exploitation using L
            x_kl  = np.zeros((self.K, 2))
            for k in range(self.K):
                for l in range(2):
                    x_kl[k, l] = np.max([np.abs(self.P_kl_user[k, l] - self.rho_users[self.u_current, k]) - self.eps_users, 0])
            L_ind = set()
            for l in range(2):
                term = 0
                for k in range(self.K):
                    cnt_k = np.sum(np.multiply(np.array(self.Xi_u), np.array(self.I) == k+1))
                    term = term + cnt_k * x_kl[k, l]**2
                cnt_user = np.sum(np.array(self.Xi_u))
                if term < 0.01 * np.log(cnt_user):
                    L_ind.add(l)
            
                
            
            
            recom_k = 0
            #print('len(L_ind) = ', end="")
            #print(len(L_ind))
            if len(L_ind) != 0:
                setbestk = set()
                for l in L_ind:
                    setbestk.add(self.argmax_k[l])
                #random.sample(L_ind,1)

                recom_k = random.sample(setbestk, 1)
            else:
                recom_k =  random.sample(range(self.K), 1)
            

            # increment k so that it align with the actual index
            recom_k = recom_k[0] + 1
            
            if recom_k ==1 or recom_k==2:
                # compute recommendable items from S to current user 
                recommendable_from_I_k = np.multiply(np.array(self.Xi_u) == 0, np.array(self.I) == recom_k)
                if sum(recommendable_from_I_k) >0:
                    self.item_selected = rand_argmax(recommendable_from_I_k)
                else:
                    self.item_selected = rand_argmin(self.Xi_u)

            else:
                # do a recommendations in a round robbins manner
                if self.k_prev[self.u_current ] == 1:
                    # recom from 2
                    recom_k = 2
                    self.k_prev[self.u_current ] = 2
                    recommendable_from_I_k = np.multiply(np.array(self.Xi_u) == 0, np.array(self.I) == recom_k)
                    if sum(recommendable_from_I_k) > 0:
                        self.item_selected = rand_argmax(recommendable_from_I_k)
                    else:
                        self.item_selected = rand_argmin(self.Xi_u)
                        #print('random sampling (exploi)')
                else:
                    # recom from 1
                    recom_k = 1
                    self.k_prev[self.u_current ] = 1
                    recommendable_from_I_k = np.multiply(np.array(self.Xi_u) == 0, np.array(self.I) == recom_k)
                    if sum(recommendable_from_I_k) > 0:
                        self.item_selected = rand_argmax(recommendable_from_I_k)
                    else:
                        self.item_selected = rand_argmin(self.Xi_u)
                        #print('random sampling (exploi)')
                
            #if self.ind_user_clust > 2:
            #   from IPython.core.debugger import Pdb; Pdb().set_trace()
        #update the counter
        self.item_cnt[self.item_selected] = self.item_cnt[self.item_selected] + 1
        
        assert self.Xi_current[self.item_selected, self.u_current] == 0
        

    def compute_eps_users(self):
        return np.sqrt(self.M / self.t * np.log(self.t/self.M))  #self.K * np.sqrt(8 * self.K * self.M / self.t * np.log(self.t/self.M)) 
                          

    def compute_prev_user_average(self):
        # for cluster 1
        # extract cnt only from cluster 1
        cnt_clust1_for_user = np.sum(np.multiply(np.array(self.Xi_current[:, self.u_prev] ), np.array(self.I) ==1))
        rewardsum_clust1_for_user = np.sum(np.multiply(np.array(self.item_user_reward_matrix[:, self.u_prev]), np.array(self.I) ==1))
        if cnt_clust1_for_user ==0:
            rho_u1 = 1/2
        else:
            rho_u1 = rewardsum_clust1_for_user / cnt_clust1_for_user 
        # for cluster 2
        # extract cnt only from cluster 2
        cnt_clust2_for_user = np.sum(np.multiply(np.array(self.Xi_current[:, self.u_prev] ), np.array(self.I) ==2))
        rewardsum_clust2_for_user = np.sum(np.multiply(np.array(self.item_user_reward_matrix[:, self.u_prev]), np.array(self.I) ==2))
        
        if cnt_clust2_for_user ==0:
            rho_u2 = 1/2
        else:
            rho_u2 = rewardsum_clust2_for_user / cnt_clust2_for_user
        
        
        # compute user's empirical average
        self.rho_users[self.u_prev,0] = rho_u1
        self.rho_users[self.u_prev,1] = rho_u2
        
    def compute_user_average(self):
        # for cluster 1
        # extract cnt only from cluster 1
        cnt_clust1_for_user = np.sum(np.multiply(np.array(self.Xi_u), np.array(self.I) ==1))
        rewardsum_clust1_for_user = np.sum(np.multiply(np.array(self.item_user_reward_matrix[:, self.u_current]), np.array(self.I) ==1))
        if cnt_clust1_for_user ==0:
            rho_u1 = 1/2
        else:
            rho_u1 = rewardsum_clust1_for_user / cnt_clust1_for_user 
        # for cluster 2
        # extract cnt only from cluster 2
        cnt_clust2_for_user = np.sum(np.multiply(np.array(self.Xi_u), np.array(self.I) ==2))
        rewardsum_clust2_for_user = np.sum(np.multiply(np.array(self.item_user_reward_matrix[:, self.u_current]), np.array(self.I) ==2))
        
        if cnt_clust2_for_user ==0:
            rho_u2 = 1/2
        else:
            rho_u2 = rewardsum_clust2_for_user / cnt_clust2_for_user
        
        
        # compute user's empirical average
        self.rho_users[self.u_current,0] = rho_u1
        self.rho_users[self.u_current,1] = rho_u2
        
    def compute_user_averages(self):
        for u in range(self.M):
            # for cluster 1
            # extract cnt only from cluster 1
            cnt_clust1_for_user = np.sum(np.multiply(np.array(self.Xi_current[:, u] ), np.array(self.I) ==1))
            rewardsum_clust1_for_user = np.sum(np.multiply(np.array(self.item_user_reward_matrix[:, u]), np.array(self.I) ==1))
            if cnt_clust1_for_user ==0:
                rho_u1 = 1/2
            else:
                rho_u1 = rewardsum_clust1_for_user / cnt_clust1_for_user 
            # for cluster 2
            # extract cnt only from cluster 2
            cnt_clust2_for_user = np.sum(np.multiply(np.array(self.Xi_current[:, u]), np.array(self.I) ==2))
            rewardsum_clust2_for_user = np.sum(np.multiply(np.array(self.item_user_reward_matrix[:, u]), np.array(self.I) ==2))

            if cnt_clust2_for_user ==0:
                rho_u2 = 1/2
            else:
                rho_u2 = rewardsum_clust2_for_user / cnt_clust2_for_user


            # compute user's empirical average
            self.rho_users[u,0] = rho_u1
            self.rho_users[u,1] = rho_u2      
        
    def do_user_clustering(self):
        
        
        self.compute_user_averages()
        
        
        self.select_U_star()
        self.eps_users = self.compute_eps_users()
        
        # find neighborhoods
        Q = [0 for u in range(self.M)]
        for u in range(self.M):
            Q[u] = set()
            for v in range(self.M):
                if np.linalg.norm(self.rho_users[u,:] - self.rho_users[v,:]) <= self.eps_users:
                    Q[u].add(v)
        Qprev = set()
        for l in range(2):
            cardinalities = [0 for y in range(self.M)]
            for u in range(self.M):
                cardinalities[u] = len(Q[u] - Qprev)
                
            u_l = rand_argmax(np.array(cardinalities))
            self.P_kl_user[:, l] = np.transpose(self.rho_users[u_l,:])
            Qprev = Qprev.union(Q[u_l])

        for l in range(2):
            self.argmax_k[l] = rand_argmax(self.P_kl_user[:, l])
            
        
            
            
    # select U_star from U_0
    def select_U_star(self):
        U_0_tmp = self.U_0.copy()
        for ind in range(self.s_u_star):
            cnt_users_star = np.zeros(self.M)
            #cnt_users_star = np.sum(np.multiply(np.array(self.Xi_current[:, u]), np.array(U_0_tmp) == 1))
            for u in range(self.M):
                cnt_users_star[u] =  np.sum(np.array(self.Xi_current[:, u])) *  U_0_tmp[u] - 1 * np.array(U_0_tmp[u]==0)
            
            
            new_u = rand_argmax(cnt_users_star)
            self.U_star[new_u] = 1
            U_0_tmp[new_u] = 0
            
                                        
                
                
            
            
            
            
        
        
        
        
        

            
        
        
    
        
        
# comparison algorithm for Model C (UCB1)
class ClusterUCB1():
    def __init__(self, horizon, s, s_user, T_0, epsilon_alg, M, N, K, L, coef_of_hyothesis_testing):
        self.N = N
        self.M = M
        self.K = K
        self.horizon = horizon
        self.t = 0 #algorithm's time counter
        self.s = s
        self.T_0 = T_0

        self.item_cnt = [0 for i in range(N)] # number of observations for each item 
        self.item_reward_sum = [0 for i in range(N)] # sum of rewards collected

        self.item_selected = 0 #current item
        self.item_prev = 0 #previously chosen item
        self.prev_reward =0 # previous reward
        self.item_user_reward_matrix = np.zeros((self.N, self.M)) # store all of the reward history of item user combinations.
        self.Xi_u = [0 for i in range(N)] 
        self.Xi_current = []
        self.u_current = 0
        self.u_prev = 0
        self.LARGE_CONST = 30
        self.A_obs_cnt = 0
        
        self.S = [0 for i in range(self.N)] # 1 if item is in S, 0 otherwise. 
        self.S_to_N =[0 for i in range(self.s)] # mapping from index of S to the index of N
        self.N_to_S =[0 for i in range(self.N)] # mapping from index of N to the index of S, if not in S, set as self.N +1
        self.num_fromS_for_u = [0 for i in range(self.M)] # number sampled from S
        
        self.two_feedbacks = np.zeros((self.M, 2)) # store first two feedbacks from the set of items in S for each user
        self.two_feedbacks_item_pairs = (self.N+1)*np.ones((self.M, 2)) # store first two item comes from S for each user
        self.cluster_estimate = [0 for i in range(self.s)] # cluster estimate of items by the algorithm
        
        self.P_kl = np.zeros((2,2))
        
        self.cluster_flag = 0
        
        self.k_prev = [0 for i in range(self.M)] # previously used cluster for each user
        
        self.I_S = [0 for i in range(self.s)] # estimated cluster in S
        self.I =  [0 for i in range(self.N)] # estimated cluster in items
        
        self.epsilon_alg = epsilon_alg # thresholding for the spectral clustering
        
        
    def choose_item(self, state):
        
        # expand the state
        (self.u_current, self.Xi_current, self.prev_reward) = state
        self.Xi_u = self.Xi_current[:, self.u_current] 
        
        assert len(self.Xi_u) == self.N
        
        if self.t ==0:
            # 1. item sampling
            self.random_S_selection()

        self.update_reward_sum()
            
        # 2. random recommendation from the set S for first 10m users
        if self.t <= self.T_0:
            self.explorations()
        else:
            if self.cluster_flag ==0:
                # 3. clustering 
                #self.do_clustering()
                self.do_clustering2()
                
                self.cluster_flag = 1

            #4. Run UCB1
            self.exploitation()

        # increment time
        self.t = self.t + 1
        
        # record previously used item and user
        self.item_prev = self.item_selected
        self.u_prev = self.u_current
        
        return self.item_selected
    
    def random_S_selection(self):
        indices = random.sample(range(self.N), self.s)
        for index in indices:
            self.S[index] = 1
            
        # compute self.N_to_S and self.S_to_N
        # 
        scnt = 0
        for i in range(self.N):
            if self.S[i] == 1:
                self.N_to_S[i] = scnt
                scnt = scnt + 1
            else:
                self.N_to_S[i] = self.N +1
        assert scnt == self.s
        scnt = 0
        for i in range(self.N):
            if self.S[i] == 1:
                self.S_to_N[scnt] = i
                scnt = scnt + 1
            else:
                pass
        assert scnt == self.s

    def update_reward_sum(self):
        if self.t > 0:
            # record reward
            self.item_reward_sum[self.item_prev] = self.item_reward_sum[self.item_prev] + self.prev_reward
            
            #record reward in the matrix
            self.item_user_reward_matrix[self.item_prev, self.u_prev] =  self.prev_reward
        
            # for the items in S
            if self.N_to_S[self.item_prev] != self.N +1:
                # collect first two feedback from S for each user 
                self.num_fromS_for_u[self.u_prev] = self.num_fromS_for_u[self.u_prev] + 1
                if self.num_fromS_for_u[self.u_prev] < 3:
                    self.two_feedbacks[self.u_prev, self.num_fromS_for_u[self.u_prev] - 1] = self.prev_reward
                    self.two_feedbacks_item_pairs[self.u_prev, self.num_fromS_for_u[self.u_prev] - 1] = self.item_prev
                    
            

                
    def explorations(self):
        
        #remove the previously recommended items for the current user from the candidate
        #candidates: element that is previously recommended items or not in S is made 0, othewise elements are 1 (recommendable)
        candidates = np.multiply(np.array(self.Xi_u == 0) ,np.array(np.array(self.S) == 1))         

        # If there are recommendable item from S,
        if sum(candidates) > 0:
            
            # select item from candidates, 
            self.item_selected = rand_argmax(candidates)
            assert self.Xi_current[self.item_selected, self.u_current] == 0
            
        else:
            # do a random sampling from a new item
            self.item_selected = rand_argmin(self.Xi_u)
            
            assert self.Xi_current[self.item_selected, self.u_current] == 0
            print('random sampling')
            
        #update the counter
        self.item_cnt[self.item_selected] = self.item_cnt[self.item_selected] + 1
        
        
    def emprical_average(self):
        return np.divide(self.item_reward_sum, self.item_cnt)
    
        
    def do_clustering(self):
        A = self.generate_A() # adj matrix (s times s)
        p_tilde = 2*np.sum(A)/self.s /(self.s - 1)
        A_low = lowrank_approx(A, self.K) # rank-2 approximation
        
        r_t = [0 for ind in range(int(np.floor(np.log(self.s))) )]
        for ind in range(int(np.floor(np.log(self.s))) ):
    
            # find neighborhoods
            Q = [0 for i in range(self.s)]
            for i in range(self.s):
                Q[i] = set()
                for j in range(self.s):
                    if np.linalg.norm(A_low[i] - A_low[j])**2 <= (ind + 1) * p_tilde * self.epsilon_alg:
                        Q[i].add(j)
                        
            T = [0 for i in range(self.K)]
            xi = np.zeros((self.K, self.s))
            Qprev = set()
            for k in range(self.K):
                cardinalities = [0 for i in range(self.s)]
                for i in range(self.s):
                    cardinalities[i] = len(Q[i] -  Qprev)
                
                # compute the index v_k^\star
                v_k = rand_argmax(np.array(cardinalities))
                T[k] = Q[v_k] -  Qprev
                Qprev = Qprev.union(Q[v_k])
                for i in range(self.s):
                    if i in T[k]:
                        xi[k] = xi[k] + A_low[i]/len(T[k])
            
            # remaining items assignment
            if len(Qprev) != self.s:
                for v in set(range(self.s)) - Qprev:
                    distances = np.zeros(self.K)
                    for k in range(self.K):
                        distances[k] =  np.linalg.norm(A_low[v] - xi[k])**2

                    k_star = rand_argmax(distances)
                    T[k_star].add(v)

            #compute r_t
            for k in range(self.K):
                for i in range(self.s):
                    if i in T[k]:
                        r_t[ind]  = r_t[ind] + np.linalg.norm(A_low[v] - xi[k])**2
                        
        #end for ind...
        minind = rand_argmin(np.array(r_t))
        ind = minind # do a clustering with a smallerst error
        
        # do a clustering again with minind
        # find neighborhoods
        Q = [0 for i in range(self.s)]
        for i in range(self.s):
            Q[i] = set()
            for j in range(self.s):
                if np.linalg.norm(A_low[i] - A_low[j])**2 <= (ind + 1) * p_tilde * self.epsilon_alg:
                    Q[i].add(j)

        T = [0 for i in range(self.K)]
        xi = np.zeros((self.K, self.s))
        Qprev = set()
        for k in range(self.K):
            cardinalities = [0 for i in range(self.s)]
            for i in range(self.s):
                cardinalities[i] = len(Q[i] -  Qprev)
                
            # compute the index v_k^\star
            v_k = rand_argmax(np.array(cardinalities))
            T[k] = Q[v_k] -  Qprev
            Qprev = Qprev.union(Q[v_k])
            for i in range(self.s):
                if i in T[k]:
                    xi[k] = xi[k] + A_low[i]/len(T[k])
            
        
        # remaining items assignment
        if len(Qprev) != self.s:
            #from IPython.core.debugger import Pdb; Pdb().set_trace()
            for v in set(range(self.s)) - Qprev:
                distances = np.zeros(self.K)
                for k in range(self.K):
                    distances[k] =  np.linalg.norm(A_low[v] - xi[k])**2

                k_star = rand_argmin(distances)
                T[k_star].add(v)
       
        for k in range(self.K):
            for i in T[k]:
                self.I_S[i] = k+1
        
        for i in range(self.N):
            if self.S[i] == 1:
                self.I[i] = self.I_S[self.N_to_S[i]]
                
        # for the debug
        err_num = 0
        for i in range(self.N):
            if i <= int(self.N/2 - 1):
                if self.I[i] == 2:
                    err_num = err_num + 1
            
            if i > int(self.N/2 - 1): 
                if self.I[i] == 1:
                    err_num = err_num + 1
            
            
        err_rate = min(err_num/self.s, 1 - err_num/self.s)
        print('err_rate after SC=', end="")
        print(err_rate)

            
        #         print('self.A_obs_cnt = ', end="")
        #         print(self.A_obs_cnt)
        #         print('ratio self.A_obs_cnt/self.s**2 = ', end="")
        #         print(self.A_obs_cnt/self.s**2)

        #estimation of \hat{p}(i, j)
        for i in range(self.K):
            for j in range(self.K):
                numerator = 0
                for v in T[i]:
                    for u in T[j]:
                        numerator = numerator + A[v, u]
                denominator = len(T[i]) * self.s
                self.P_kl[i, j]  = numerator/denominator
                
        

        
        
        # local improvement        
        S = [0 for i in range(self.K)]
        Sprev = [0 for i in range(self.K)]
        for k in range(self.K):
            Sprev[k] = T[k]
        
        for ind in range(int(np.floor(np.log(self.s))) ):
            for k in range(self.K):
                S[k] = set()
            
            for v in range(self.s):
                
                # computation of likelihood
                likelihoods = np.zeros(self.K)
                for i in range(self.K):
                    # sum up over all k
                    wegihtsum = 0
                    psum = 0
                    for k in range(self.K):
                        
                        weight_by_Avw = 0
                        
                        for w in Sprev[i]:
                            weight_by_Avw = weight_by_Avw + A[v, w] 
                        wegihtsum = wegihtsum + weight_by_Avw
                        psum = psum + self.P_kl[i, k]
                        likelihoods[i] = likelihoods[i] + weight_by_Avw * np.log(self.P_kl[i, k]) 
                    
                    # add the case of k = 0 (in the paper's notations)
                    likelihoods[i] = likelihoods[i] + (self.s - wegihtsum) * (1 - psum)
                    
                # maximum likelihood
                i_star = rand_argmax(likelihoods)
                S[i_star].add(v)
            
            #update Sprev
            for k in range(self.K):
                Sprev[k] = S[k]
        
        # (end for ind loop)
        
            
            
                
          
        for k in range(self.K):
            for i in S[k]:
                self.I_S[i] = k+1
        
        for i in range(self.N):
            if self.S[i] == 1:
                self.I[i] = self.I_S[self.N_to_S[i]]
                
        # for the debug (compuation of err rate)
        err_num = 0
        for i in range(self.N):
            if i <= int(self.N/2 - 1):
                if self.I[i] == 2:
                    err_num = err_num + 1
            
            if i > int(self.N/2 - 1): 
                if self.I[i] == 1:
                    err_num = err_num + 1
            
            
        err_rate2 = min(err_num/self.s, 1 - err_num/self.s)
        print('err_rate after SP=', end="")
        print(err_rate2)
        print('err_rate improvement = ', end="")
        print(err_rate - err_rate2)
        
    # for comparison of the userclustering part only. (give true item cluster indexes)
    def do_clustering2(self):
        for i in range(self.N):
            if i <= int(self.N/2 - 1):
                self.I[i] = 1
            
            if i > int(self.N/2 - 1): 
                self.I[i] = 2  
        
    def generate_A(self):
        A = np.zeros((self.s, self.s)) # observation matrix
        
        for u in range(self.M):
             if self.two_feedbacks_item_pairs[u, 1] != self.N + 1:
                    i, j = int(self.two_feedbacks_item_pairs[u, 0]), int(self.two_feedbacks_item_pairs[u, 1])
                    assert i <self.N+1
                    assert j < self.N +1
                    assert self.S[i] ==1
                    assert self.S[j] ==1
                 
                    i = self.N_to_S[i]
                    j = self.N_to_S[j]
                    self.A_obs_cnt = self.A_obs_cnt + 1
                    if self.two_feedbacks[u, 0] ==1 and self.two_feedbacks[u, 1] ==1:
                        
                        A[i, j]= A[i, j] + 1
                        A[j, i]= A[j, i] + 1
                        
        return A
        
        
        # run UCB1 for each user
    def exploitation(self):
        
        k = self.compute_UCB_ind() 

        
      
        if k ==1 or k==2:
            # compute recommendable items from S to current user 
            recommendable_from_I_k = np.multiply(np.array(self.Xi_u) == 0, np.array(self.I) == k)
            if sum(recommendable_from_I_k) >0:
                self.item_selected = rand_argmax(recommendable_from_I_k)
            else:
                self.item_selected = rand_argmin(self.Xi_u)
                
        else:
            # do a recommendations in a round robbins manner
            if self.k_prev[self.u_current ] == 1:
                # recom from 2
                k = 2
                self.k_prev[self.u_current ] = 2
                recommendable_from_I_k = np.multiply(np.array(self.Xi_u) == 0, np.array(self.I) == k)
                if sum(recommendable_from_I_k) > 0:
                    self.item_selected = rand_argmax(recommendable_from_I_k)
                else:
                    self.item_selected = rand_argmin(self.Xi_u)
                    #print('random sampling (exploi)')
            else:
                # recom from 1
                k = 1
                self.k_prev[self.u_current ] = 1
                recommendable_from_I_k = np.multiply(np.array(self.Xi_u) == 0, np.array(self.I) == k)
                if sum(recommendable_from_I_k) > 0:
                    self.item_selected = rand_argmax(recommendable_from_I_k)
                else:
                    self.item_selected = rand_argmin(self.Xi_u)
                    #print('random sampling (exploi)')
                    
        #update the counter
        self.item_cnt[self.item_selected] = self.item_cnt[self.item_selected] + 1
        
        assert self.Xi_current[self.item_selected, self.u_current] == 0
        
    def compute_UCB_ind(self):
        # for cluster 1
        # extract cnt only from cluster 1
        cnt_clust1_for_user = np.sum(np.multiply(np.array(self.Xi_u), np.array(self.I) ==1))
        rewardsum_clust1_for_user = np.sum(np.multiply(np.array(self.item_user_reward_matrix[:, self.u_current]), np.array(self.I) ==1))
        rho_u1 = rewardsum_clust1_for_user / cnt_clust1_for_user
        UCBind_1 = rho_u1 + np.sqrt(2 * np.log(self.t) / cnt_clust1_for_user)
        # for cluster 2
        # extract cnt only from cluster 2
        cnt_clust2_for_user = np.sum(np.multiply(np.array(self.Xi_u), np.array(self.I) ==2))
        rewardsum_clust2_for_user = np.sum(np.multiply(np.array(self.item_user_reward_matrix[:, self.u_current]), np.array(self.I) ==2))
        rho_u2 = rewardsum_clust2_for_user / cnt_clust2_for_user       
        UCBind_2 = rho_u2 
        
        if cnt_clust1_for_user == 0 or cnt_clust2_for_user == 0:
            return 0
        elif rho_u1 >  rho_u2:
            return 1
        elif rho_u2 >  rho_u1:
            return 2
        else:
            return 0

        
# comparison algorithm for Model C (matrix completion)
class NaiveClusterRecommend():
    def __init__(self, horizon, s, T_0, epsilon_alg, M, N, K, L, coef_of_hyothesis_testing):
        self.N = N
        self.M = M
        self.K = K
        self.horizon = horizon
        self.t = 0 #algorithm's time counter
        self.s = s
        self.T_0 = T_0

        self.item_cnt = [0 for i in range(N)] # number of observations for each item 
        self.item_reward_sum = [0 for i in range(N)] # sum of rewards collected

        self.item_selected = 0 #current item
        self.item_prev = 0 #previously chosen item
        self.prev_reward =0 # previous reward
        self.item_user_reward_matrix = [] # store all of the reward history of item user combinations.
        self.hatA = np.zeros((self.N, self.M)) # estimated matrix
        self.maxhatA = 0
        
        self.Xi_u = [0 for i in range(N)] 
        self.Xi_current = []
        self.u_current = 0
        self.u_prev = 0
        self.LARGE_CONST = 30
        
        self.S = [0 for i in range(self.N)] # 1 if item is in S, 0 otherwise. 
        self.S_to_N =[0 for i in range(self.s)] # mapping from index of S to the index of N
        self.N_to_S =[0 for i in range(self.N)] # mapping from index of N to the index of S, if not in S, set as self.N +1
        self.num_fromS_for_u = [0 for i in range(self.M)] # number sampled from S
        
        self.two_feedbacks = np.zeros((self.M, 2)) # store first two feedbacks from the set of items in S for each user
        self.two_feedbacks_item_pairs = (self.N+1)*np.ones((self.M, 2)) # store first two item comes from S for each user
        self.cluster_estimate = [0 for i in range(self.s)] # cluster estimate by the algorithm
        
        self.cluster_flag = 0
        
        self.p_q = 0 # estimate on pq
        self.p2_q2_over_2 = 0 # estimate on (p^2 + q^2)/2
        
        self.thre_estimate = 0 # estimate of (p+q)/2
        
        self.k_prev = [0 for i in range(self.M)] # previously used cluster for each user
        
        self.I_S = [0 for i in range(self.s)] # estimated cluster in S
        self.I =  [0 for i in range(self.N)] # estimated cluster in items
        
        self.epsilon_alg = epsilon_alg # thresholding for the spectral clustering
        
        self.LARGE_CONSTANT = 30

    
    def choose_item(self, state):
        
        # expand the state
        (self.u_current, self.Xi_current, self.prev_reward) = state
        self.Xi_u = self.Xi_current[:, self.u_current] 
        
        assert len(self.Xi_u) == self.N

        self.update_reward_sum()
            
        # 2. random recommendation from the set S for first 10m users
        if self.t <= self.T_0:
            self.explorations()
        else:
            if self.cluster_flag ==0:
                # 3. matrix completions
                self.do_matrix_completion()
                self.cluster_flag = 1
                
            #4. exploitations
            self.exploitation()

        # increment time
        self.t = self.t + 1
        
        # record previously used item and user
        self.item_prev = self.item_selected
        self.u_prev = self.u_current
        
        return self.item_selected
       
        
    def update_reward_sum(self):
        if self.t > 0:
            # record reward
            self.item_reward_sum[self.item_prev] = self.item_reward_sum[self.item_prev] + self.prev_reward
            
            #record reward in the matrix
            #self.item_user_reward_matrix[self.item_prev, self.u_prev]
            self.item_user_reward_matrix.append((self.item_prev, self.u_prev, self.prev_reward))
        

                
    def explorations(self):
        
        #random explorations
        self.item_selected = rand_argmin(self.Xi_u)
        self.item_cnt[self.item_selected] = self.item_cnt[self.item_selected] + 1

    def do_matrix_completion(self):
        [U,S,V,out_niter] = optspace.optspace(self.item_user_reward_matrix , rank_n= int(self.K), 
                                    num_iter=1000, 
                                    tol=1e-4, 
                                    verbosity=1, 
                                    outfile="")
        self.hatA = (U @ S @ V.T)
        self.maxhatA = np.max(self.hatA)

    def emprical_average(self):
        return np.divide(self.item_reward_sum, self.item_cnt)

    def exploitation(self):

        # compute user's tendency if none, return 0
        A_u = np.multiply(np.array(self.Xi_u)==0, np.array(self.hatA[:, self.u_current])) - np.array(self.LARGE_CONSTANT *self.maxhatA* np.array(self.Xi_u)==1)
        self.item_selected = rand_argmax(A_u)

        # if we cannot select good item, force to do a random sampling.
        if self.Xi_current[self.item_selected, self.u_current] == 1:
            self.item_selected = rand_argmin(self.Xi_u)

        #update the counter
        self.item_cnt[self.item_selected] = self.item_cnt[self.item_selected] + 1

        assert self.Xi_current[self.item_selected, self.u_current] == 0
                          
                    
    def compute_user_tendency(self):
        # for cluster 1
        # extract cnt only from cluster 1
        cnt_clust1_for_user = np.sum(np.multiply(np.array(self.Xi_u), np.array(self.I) ==1))
        rewardsum_clust1_for_user = np.sum(np.multiply(np.array(self.item_user_reward_matrix[:, self.u_current]), np.array(self.I) ==1))
        rho_u1 = rewardsum_clust1_for_user / cnt_clust1_for_user
        # for cluster 2
        # extract cnt only from cluster 2
        cnt_clust2_for_user = np.sum(np.multiply(np.array(self.Xi_u), np.array(self.I) ==2))
        rewardsum_clust2_for_user = np.sum(np.multiply(np.array(self.item_user_reward_matrix[:, self.u_current]), np.array(self.I) ==2))
        rho_u2 = rewardsum_clust2_for_user / cnt_clust2_for_user       
        
        
        if cnt_clust1_for_user == 0 or cnt_clust2_for_user == 0:
            return 0
        elif rho_u1 > self.thre_estimate and rho_u2 < self.thre_estimate:
            return 1
        elif rho_u2 > self.thre_estimate and rho_u1 < self.thre_estimate:
            return 2
        else:
            return 0
              
        
            
        
        
        
    
            
    
        
        
    
