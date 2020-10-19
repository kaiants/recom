import numpy as np
import random


class RecommendationsSynthetic():
    
    
    def __init__(self, algorithm, statparams, algoparams):
        # Item-User histroy matrix
        # current user
        # Parameters related to the item cluster
        # Parameters related to the user cluster
        # Bernoulli statistical parameters matrix
        (self.horizon, self.M, self.N, self.Alpha, self.Beta, self.K, self.L, self.sigma, self.tau, self.P_kl) = statparams
        
        self.algorithm = algorithm(*algoparams)
        
        # state
        u = self.draw_user_uniform()
        Xi = np.zeros((self.N, self.M)) # set of used (item, user) pairs
        assert Xi.shape == (self.N, self.M)
        self.state_problem = (u, Xi, 0)
        
        # current item
        self.item_selected = 0
        

        
    def simulate_single_step_rewards(self):
        k = int(self.sigma[self.item_selected])
        current_user, _, __ = self.state_problem
        ell = int(self.tau[current_user])
        return np.random.binomial(1, self.P_kl[k - 1, ell - 1]) # bernoulli with param P_kl
    
    def update_state(self, prev_reward):
        (u, Xi, _) = self.state_problem
        # mark current (user, item) pair
        Xi[self.item_selected, u] = Xi[self.item_selected, u] + 1
        # draw a new user
        u = self.draw_user_uniform()
        # state update
        self.state_problem = (u, Xi, prev_reward)
        
        
    def draw_user_uniform(self):
        return random.choices(range(self.M), k=1)[0]
    
    
    
    def simulate(self, exit_condition=None):
        
        rewards = np.zeros(self.horizon)
        play_history = [0 for i in range(self.horizon)]
        state_history = [0 for i in range(self.horizon)]
        cum_regret_history = [0 for i in range(self.horizon)]
        cum_regret = 0
        
        prev_reward = 0
        
        # itereate the simulation for horizon times
        for t in range(self.horizon):
            (u, Xi, prev_reward) = self.state_problem
            Xi_u = Xi[:, u] 
            #choose item
            self.item_selected = self.algorithm.choose_item(self.state_problem)
            if Xi[self.item_selected, u] == 1:
                print('no items!')
                print('sum(Xi_u==0):', end="")
                print(sum(Xi_u==0))
                reward =0
            else:
                #receive a reward
                reward = self.simulate_single_step_rewards()
            
            # record: reward, (user, item) pair,  state
            rewards[t] = reward
            play_history[t] = [self.item_selected, u]
            state_history[t] = self.state_problem
            prev_reward = reward
            
            # compute and record regret
            beta_uni_vec = self.Beta/sum(self.Beta)
            
            p_ellstar = self.P_kl.max(0)
            #instant_regret = np.dot(beta_uni_vec, p_ellstar) - reward
            k_tmp = int(self.sigma[self.item_selected])
            ell_tmp = int(self.tau[u])
            
            instant_regret = np.dot(beta_uni_vec, p_ellstar) - self.P_kl[k_tmp - 1, ell_tmp - 1]
            
            cum_regret = cum_regret + instant_regret
            cum_regret_history[t] = cum_regret
            
            #Update the state
            self.update_state(prev_reward)
            
            if np.mod(t, 10000) == 0:
                print('t=', end="")
                print(t)

            
        print('sum(sum(Xi))=',end="")
        print(sum(sum(Xi)))
        print('max(Xi)=',end="")
        print(np.max(Xi))
        
        return cum_regret_history, play_history
    
    
    
class Recommendations_Netflix():
        def __init__(self):
            return 0
    
    

class Recommendations_Movielens():
        def __init__(self):
            return 0

    
    
    

    
        