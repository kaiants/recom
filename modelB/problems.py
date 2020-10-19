import numpy as np
import random

class RecommendationsSyntheticContinuous():
    
    
    def __init__(self, algorithm, statparams, algoparams):
        (horizon, N, M, mu_0, mu_1, epsilon_satis_regret) = statparams
        self.horizon = horizon
        self.epsilon_satis_regret = epsilon_satis_regret
        self.M = M
        self.N = N
        self.mu_0 = mu_0
        self.mu_1= mu_1
        self.mu_1_minus_eps = (1 - self.epsilon_satis_regret) * self.mu_1 + self.epsilon_satis_regret * self.mu_0
        self.item_cnt = [0 for i in range(N)] # number of observations for each item 

        
        self.algorithm = algorithm(*algoparams)
        
        # item's statistical paramter initialization
        self.true_avg_item = [0 for i in range(self.N)]
        for i in range(self.N):
            self.true_avg_item[i] = np.random.uniform(self.mu_0, self.mu_1)
        
        # state initialization
        u = self.draw_user_uniform()
        Xi = np.zeros((self.N, self.M)) # set of used (item, user) pairs
        assert Xi.shape == (self.N, self.M)
        self.state_problem = (u, Xi, 0)
        
        # current item
        self.item_selected = 0

    def simulate_single_step_rewards(self):
        avg = self.true_avg_item[self.item_selected]
        return np.random.binomial(1, avg) # bernoulli with param avg
        
        
    def simulate(self):
        rewards = np.zeros(self.horizon)
        play_history = [0 for i in range(self.horizon)]
        state_history = [0 for i in range(self.horizon)]
        cum_regret_history = [0 for i in range(self.horizon)]
        cum_regret = 0
        
        prev_reward = 0
    
        # itereate the simulation for horizon times
        for t in range(self.horizon):
            (u, Xi, _) = self.state_problem
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

            instant_regret =  max(0, self.mu_1_minus_eps - self.true_avg_item[self.item_selected])
            
        
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
    
    
    def draw_user_uniform(self):
        return random.choices(range(self.M), k=1)[0]    
    
    def update_state(self, prev_reward):
        (u, Xi, _) = self.state_problem
        # mark current (user, item) pair
        Xi[self.item_selected, u] = Xi[self.item_selected, u] + 1
        # draw a new user
        u = self.draw_user_uniform()
        # state update
        self.state_problem = (u, Xi, prev_reward)
        



    
    
    

    
        