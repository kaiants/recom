import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

from multiprocessing import Pool
import time
import random

NUM_PROCESSES = 10

from problems import RecommendationsSynthetic
from algorithms import ClusterFirstRecommendNext


def wrapper(args):
    algorithm, statparams, algoparams, seed, index = args
    return simulate_instance(algorithm, statparams, algoparams, int(seed), index)



def simulate_instance(algorithm, statparams, algoparams, seed, index):
    
    print('seed=',end="")
    print(seed)
    random.seed(seed)
    np.random.seed(seed)
      
    recom_problem = RecommendationsSynthetic(algorithm, statparams, algoparams)
    cum_regret_history, play_history = recom_problem.simulate()
    
    return cum_regret_history
    


if __name__ == '__main__':
    
    Maxitr = 200#iteration number 
   
    '''
    Statistical parameters
    '''
    
    horizon = 20000 # time horizon
    N = 3000 # number of items
    M = 5000 # number of users
    # alpha before the normalization sum alpha_k = N
    Alpha = np.array([1500,1500])   
    # beta before the normalization sum alpha_ell = M
    Beta = np.array([5000])
    K = len(Alpha) # of item clusters
    L = len(Beta) # of user clusters
    
    assert horizon <= M * N
    assert np.sum(Alpha) == N
    assert np.sum(Beta) == M

    # item clusters ; assign cluster index in an ascending order
    sigma = np.zeros(N) #takes values 1,...,K
    cnt = 0
    ind = 0
    for alpha in Alpha:
        ind = ind + 1
        for i in range(alpha):
            sigma[i + cnt] = ind
        
        cnt = cnt + alpha
        
    assert min(sigma) == 1
    assert max(sigma) == K
    assert sum(sigma == 1) == Alpha[0]
    assert sum(sigma == 2) == Alpha[1]
    
    # user clusters ; assign cluster index in an ascending order
    tau = np.zeros(M)
    cnt = 0
    ind = 0
    for beta in Beta:
        ind = ind + 1
        for i in range(beta):
            tau[i + cnt] = ind
        
        cnt = cnt + beta
        
    assert min(tau) == 1
    assert max(tau) == L
    assert sum(tau == 1) == Beta[0] 
    #assert sum(tau == 2) == Beta[1]
    

    P_kl = np.array([[0.7], \
                         [0.2]])
    assert P_kl.shape == (K, L)
    
    statparams = (horizon, M, N, Alpha, Beta, K, L, sigma, tau, P_kl)
    
    '''
    Algorithm's input paramters
    '''
    
    s = 0.225 * (np.log(horizon))**2
    T_0 = 1* (np.log(horizon))
    epsilon_alg = 0.2/(np.log(horizon))**(1/4)
    coef_of_hyothesis_testing = 2 * np.log(3) #is a default
    print('epsilon_alg=', end="")
    print(epsilon_alg)
    
    
    assert s < N
    assert T_0 < horizon
    assert 0 < epsilon_alg and epsilon_alg < 1
    
    index = 1
    algoparams = (horizon, s, T_0, epsilon_alg, M, N, K, L, coef_of_hyothesis_testing)
    
    '''
    simulation
    '''
    emp_avg_regret = np.array([0 for i in range(horizon)])
    emp_avg_regret_plus_err = np.array([0 for i in range(horizon)])
    emp_avg_regret_minus_err = np.array([0 for i in range(horizon)])
    
    cum_regret_histories = np.zeros((Maxitr, horizon))
    
    start = time.time()
    
    # pararellized version
    p = Pool(processes=NUM_PROCESSES)
    itrs = [[ClusterFirstRecommendNext, statparams, algoparams, int(itr), index] for itr in np.linspace(1, Maxitr, Maxitr)]
    cum_regret_histories = p.map(wrapper, itrs)
    p.close()

    for itr in np.linspace(1, Maxitr, Maxitr):
        itr = int(itr)
        emp_avg_regret = ((itr - 1)*emp_avg_regret + 1*cum_regret_histories[itr-1]) / (itr)
    
    cum_regret_histories = np.array(cum_regret_histories)
    
#     # non parallerized version
#     for itr in np.linspace(1, Maxitr, Maxitr):
#         itr = int(itr)
#         cum_regret_history = simulate_instance(ClusterFirstRecommendNext, statparams, algoparams, int(itr), index)
#         cum_regret_history = np.array(cum_regret_history)
#         emp_avg_regret = ((itr - 1)*emp_avg_regret + 1*cum_regret_histories[itr-1]) / (itr)
#         cum_regret_histories[int(itr-1)] = cum_regret_history
        

        
    process_time = time.time() - start
    print('time:', end="")
    print(process_time)
    
    
    times = np.linspace(1, horizon, horizon)
    delta = P_kl[0,0] -  P_kl[1,0]
    cum_regret_avg =  Alpha[1] / sum(Alpha) * delta * times
    
    #T/m
    cum_regret_avg = horizon / M * times
    
    #log T
    cum_regret_avg = np.log( times)
    
    #compute variances
    stds = np.zeros(horizon)
    for t in range(horizon):
        stds[t] = np.std(cum_regret_histories[:, t], ddof=1)
    
    #compute average regret history +- err
    emp_avg_regret_plus_err = emp_avg_regret + stds
    emp_avg_regret_minus_err = emp_avg_regret - stds
    
    #plot save
    pdf = PdfPages('cumlative_regret.pdf')
    plt.figure()
    #plt.plot(times, cum_regret_history, label='algorithm regret (one instance)', )
    
    plt.plot(times, emp_avg_regret, label='algorithm regret (instances average)', color='navy')
    plt.fill_between(times, emp_avg_regret_plus_err, emp_avg_regret_minus_err, alpha=0.3, color='navy')
    #plt.plot(times, cum_regret_avg , label='T/m', color = 'firebrick')
    plt.legend()
    plt.xlabel('t')
    plt.ylabel('Regret')
    #plt.show()
    pdf.savefig()
    pdf.close()
    
    
    