import sys
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

from multiprocessing import Pool
import time
import random
import datetime


NUM_PROCESSES = 10

from problems import RecommendationsSynthetic
from algorithms import ClusterUCB1


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
    
    horizon = int(sys.argv[1])
    Maxitr = 10#iteration number 
   
    '''
    Statistical parameters
    '''
    
    #     horizon = 800000 # time horizon
    N = 2000 # number of items
    M = 5000 # number of users
    # alpha before the normalization sum alpha_k = N
    Alpha = np.array([int(N/2),int(N/2)])   
    # beta before the normalization sum alpha_ell = M
    Beta = np.array([int(M/2), int(M/2)])
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
    assert sum(tau == 2) == Beta[1]
    

    P_kl = np.array([[0.2, 0.8], [0.8, 0.2]])
    assert P_kl.shape == (K, L)
    
    statparams = (horizon, M, N, Alpha, Beta, K, L, sigma, tau, P_kl)
    
    '''
    Algorithm's input paramters
    '''
    # number of random item selections
    s =  4 * int(np.floor(min(N,  M / (np.log(horizon))**2 )))
    print('s=',end="")
    print(s)
    s_user = int(M / np.log(horizon))
    print('s_user=',end="")
    print(s_user) 
    
    
    
    print('T/m=', end="")
    print(horizon/M)
    
    # duration of exploration phase
    T_0 = 0 #T_0 = 10 * M
    print('T_0=',end="")
    print(T_0)
    
    
    epsilon_alg = 1/100
    coef_of_hyothesis_testing = 2 * np.log(3) #is a default
    
    
    assert s < N
    assert T_0 < horizon
    assert 0 < epsilon_alg and epsilon_alg < 1
    
    index = 1
    algoparams = (horizon, s, s_user, T_0, epsilon_alg, M, N, K, L, coef_of_hyothesis_testing)
    
    '''
    simulation
    '''
    emp_avg_regret = np.array([0 for i in range(horizon)])
    emp_avg_regret_plus_err = np.array([0 for i in range(horizon)])
    emp_avg_regret_minus_err = np.array([0 for i in range(horizon)])
    
    cum_regret_histories = np.zeros((Maxitr, horizon))
    
    start = time.time()
    
        # parallelized version
    pools = Pool(processes=NUM_PROCESSES)
    itrs = [[ClusterUCB1, statparams, algoparams, int(itr), index] for itr in np.linspace(1, Maxitr, Maxitr)]
    cum_regret_histories = pools.map(wrapper, itrs)
    pools.close()
    
    cum_regret_histories = np.array(cum_regret_histories)
    for itr in np.linspace(1, Maxitr, Maxitr):
        itr = int(itr)
        emp_avg_regret = ((itr - 1)*emp_avg_regret + 1*cum_regret_histories[itr-1]) / (itr)
    
#     # non parallelized version
#     for itr in np.linspace(1, Maxitr, Maxitr):
#         itr = int(itr)
#         cum_regret_history = simulate_instance(ClusterUCB1, statparams, algoparams, 1, index)
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
        
#     dt = datetime.datetime.now()
#     np.savetxt("emp_avg_regret_with"+ dt.strftime("%m_%d_%H_%M")+".csv", emp_avg_regret, delimiter=",")
#     np.savetxt("stds"+ dt.strftime("%m_%d_%H_%M")+".csv", stds, delimiter=",")
    #compute average regret history +- err
    emp_avg_regret_plus_err = emp_avg_regret + stds
    emp_avg_regret_minus_err = emp_avg_regret - stds
    
    np.savetxt("emp_avg_regret_UCB_with_T_"+str(horizon)+".csv", emp_avg_regret, delimiter=",")
    np.savetxt("stds_T_"+str(horizon)+".csv", stds, delimiter=",")
    
#     #plot save
#     pdf = PdfPages('cumlative_regret'+ dt.strftime("%m_%d_%H_%M")+'.pdf')
#     plt.figure()
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
    
    
    