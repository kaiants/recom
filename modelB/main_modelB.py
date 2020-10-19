import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from multiprocessing import Pool
import random

import multiprocessing
import time


from problems import RecommendationsSyntheticContinuous
from algorithms import KLStopping

NUM_PROCESSES = 10
PARALLELIZE_ON = True

def wrapper(args):
    algorithm, statparams, algoparams, seed, index = args
    return simulate_instance(algorithm, statparams, algoparams, int(seed), index)


def simulate_instance(algorithm, statparams, algoparams, seed, index):
    print('seed=',end="")
    print(seed)
    random.seed(seed)
    np.random.seed(seed)
      
    recom_problem = RecommendationsSyntheticContinuous(algorithm, statparams, algoparams)
    cum_regret_history, play_history = recom_problem.simulate()
    
    return cum_regret_history
    

if __name__ == '__main__':
    
    Maxitr = 100 #iteration number 
   
    '''
    Statistical parameters
    '''
    
    horizon = 100000 # time horizon
    
    #N = 30000 # items
    #M = 60000 # users
    N = 700
    M = 1300
    
    epsilon_satis_regret = 0.3 #satisficing regret
    
    
    # hidden paramters
    mu_0 = 0.1
    mu_1 = 0.9
    assert mu_0>=0 and mu_0<= 1
    assert mu_1>=0 and mu_1<= 1
    assert mu_1> mu_0
    assert mu_1 - mu_0 > epsilon_satis_regret
    
    statparams = (horizon, N, M, mu_0, mu_1, epsilon_satis_regret)
    
    
    '''
    Algorithm's input paramters
    '''
   
    index = 1
    
    s = int(8* 0.001 * int(np.floor( (8**2 / epsilon_satis_regret**2 * np.log(horizon))))) # initial item selection numbers
    
    
    
    T_0 = int(2* 0.001 * np.floor(32**2 / epsilon_satis_regret**2 * (np.log(horizon))**2) )#  exploration durations
#     s  = int( * np.log(horizon)**2)
#     T_0 = int(1 * np.log(horizon)**3)
    
    print('s=', end="")
    print(s)
    print('T_0 =', end="")
    print(T_0)
    assert s < N
    assert T_0 < horizon
    
    algoparams = (horizon, epsilon_satis_regret, M, N, s, T_0)
    
    
    '''
    simulation
    '''
    emp_avg_regret = np.array([0 for i in range(horizon)])
    emp_avg_regret_plus_err = np.array([0 for i in range(horizon)])
    emp_avg_regret_minus_err = np.array([0 for i in range(horizon)])
    
    cum_regret_histories = np.zeros((Maxitr, horizon))
    
    start = time.time()

    if PARALLELIZE_ON:
        # parallelized version    
        p = Pool(processes=NUM_PROCESSES)
        itrs = [[KLStopping, statparams, algoparams, int(itr), index] for itr in np.linspace(1, Maxitr, Maxitr)]
        cum_regret_histories = p.map(wrapper, itrs)
        p.close()
        for itr in np.linspace(1, Maxitr, Maxitr):
            itr = int(itr)
            emp_avg_regret = ((itr - 1)*emp_avg_regret + 1*cum_regret_histories[itr-1]) / (itr)
        cum_regret_histories = np.array(cum_regret_histories)
    else:
        #non parallelized
        for itr in np.linspace(1, Maxitr, Maxitr):
            cum_regret_history = simulate_instance(KLStopping, statparams, algoparams, int(itr), index)
            cum_regret_history = np.array(cum_regret_history)
            emp_avg_regret = ((itr - 1)*emp_avg_regret + 1*cum_regret_history) / (itr)
            cum_regret_histories[int(itr-1)] = cum_regret_history
            print(itr)
        

        
    process_time = time.time() - start
    print('time:', end="")
    print(process_time)
    
    times = np.linspace(1, horizon, horizon)

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
    
    
    