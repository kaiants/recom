import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

from multiprocessing import Pool
import time
import random
import datetime

NUM_PROCESSES = 30

from problems import RecommendationsSyntheticContinuous
from algorithms import BKLUCB


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
    
    
    Maxitr = 100#iteration number 
   
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
    
    s =  min(int(horizon / M * np.log(horizon)), N)
    T_0 = int(np.log(horizon)**3)
    
    epsilon_alg = 0.02/(np.log(horizon))**(1/4)
    coef_of_hyothesis_testing = 2 * np.log(3) #is a default
    
    print('s=', end="")
    print(s)
    print('T_0=', end="")
    print(T_0)
    
    assert s<= N
    assert T_0 < horizon
    
    algoparams = (horizon, epsilon_satis_regret, M, N, s, T_0)
    
    index = 1
    
    '''
    simulation
    '''
    emp_avg_regret = np.array([0 for i in range(horizon)])
    emp_avg_regret_plus_err = np.array([0 for i in range(horizon)])
    emp_avg_regret_minus_err = np.array([0 for i in range(horizon)])
    
    cum_regret_histories = np.zeros((Maxitr, horizon))
    
    start = time.time()
    
    p = Pool(processes=NUM_PROCESSES)
    itrs = [[BKLUCB, statparams, algoparams, int(itr), index] for itr in np.linspace(1, Maxitr, Maxitr)]
    cum_regret_histories = p.map(wrapper, itrs)
    p.close()
    
    cum_regret_histories = np.array(cum_regret_histories)
    
        
    # parallelized version
    for itr in np.linspace(1, Maxitr, Maxitr):
        itr = int(itr)
        emp_avg_regret = ((itr - 1)*emp_avg_regret + 1*cum_regret_histories[itr-1]) / (itr)
        
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
    
    plt.plot(times, emp_avg_regret, label='algorithm regret (instances average)', color='navy')
    plt.fill_between(times, emp_avg_regret_plus_err, emp_avg_regret_minus_err, alpha=0.3, color='navy')

    plt.legend()
    plt.xlabel('t')
    plt.ylabel('Regret')
    pdf.savefig()
    pdf.close()
    
    dt = datetime.datetime.now()
    datas = [emp_avg_regret, emp_avg_regret_plus_err, emp_avg_regret_minus_err]
    np.savetxt("emp_avg_regret__modelB_KLUCB"+ dt.strftime("%m_%d_%H_%M")+".csv", datas, delimiter=",")
    