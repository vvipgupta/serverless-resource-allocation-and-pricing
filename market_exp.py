"""
This script implements the market simulation
as described in the paper
"""

from functions_sys_user_cloud import *
from numpy import array
import numpy as np
import matplotlib.pyplot as plt
import os
from os import path

if not os.path.exists('figs'):
    os.mkdir('figs')

# Generating random utilities
N = 100 
T = 5
C,J,M,H,P = random_util_generator(N,T)

# Finding the optimal system solution
x_sys, q_sys, m_sys = solving_system(H, J, M)
q_sys_orig = q_sys

# Intializing arrays
C_new = C
opt_util = []
opt_price = []
opt_q = []
opt_x = []

grad_util = []
grad_price = []
grad_q = []
grad_x = []

rand_util = []
rand_x = []
grad_pay_by_users = []
opt_pay_by_users = []
rand_pay_by_users = []

# Defining the probability vector corresponding to which the market 
# utilities will fluctuate
p = 30*[[0.45, 0.55]]
q = 30*[[0.55, 0.45]]
p = p+q

# Running simulations for 60 days
for i in range(60):
    H_new = C_new[...,::-1].cumsum(axis=-1)[...,::-1]  

    # Scheme 1: Finding the optimal resource allocation and prices
    x_sys, q_sys, m_sys = solving_system(H_new, J, M)
    opt_util.append(np.multiply(x_sys,np.divide(H_new,J[:,np.newaxis])))
    opt_price.append(np.sum(m_sys))
    opt_q.append(q_sys)
    opt_pay_by_users.append(np.sum(np.multiply(x_sys, q_sys.T),1))
    
    # Scheme 2: Finding resource allocation and pricing through user feedback and tracking
    q, q_all, m, m_all, util, x_proj = price_tracking(H_new, C_new, J, M, m_sys, 
    	q_sys, grads_steps = 40, q_init=q_sys_orig.T, iter_max=1)
    grad_util.append(util)
    grad_price.append(np.sum(m))
    grad_q.append(q)
    grad_pay_by_users.append(np.sum(np.multiply(x_proj, q),1))
    
    # Scheme 3: Finding resource allocation for a first-come-first-serve/random allocation scheme
    # with constant pricing determined on day 1
    u_r, x_r = random_allocation(M, J, C, H)
    rand_util.append(u_r)
    rand_x.append(x_r)
    rand_pay_by_users.append(np.sum(np.multiply(x_r, q_sys_orig.T),1))
    
    # Updating the utilities for the next day
    randsigns = np.random.choice([-0.5,0.5], size = (N,T), p=p[i])
    C_new = np.maximum(C_new + np.multiply(np.random.rand(N,T), randsigns), 0)

# Collecting data and plotting results
opt_util_sum = [np.sum(x) for x in opt_util]
grad_util_sum = [np.sum(x) for x in grad_util]
rand_util_sum = [np.sum(x) for x in rand_util]

plt.figure(figsize=(5,3))
plt.plot(grad_util_sum, '-', opt_util_sum, '-', rand_util_sum, 'y:')
plt.legend(['Tracking-based', 'Optimal', 'First-come-first-serve'])
plt.xlabel('Day number')
plt.ylabel('Sum utilities')
plt.grid(True)
plt.savefig('figs/grad_vs_opt_vs_rand_util_wide2.pdf',format='pdf', bbox_inches = "tight")

opt_util_user0 = [np.sum(x,1)[0] for x in opt_util]
grad_util_user0 = [np.sum(np.sum(x,0),1)[0] for x in grad_util]
rand_util_user0 = [np.sum(x,1)[0] for x in rand_util]

opt_mean_utils = np.mean(np.sum(array(opt_util),axis = 2),axis=1)
opt_std_utils = np.std(np.sum(array(opt_util),axis = 2),axis=1)

grad_mean_utils = np.mean(np.sum(np.squeeze(array(grad_util)),axis = 2),axis=1)
grad_std_utils = np.std(np.sum(np.squeeze(array(grad_util)),axis = 2),axis=1)

rand_mean_utils = np.mean(np.sum(array(rand_util),axis = 2),axis=1)
rand_std_utils = np.std(np.sum(array(rand_util),axis = 2),axis=1)

x = range(60)
plt.figure(figsize=(4,3))
plt.plot(grad_util_user0, '-', opt_util_user0, '-', rand_util_user0, 'y:')
plt.legend(['Tracking-based', 'Optimal', 'First-come-first-serve'])
plt.xlabel('Day number')
plt.ylabel('Utility of a random user')
plt.grid(True)
plt.savefig('figs/grad_vs_opt_vs_rand_utility_user0.pdf',format='pdf',bbox_inches = "tight")

rand_pay_user0 = [x[0] for x in rand_pay_by_users]
grad_pay_user0 = [x[0] for x in grad_pay_by_users]
opt_pay_user0 = [x[0] for x in opt_pay_by_users]

plt.figure(figsize=(4,3))
plt.plot(grad_pay_user0, '-', opt_pay_user0, '-', rand_pay_user0, 'y:')
plt.legend(['Tracking-based', 'Optimal', 'First-come-first-serve'])
plt.xlabel('Day number')
plt.ylabel('Price paid by a random user')
plt.grid(True)
plt.savefig('figs/grad_vs_opt_vs_rand_price_paid_user0.pdf',format='pdf', bbox_inches = "tight")

opt_util_hour0 = [np.sum(x,0)[0] for x in opt_util]
grad_util_hour0 = [np.sum(np.sum(x,0),0)[0] for x in grad_util]
rand_util_hour0 = [np.sum(x,0)[0] for x in rand_util]

plt.figure(figsize=(4,3))
plt.plot(grad_util_hour0, '-', opt_util_hour0, '-', rand_util_hour0, 'y:')
plt.legend(['Tracking-based', 'Optimal', 'First-come-first-serve'])
plt.xlabel('Day number')
plt.ylabel('Sum utility in the first tier')
plt.grid(True)
plt.savefig('figs/grad_vs_opt_vs_rand_utility_hour0.pdf',format='pdf', bbox_inches = "tight")

grad_q_0 = [x[0][0] for x in grad_q]
opt_q_0 = [x[0] for x in opt_q]

plt.figure(figsize=(4,3))
plt.plot(grad_q_0, '-', opt_q_0, '-', list(q_sys_orig[0])*60,'y:')
plt.legend(['Tracking-based', 'Optimal', 'First-come-first-serve'])
plt.xlabel('Day number')
plt.ylabel('Price charged in the first tier')
plt.grid(True)
plt.savefig('figs/grad_vs_opt_vs_rand_price_hour0_dotted.pdf',format='pdf', bbox_inches = "tight")