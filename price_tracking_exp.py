"""
This script validates the price tracking algorithm
(Algorithm 1 in the paper) by implementing it on
synthetically generated utilities.
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

# Finding the optimal utilities
x_sys, q_sys, m_sys = solving_system(H, J, M)

obtained_utilities = np.multiply(x_sys, P)
opt_utility = np.sum(obtained_utilities)

# Running price tracking
q, q_all, m, m_all, utils, x_proj = price_tracking(H, C, J, M, m_sys, q_sys, tol=5*1e-4)
util_sum = [np.sum(x) for x in utils]

f = plt.figure(figsize=(4,3))
plt.plot(util_sum, '.-', opt_utility*np.ones(len(utils)),'.-')
plt.legend(['Tracking-based Utility','Optimal Utility'])
plt.xlabel('User Budget Updates')
plt.ylabel('System Sum Utility')
plt.grid(True)
plt.savefig('figs/util_vs_user_update.pdf',format='pdf', bbox_inches = "tight")

sum_q = [np.sum(x) for x in q_all]
q_0 = [M*x[0] for x in q_all]
m_0 = [x[0] for x in m_all]

f = plt.figure(figsize=(4,3))
plt.plot(m_0,'.-',q_0[1:],'.-')
plt.legend(['Users\' budgets in the first tier','Price charged in the first tier'])
plt.xlabel('User Budget Updates')
plt.ylabel('USD')
plt.grid(True)
plt.savefig('figs/price_vs_user_budget_first_tier.pdf',format='pdf', bbox_inches = "tight")
