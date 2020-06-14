from cvxopt import matrix, solvers, log, exp
from numpy import array
import numpy as np

"""
Generating random utility matrix H, 
Jobs J and other variables such as 
the number of machines M
"""
def random_util_generator(N,T):
    C = np.random.uniform(5,10,size=(N,T))
    # C.sort()
    J = np.random.randint(10,99,size=(N))
    M = int(50*N/T)
    H = C[...,::-1].cumsum(axis=-1)[...,::-1]  
    P = np.divide(H,J[:,np.newaxis])
    return C,J,M,H,P

"""
Solving the System problem using CVX.
Adding a quadratic regularizer with
parameter reg to ensure uniqueness of solution.
"""
def solving_system(H, J, M, reg=1e-3):
    N, T = H.shape
    A1 = np.zeros((N,N*T))
    for i in range(N):
        A1[i, i*T:(i+1)*T] = np.ones((T))

    A2 = np.zeros((T,N*T))
    for i in range(T):
        A2[i, i:N*T:T] = np.ones((N))
    A = np.concatenate((A1, A2, -1*np.identity(N*T)))

    obj_sys = np.divide(H,J[:,np.newaxis])
    c = -1*obj_sys.reshape((N*T,1))

    b = np.concatenate((J, M*np.ones(T), np.zeros(N*T)))
    P = reg*np.identity(N*T)
    solvers.options['show_progress'] = False
    sol=solvers.qp(matrix(P, tc='d'), matrix(c, tc='d'), matrix(A, tc='d'), matrix(b, tc='d'))
    x_sys = np.array(sol['x']).reshape((N,T))
    q_sys = array(sol['z'])
    q_sys = q_sys[N:N+T]
    m_sys = np.multiply(x_sys,q_sys.T)
    return x_sys, q_sys, m_sys

"""
Solving the User problem using CVX
"""
def solving_for_user_m(q, H, J, reg=1e-3):
    N, T = H.shape
    q = q.reshape(1,T)
    qJ = np.dot(J[:,np.newaxis],q)
    obj_user = np.ones((N,T)) - np.divide(H, qJ)
    m = np.zeros((N,T))

    A = (1/q)
    A = np.concatenate((A, -1*np.identity(T)))
    
    P = np.divide(reg*np.identity(T),(q**2))

    for i in range(N):
        b = np.concatenate(([J[i]], np.zeros(T)))
        solvers.options['show_progress'] = False
        solvers.options['maxiters'] = 10000 
        sol=solvers.qp(matrix(P, tc='d'), matrix(obj_user[i,:]), 
                       matrix(A, tc='d'), matrix(b, tc='d'), warm_start=True)
        if sol['status'] != 'optimal':
            print("Optimal not found for user {}".format(i+1))
        m[i,:] = np.array(sol['x']).reshape((1,T))
    print("Budgets m summed across users",np.sum(m,0))
    return m

"""
Solving the Cloud problem using CVX
"""
def solving_for_cloud_x(m, M):
    N, T = m.shape
    K = [1]
    for i in range(T):
        K.append(N)

    g = [1]
    for i in range(N*T):
        g.append(1/M)

    F = np.zeros((N*T+1, N*T))
    F[0,:] = -1*m.reshape(1,N*T)
    for i in range(T):
        for j in range(N):
            F[N*i + j + 1, j*T + i] = 1
    
    solvers.options['show_progress'] = False
    sol_cloud = solvers.gp(K, matrix(F), log(matrix(g, tc='d')))
    x_cloud = array(exp(sol_cloud['x'] )).reshape((N,T))
    return x_cloud

"""
Dual version of the Lyapunov function
"""
def solving_for_q_dual(m, M, grads_steps, q_init=None, inc=1000, ep=1e-4, ss=1e-5):
    N, T = m.shape
    if q_init is None:
        q = np.ones((1,T))
    else:
        q = q_init
    m_sum = np.sum(m,0)
    for j in range(1,grads_steps):
        q_prev = q
        q = np.maximum(q + ss*np.minimum((np.divide(m_sum,q) - M),inc),ep)
        if np.linalg.norm(q_prev - q) < (1e-10)*np.linalg.norm(q):
            break
    print("Converged after {} iters with q: {}".format(j,q))
    return q

"""
Price tracking with users' budgets
"""
def price_tracking(H, C, J, M, m_sys, q_sys, grads_steps = 40, q_init = array([0.5, 0.4, 0.3, 0.2, 0.1]), iter_max = 100, tol=1e-3, ss=1e-6):
    N, T = H.shape
    q = q_init
    q_prev = np.ones((1,T))
    count = 0
    utils = []
    q_all = [q]
    m_all = []
    while np.linalg.norm(q_prev - q) > tol*np.linalg.norm(q_prev) and count < iter_max:
        q_prev = q
        count = count + 1

        m = solving_for_user_m(q, H, J)
        m_norm_err = 100*np.linalg.norm(m - m_sys)/np.linalg.norm(m_sys)
        print("Iter: ", count)
        print("Norm error percentange users m {:.2f}".format(m_norm_err))

        q = solving_for_q_dual(m, M, grads_steps = grads_steps, q_init = q_prev, ss=ss)
        q_norm_err = 100*np.linalg.norm(q - q_sys.T)/np.linalg.norm(q_sys)
        print("Norm error percentange cloud q {:.2f}".format(q_norm_err)) 
        
        print("Convergence ratio", np.linalg.norm(q_prev - q)/np.linalg.norm(q_prev))
        x = np.divide(m,q)
        x_proj = x - np.dot(np.ones((N,1)), (np.maximum(np.sum(x,0) - M, 0)/N).reshape(1,T))
        utils.append(np.multiply(x_proj,np.divide(H,J[:,np.newaxis])))
        q_all.append(q)
        m_all.append(np.sum(m,0))
    return q, q_all, m, m_all, utils, x_proj

"""
Random or first-come-first-serve allocation
"""
def random_allocation(M, J, C, H):
    N, T = H.shape
    arr = np.arange(N)
    np.random.shuffle(arr)
    x_rand = np.zeros((N,T))
    x_done = []
    for t in range(T):
        M_rem = M
        for i in arr:
            if i in x_done:
                continue
            if M_rem >= J[i] - np.sum(x_rand[i,:]):
                x_rand[i,t] = J[i] - np.sum(x_rand[i,:])
            else:
                x_rand[i,t] = M_rem
                break
            M_rem = M_rem - x_rand[i,t]
            x_done.append(i)
    util_rand = np.multiply(x_rand, np.divide(H, J[:,np.newaxis]))
    return util_rand, x_rand