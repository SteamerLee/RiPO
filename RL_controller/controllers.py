#！/usr/bin/python
# -*- coding: utf-8 -*-#

'''
---------------------------------
 Name:         controllers.py
 Author:       RiPO
---------------------------------
'''

import numpy as np
from cvxopt import matrix, solvers
solvers.options['show_progress'] = False

def RL_withoutController(a_rl, env=None):
    a_cbf = np.array([0]*env.stock_num)
    env.action_cbf_memeory.append(a_cbf)
    return a_cbf

def RL_withController(a_rl, env=None):
    if env.config.pricePredModel == 'MA':
        pred_prices_change = get_pred_price_change(env=env)
        pred_cov = None
    else:
        raise ValueError("Cannot find the price prediction model [{}]..".format(env.config.pricePredModel))

    a_cbf = cbf_opt(env=env, a_rl=a_rl, pred_prices_change=pred_prices_change, pred_cov=pred_cov)    
    a_cbf = a_cbf * env.sw_weight_lst[-1]
    env.action_cbf_memeory.append(a_cbf)
    return a_cbf

def get_pred_price_change(env):
    ma_lst = env.ctl_state['MA-{}'.format(env.config.otherRef_indicator_ma_window)]
    pred_prices = ma_lst
    cur_close_price = np.array(env.curData['close'].values)
    pred_prices_change = (pred_prices - cur_close_price) / cur_close_price 
    return pred_prices_change

def cbf_opt(env, a_rl, pred_prices_change, pred_cov=None):
    # Objective function: Linear objecive function, Constraints: Linear inequality and Second-order Cone programs (SOCP)
    # Minimize the loss of expected returns due to weighting adjustments.
    N = env.stock_num

    # Past N days daily return rate of each stock, (num_of_stocks, lookback_days), [[t-N+1, t-N+2, .., t-1, t]]
    daily_return_ay = env.ctl_state['DAILYRETURNS-{}'.format(env.config.dailyRetun_lookback)]
    
    # t
    cov_r_t0 = np.cov(daily_return_ay)
    w_t0 = np.array([env.actions_memory[-1]])
    risk_stg_t0 = np.sqrt(np.matmul(np.matmul(w_t0, cov_r_t0), w_t0.T)[0][0])
    risk_market_t0 = env.config.risk_market
    if len(env.risk_adj_lst) <= 1:
        risk_safe_t0 = env.risk_adj_lst[-1]
    else:
        risk_safe_t0 = env.risk_adj_lst[-2]
        if risk_safe_t0 == 0:
            risk_safe_t0 = risk_stg_t0 + risk_market_t0
            env.risk_adj_lst[-2] = risk_safe_t0

    # t+1
    gamma = env.config.cbf_gamma
    risk_market_t1 = env.config.risk_market
    risk_safe_t1 = env.risk_adj_lst[-1]

    pred_prices_change_reshape = np.reshape(pred_prices_change, (-1, 1))
    r_t1 = np.append(daily_return_ay[:, 1:], pred_prices_change_reshape, axis=1)
    cov_r_t1 = np.cov(r_t1)

    eig_value, eig_vector = np.linalg.eig(cov_r_t1)
    eig_diag = np.diag(np.power(eig_value, 0.5))
    # cov_sqrt_t1 = sqrtm(cov_r_t1), from scipy.linalg import sqrtm
    cov_sqrt_t1 = np.matmul(np.matmul(eig_vector, eig_diag), np.linalg.inv(eig_vector))
    
    G_ay = np.array([]).reshape(-1, N)

    h_0 = np.array([])
    h_0 = np.append(h_0, [0], axis=0) # linear_h1
    h_0 = np.append(h_0, [0], axis=0) # linear_h2

    h_0 = np.append(h_0, np.array(a_rl), axis=0) # linear_h3
    h_0 = np.append(h_0, 1-np.array(a_rl), axis=0) # linear_h4

    socp_b = np.matmul(cov_sqrt_t1, a_rl)

    socp_d = -risk_market_t1 + risk_safe_t1 + (gamma - 1) * (-risk_market_t0 - risk_stg_t0 + risk_safe_t0)
    h = np.append(h_0, [socp_d], axis=0) # socp_d
    h = np.append(h, socp_b, axis=0) # socp_b
    h = matrix(h)

    linear_g1 = np.array([[1] * N])
    G_ay = np.append(G_ay, linear_g1, axis=0)
    linear_g2 = np.array([[-1] * N])
    G_ay = np.append(G_ay, linear_g2, axis=0)

    linear_g3 = np.diag([-1] * N)
    G_ay = np.append(G_ay, linear_g3, axis=0)
    linear_g4 = np.diag([1] * N) 
    G_ay = np.append(G_ay, linear_g4, axis=0)

    socp_cx = np.array([[0] * N])
    G_ay = np.append(G_ay, socp_cx, axis=0)

    linear_eq_num = 2*N+2
    dims = {'l': linear_eq_num, 'q': [N+1], 's': []}
    
    solver_flag = True
    step_add_lst = [0.002, 0.002, 0.002, 0.002, 0.002, 0.005, 0.005, 0.005, 0.005, 0.005]
    cnt = 1
    if env.config.is_dynamic_risk_bound:
        cnt_th = 10 # Iterative acceptable risk relaxation in ARS when the risk controller fails to solve the constraints.
    else:
        cnt_th = 1 # original risk constraint strategy.
    if env.config.controller_obj == 1:
        # LP: min. losses of profits
        c = matrix(-pred_prices_change)
        G_ay = np.append(G_ay, -cov_sqrt_t1, axis=0) 
        G = matrix(G_ay) # G = matrix(np.transpose(np.transpose(G_ay)))
        while cnt <= cnt_th:
            try:
                sol = solvers.conelp(c, G, h, dims)
                if sol['status'] == 'optimal':
                    solver_flag = True
                    break
                else:
                    raise
            except:
                cnt += 1
                solver_flag = False
                risk_safe_t1 = risk_safe_t1 + step_add_lst[cnt-2]
                socp_d = -risk_market_t1 + risk_safe_t1 + (gamma - 1) * (-risk_market_t0 - risk_stg_t0 + risk_safe_t0)
                h = np.append(h_0, [socp_d], axis=0) # socp_d
                h = np.append(h, socp_b, axis=0) # socp_b
                h = matrix(h)

    else:
        raise ValueError('Invalid controller_obj value: {}'.format(env.config.controller_obj))

    if solver_flag:
        if sol['status'] == 'optimal':
            a_cbf = np.reshape(np.array(sol['x']), -1)
            env.solver_stat['solvable'] = env.solver_stat['solvable'] + 1
            env.risk_adj_lst[-1] = risk_safe_t1
        else:
            a_cbf = np.zeros(N)
            env.solver_stat['insolvable'] = env.solver_stat['insolvable'] + 1
            env.risk_adj_lst[-1] = 0
    else:
        a_cbf = np.zeros(N)
        env.solver_stat['insolvable'] = env.solver_stat['insolvable'] + 1
        env.risk_adj_lst[-1] = 0
    return a_cbf