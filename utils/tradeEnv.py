#！/usr/bin/python
# -*- coding: utf-8 -*-#
'''
---------------------------------
 Name: tradeEnv.py  
 Author: RiPO
--------------------------------
'''
import numpy as np
import os
import pandas as pd
import time
import copy
from gym.utils import seeding
import gym
from gym import spaces
from stable_baselines3.common.vec_env import DummyVecEnv

class StockPortfolioEnv(gym.Env):

    def __init__(self, config, rawdata, mode, stock_num, action_dim, tech_indicator_lst, max_shares,
                 initial_asset=1000000, reward_scaling=1, norm_method='softmax', transaction_cost=0.001, slippage=0.001, seed_num=2022):
        
        self.config = config
        self.rawdata = rawdata
        self.mode = mode
        self.stock_num = stock_num # Number of stocks
        self.action_dim = action_dim # Number of stocks
        self.tech_indicator_lst = tech_indicator_lst
        self.tech_indicator_lst_wocov = copy.deepcopy(self.tech_indicator_lst) # without cov
        self.tech_indicator_lst_wocov.remove('cov')
        self.max_shares = max_shares # Maximum shares per trading
        self.seed_num = seed_num 
        self.seed(seed=self.seed_num)
        self.epoch = 0
        self.curTradeDay = 0

        self.initial_asset = initial_asset # Start asset
        self.reward_scaling = reward_scaling 
        self.norm_method = norm_method
        self.transaction_cost = transaction_cost # w/o percentage, 0.001
        self.slippage = slippage # 0.001 for one-side, 0.002 for two-side
        if self.norm_method == 'softmax':
            self.weights_normalization = self.softmax_normalization
        elif self.norm_method == 'sum':
            self.weights_normalization = self.sum_normalization
        else:
            raise ValueError("Unexpected normalization method of stock weights: {}".format(self.norm_method))

        self.state_dim = ((len(self.tech_indicator_lst_wocov)+self.stock_num) * self.stock_num) + 1 # 1: cash
        self.action_space = spaces.Box(low=0, high=1, shape=(self.action_dim, ))
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(self.state_dim, ))

        self.rawdata.sort_values(['date', 'stock'], ascending=True, inplace=True)
        self.rawdata.index = self.rawdata.date.factorize()[0]
        self.totalTradeDay = len(self.rawdata['date'].unique())
        self.stock_lst = np.sort(self.rawdata['stock'].unique())

        self.curData = copy.deepcopy(self.rawdata.loc[self.curTradeDay, :])
        self.curData.sort_values(['stock'], ascending=True, inplace=True)
        self.curData.reset_index(drop=True, inplace=True)
        self.covs = np.array(self.curData['cov'].values[0])
        self.state = np.append(self.covs, np.transpose(self.curData[self.tech_indicator_lst_wocov].values), axis=0)
        self.state = self.state.flatten()
        self.state = np.append(self.state, [0], axis=0)
        self.ctl_state = {k:np.array(list(self.curData[k].values)) for k in self.config.otherRef_indicator_lst}
        self.terminal = False
        
        self.cur_capital = self.initial_asset

        self.mdd = 0
        self.mdd_high = self.initial_asset
        self.mdd_low = self.initial_asset
        self.mdd_curHigh = self.initial_asset
        self.mdd_curLow = self.initial_asset
        self.mdd_highTimepoint = 0
        self.mdd_lowTimepoint = 0
        self.mdd_curHighTimepoint = 0
        self.mdd_curLowTimepoint = 0

        self.asset_lst = [self.initial_asset] 
        self.profit_lst = [0] # percentage of portfolio daily returns
        self.actions_memory = [[1/self.stock_num]*self.stock_num]
        self.date_memory = [self.curData['date'].unique()[0]]
        self.reward_lst = [0]
        self.stg_vol_lst = [0] # Strategy volatility
        self.action_cbf_memeory = [[0] * self.stock_num]

        self.risk_adj_lst = [self.config.risk_default]
        self.risk_raw_lst = [0] # For performance analysis. Record the risk without using risk controllrt during the validation/test period.
        self.risk_cbf_lst = [0]
        self.return_raw_lst = [self.initial_asset] 
        self.solver_stat = {'solvable': 0, 'insolvable': 0} 

        self.sw_weight_lst = [1.0]

        risk_free = self.config.mkt_rf[self.config.market_name] / 100
        re_lb = (1 - self.config.mulrf) * risk_free
        re_ub = (1 + self.config.mulrf) * risk_free
        self.linearF_a = (self.config.risk_accepted_max - self.config.risk_accepted) / (re_ub - re_lb)
        self.linearF_b = (((1 + self.config.mulrf) * self.config.risk_accepted) - ((1 - self.config.mulrf) * self.config.risk_accepted_max)) / (2 * self.config.mulrf)

        self.start_cputime = time.process_time()
        self.start_systime = time.perf_counter()

        # For saveing profile
        self.profile_hist_field_lst = [            
            'ep', 'trading_days', 'annualReturn_pct', 'mdd', 'sharpeRatio', 'final_capital', 'volatility', 'netProfit', 'netProfit_pct', 'winRate',
            'vol_max', 'vol_min', 'vol_avg', 
            'risk_max', 'risk_min', 'risk_avg', 'riskRaw_max', 'riskRaw_min', 'riskRaw_avg',
            'dailySR_max', 'dailySR_min', 'dailySR_avg', 'dailySR_wocbf_max', 'dailySR_wocbf_min', 'dailySR_wocbf_avg',
            'dailyReturn_pct_max', 'dailyReturn_pct_min', 'dailyReturn_pct_avg',
            'sigReturn_max', 'sigReturn_min', 'mdd_high', 'mdd_low', 'mdd_high_date', 'mdd_low_date', 'sharpeRatio_wocbf',
            'reward_sum', 'final_capital_wocbf',
            'solver_solvable', 'solver_insolvable', 'cputime', 'systime'
        ]
        self.profile_hist_ep = {k: [] for k in self.profile_hist_field_lst}

    def step(self, actions):
        self.terminal = self.curTradeDay >= (self.totalTradeDay - 1)
        if self.terminal:
            self.end_cputime = time.process_time()
            self.end_systime = time.perf_counter()
            self.model_save_flag = True
            invest_profile = self.get_results()
            self.save_profile(invest_profile=invest_profile)
            return self.state, self.reward, self.terminal, {}
        else:
            actions = np.reshape(actions, (-1)) # [1, num_of_stocks] or [num_of_stocks, ]
            weights = self.weights_normalization(actions=actions) # Unnormalized weights -> normalized weights 
            self.actions_memory.append(weights)
            lastDayData = self.curData            
            # Jump to next day
            self.curTradeDay = self.curTradeDay + 1
            self.curData = copy.deepcopy(self.rawdata.loc[self.curTradeDay, :])
            self.curData.sort_values(['stock'], ascending=True, inplace=True)
            self.curData.reset_index(drop=True, inplace=True)
            self.covs = np.array(self.curData['cov'].values[0])
            self.state = np.append(self.covs, np.transpose(self.curData[self.tech_indicator_lst_wocov].values), axis=0)
            self.state = self.state.flatten()
            self.ctl_state = {k:np.array(list(self.curData[k].values)) for k in self.config.otherRef_indicator_lst} # State data for the controller

            curDay_ClosePrice_withSlippage = np.array(self.curData['close'].values) * (1 + (np.random.random(self.stock_num) * (self.slippage * 2) - self.slippage))
            sigDayReturn = ((curDay_ClosePrice_withSlippage / np.array(lastDayData['close'].values)) - 1) * weights # [s1_pct, s2_pct, .., px_pct_returns]
            poDayReturn = np.sum(sigDayReturn)
            poDayReturn_withcost = poDayReturn - self.transaction_cost
            updatePoValue = self.cur_capital * (1 + poDayReturn_withcost)
            self.cur_capital = updatePoValue
            self.state = np.append(self.state, [np.log((self.cur_capital/self.initial_asset))], axis=0) # Account observation
            
            self.profit_lst.append(poDayReturn_withcost) 
            cur_date = self.curData['date'].unique()[0]
            self.date_memory.append(cur_date)
            self.asset_lst.append(self.cur_capital)

            # Adaptive Risk Strategy (ARS) module 
            if self.config.is_dynamic_risk_bound:
                if len(self.profit_lst) <= self.config.ref_return_lookback:
                    dynamic_risk = self.config.risk_default
                else:
                    ma_r_daily = np.mean(self.profit_lst[-self.config.ref_return_lookback:])
                    ma_r_annual = np.power((1 + ma_r_daily), (self.config.tradeDays_per_year)) - 1
                    
                    dynamic_risk = (self.linearF_a * ma_r_annual) + self.linearF_b
                    dynamic_risk = np.clip(dynamic_risk, self.config.risk_accepted, self.config.risk_accepted_max)

                self.risk_adj_lst.append(dynamic_risk)
            else:
                self.risk_adj_lst.append(self.config.risk_default)

            # For performance analysis. Record the risk without using risk controllrt during the validation/test period.
            daily_return_ay = np.array(list(self.curData['DAILYRETURNS-{}'.format(self.config.dailyRetun_lookback)].values))
            cur_cov = np.cov(daily_return_ay) 
            self.risk_cbf_lst.append(np.sqrt(np.matmul(np.matmul(weights, cur_cov), weights.T)))
            w_rl = weights - self.action_cbf_memeory[-1]
            self.risk_raw_lst.append(np.sqrt(np.matmul(np.matmul(w_rl, cur_cov), w_rl.T)))
            return_raw = self.return_raw_lst[-1] * (1 + (np.sum(((curDay_ClosePrice_withSlippage / np.array(lastDayData['close'].values)) - 1) * w_rl) - self.transaction_cost))
            self.return_raw_lst.append(return_raw)

            if self.cur_capital >= self.mdd_curHigh:
                self.mdd_curHigh = self.cur_capital
                self.mdd_curHighTimepoint = cur_date
                self.mdd_curLow = self.cur_capital
                self.mdd_curLowTimepoint = cur_date
            else:
                if self.cur_capital <= self.mdd_curLow:
                    self.mdd_curLow = self.cur_capital
                    self.mdd_curLowTimepoint = cur_date

                    if ((self.mdd_curHigh - self.mdd_curLow)/self.mdd_curHigh) >= self.mdd:
                        self.mdd = (self.mdd_curHigh - self.mdd_curLow)/self.mdd_curHigh
                        self.mdd_high = self.mdd_curHigh
                        self.mdd_low = self.mdd_curLow
                        self.mdd_highTimepoint = self.mdd_curHighTimepoint
                        self.mdd_lowTimepoint = self.mdd_curLowTimepoint

            self.reward = poDayReturn_withcost * 100 * self.reward_scaling
            self.reward_lst.append(self.reward)
            cur_stg_vol = np.sqrt((np.sum(np.power((self.profit_lst - np.mean(self.profit_lst)), 2)) * self.config.tradeDays_per_year / (len(self.profit_lst) - 1)))
            self.stg_vol_lst.append(cur_stg_vol)
            self.model_save_flag = False

            # Dynamic Contribution Mechanism (DCM) module
            if self.config.is_switch_weighting:
                if len(self.profit_lst) <= self.config.sw_ref_return_lookback:
                    self.sw_weight_lst.append(1.0)
                else:
                    ma_r_daily = np.mean(self.profit_lst[-self.config.sw_ref_return_lookback:])
                    risk_free = self.config.mkt_rf[self.config.market_name] / 100
                    # |Rs - rf|
                    rs_norm = ma_r_daily - risk_free
                    if rs_norm < 0:
                        ma_abs = np.abs(rs_norm)
                        ma_abs = np.min([ma_abs, self.config.sw_loss_lb])
                        if self.config.sw_is_linear:
                            sw_update_val = 1/self.config.sw_loss_lb * ma_abs # Linear
                        else:
                            # Non-linear
                            sw_update_val = np.power((self.config.sw_m + (ma_abs/self.config.sw_loss_lb)), (1-(ma_abs/self.config.sw_loss_lb)))
                        sw_update_val = np.min([sw_update_val, 1.0])
                        self.sw_weight_lst.append(sw_update_val)
                    else:
                        self.sw_weight_lst.append(self.config.sw_m)
            else:
                self.sw_weight_lst.append(1.0)

            return self.state, self.reward, self.terminal, {}

    def reset(self):
        self.epoch = self.epoch + 1
        self.curTradeDay = 0

        self.curData = copy.deepcopy(self.rawdata.loc[self.curTradeDay, :])
        self.curData.sort_values(['stock'], ascending=True, inplace=True)
        self.curData.reset_index(drop=True, inplace=True)
        self.covs = np.array(self.curData['cov'].values[0])
        self.state = np.append(self.covs, np.transpose(self.curData[self.tech_indicator_lst_wocov].values), axis=0)
        self.state = self.state.flatten()
        self.state = np.append(self.state, [0], axis=0)
        self.ctl_state = {k:np.array(list(self.curData[k].values)) for k in self.config.otherRef_indicator_lst} # State data for the controller
        self.terminal = False
        
        self.cur_capital = self.initial_asset

        self.mdd = 0
        self.mdd_high = self.initial_asset
        self.mdd_low = self.initial_asset
        self.mdd_curHigh = self.initial_asset
        self.mdd_curLow = self.initial_asset
        self.mdd_highTimepoint = 0
        self.mdd_lowTimepoint = 0
        self.mdd_curHighTimepoint = 0
        self.mdd_curLowTimepoint = 0

        self.asset_lst = [self.initial_asset] 
        self.profit_lst = [0] 
        self.actions_memory = [[1/self.stock_num]*self.stock_num]
        self.date_memory = [self.curData['date'].unique()[0]]
        self.reward_lst = [0]
        self.stg_vol_lst = [0]
        self.action_cbf_memeory = [[0] * self.stock_num]

        self.risk_adj_lst = [self.config.risk_default]
        self.risk_raw_lst = [0]
        self.risk_cbf_lst = [0]
        self.return_raw_lst = [self.initial_asset]
        self.solver_stat = {'solvable': 0, 'insolvable': 0} 

        self.sw_weight_lst = [1.0]

        self.start_cputime = time.process_time()
        self.start_systime = time.perf_counter()

        return self.state

    def render(self, mode='human'):
        return self.state
    

    def softmax_normalization(self, actions):
        if np.sum(actions) == 0:  
            norm_weights = np.array([1/self.stock_num]*self.stock_num)
        else:
            norm_weights = np.exp(actions)/np.sum(np.exp(actions))
        return norm_weights
    
    def sum_normalization(self, actions):
        if np.sum(actions) == 0:
            norm_weights = np.array([1/self.stock_num]*self.stock_num)
        else:
            norm_weights = actions / np.sum(actions)
        return norm_weights

    def save_action_memory(self):

        action_pd = pd.DataFrame(np.array(self.actions_memory), columns=self.stock_lst)
        action_pd['date'] = self.date_memory
        return action_pd

    def seed(self, seed=2022):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def get_sb_env(self):
        e = DummyVecEnv([lambda: self])
        obs = e.reset()
        return e, obs


    def get_results(self):
        netProfit = self.cur_capital - self.initial_asset # Profits
        netProfit_pct = netProfit / self.initial_asset # Rate of profirs

        diffPeriodAsset = np.diff(self.asset_lst)
        sigReturn_max = np.max(diffPeriodAsset) # Maximal returns in a single transaction.
        sigReturn_min = np.min(diffPeriodAsset) # Minimal returns in a single transaction

        # Annual Returns
        annualReturn_pct = np.power((1 + netProfit_pct), (self.config.tradeDays_per_year/len(self.asset_lst))) - 1
        
        dailyReturn_pct_max = np.max(self.profit_lst)
        dailyReturn_pct_min = np.min(self.profit_lst)
        avg_dailyReturn_pct = np.mean(self.profit_lst)
        # strategy volatility
        volatility = np.sqrt((np.sum(np.power((self.profit_lst - avg_dailyReturn_pct), 2)) * self.config.tradeDays_per_year / (len(self.profit_lst) - 1)))

        # SR_Vol, Long-term risk
        sharpeRatio = ((annualReturn_pct * 100) - self.config.mkt_rf[self.config.market_name])/ (volatility * 100)
        sharpeRatio = np.max([sharpeRatio, 0])

        dailyAnnualReturn_lst = np.power((1+np.array(self.profit_lst)), self.config.tradeDays_per_year) - 1
        dailyRisk_lst = np.array(self.risk_cbf_lst) * np.sqrt(self.config.tradeDays_per_year) # Daily Risk to Anuual Risk
        dailySR = ((dailyAnnualReturn_lst[1:] * 100) - self.config.mkt_rf[self.config.market_name]) / (dailyRisk_lst[1:] * 100)
        dailySR = np.append(0, dailySR)
        dailySR = np.where(dailySR < 0, 0, dailySR)
        dailySR_max = np.max(dailySR)
        dailySR_min = np.min(dailySR[dailySR!=0])
        dailySR_avg = np.mean(dailySR)

        # For performance analysis
        dailyReturnRate_wocbf = np.diff(self.return_raw_lst)/np.array(self.return_raw_lst)[:-1]
        dailyReturnRate_wocbf = np.append(0, dailyReturnRate_wocbf)
        dailyAnnualReturn_wocbf_lst = np.power((1+dailyReturnRate_wocbf), self.config.tradeDays_per_year) - 1
        dailyRisk_wocbf_lst = np.array(self.risk_raw_lst) * np.sqrt(self.config.tradeDays_per_year)
        dailySR_wocbf = ((dailyAnnualReturn_wocbf_lst[1:] * 100) - self.config.mkt_rf[self.config.market_name]) / (dailyRisk_wocbf_lst[1:] * 100)
        dailySR_wocbf = np.append(0, dailySR_wocbf)
        dailySR_wocbf = np.where(dailySR_wocbf < 0, 0, dailySR_wocbf)
        dailySR_wocbf_max = np.max(dailySR_wocbf)
        dailySR_wocbf_min = np.min(dailySR_wocbf[dailySR_wocbf!=0])
        dailySR_wocbf_avg = np.mean(dailySR_wocbf)

        annualReturn_wocbf_pct = np.power((1 + ((self.return_raw_lst[-1] - self.initial_asset) / self.initial_asset)), (self.config.tradeDays_per_year/len(self.return_raw_lst))) - 1
        volatility_wocbf = np.sqrt((np.sum(np.power((dailyReturnRate_wocbf - np.mean(dailyReturnRate_wocbf)), 2)) * self.config.tradeDays_per_year / (len(self.return_raw_lst) - 1)))
        sharpeRatio_woCBF = ((annualReturn_wocbf_pct * 100) - self.config.mkt_rf[self.config.market_name])/ (volatility_wocbf * 100)
        sharpeRatio_woCBF = np.max([sharpeRatio_woCBF, 0])

        winRate = len(np.argwhere(diffPeriodAsset>0))/(len(diffPeriodAsset) + 1)

        # Strategy volatility during trading
        vol_max = np.max(self.stg_vol_lst)
        vol_min = np.min(np.array(self.stg_vol_lst)[np.array(self.stg_vol_lst)!=0])
        vol_avg = np.mean(self.stg_vol_lst)

        # short-term risk
        risk_max = np.max(self.risk_cbf_lst)
        risk_min = np.min(np.array(self.risk_cbf_lst)[np.array(self.risk_cbf_lst)!=0])
        risk_avg = np.mean(self.risk_cbf_lst)

        risk_raw_max = np.max(self.risk_raw_lst)
        risk_raw_min = np.min(np.array(self.risk_raw_lst)[np.array(self.risk_raw_lst)!=0])
        risk_raw_avg = np.mean(self.risk_raw_lst)


        cputime_use = self.end_cputime - self.start_cputime
        systime_use = self.end_systime - self.start_systime

        info_dict = {
            'ep': self.epoch, 'trading_days': self.totalTradeDay, 'annualReturn_pct': annualReturn_pct, 'volatility': volatility, 'sharpeRatio': sharpeRatio, 'sharpeRatio_wocbf': sharpeRatio_woCBF,
            'mdd': self.mdd, 'netProfit': netProfit, 'netProfit_pct': netProfit_pct, 'winRate': winRate,
            'vol_max': vol_max, 'vol_min': vol_min, 'vol_avg': vol_avg,
            'risk_max': risk_max, 'risk_min': risk_min, 'risk_avg': risk_avg,
            'riskRaw_max': risk_raw_max, 'riskRaw_min': risk_raw_min, 'riskRaw_avg': risk_raw_avg,
            'dailySR_max': dailySR_max, 'dailySR_min': dailySR_min, 'dailySR_avg': dailySR_avg, 'dailySR_wocbf_max': dailySR_wocbf_max, 'dailySR_wocbf_min': dailySR_wocbf_min, 'dailySR_wocbf_avg': dailySR_wocbf_avg,
            'dailyReturn_pct_max': dailyReturn_pct_max, 'dailyReturn_pct_min': dailyReturn_pct_min, 'dailyReturn_pct_avg': avg_dailyReturn_pct,
            'sigReturn_max': sigReturn_max, 'sigReturn_min': sigReturn_min, 
            'mdd_high': self.mdd_high, 'mdd_low': self.mdd_low, 'mdd_high_date': self.mdd_highTimepoint, 'mdd_low_date': self.mdd_lowTimepoint, 
            'final_capital': self.cur_capital, 'reward_sum': np.sum(self.reward_lst),
            'final_capital_wocbf': self.return_raw_lst[-1], 
            'solver_solvable': self.solver_stat['solvable'], 'solver_insolvable': self.solver_stat['insolvable'], 'cputime': cputime_use, 'systime': systime_use,
            'asset_lst': copy.deepcopy(self.asset_lst), 'daily_return_lst': copy.deepcopy(self.profit_lst), 'reward_lst': copy.deepcopy(self.reward_lst), 
            'stg_vol_lst': copy.deepcopy(self.stg_vol_lst), 'risk_lst': copy.deepcopy(self.risk_cbf_lst), 'risk_wocbf_lst': copy.deepcopy(self.risk_raw_lst),
            'capital_wocbf_lst': copy.deepcopy(self.return_raw_lst), 'daily_sr_lst': copy.deepcopy(dailySR), 'daily_sr_wocbf_lst': copy.deepcopy(dailySR_wocbf),
            'risk_adj_lst': copy.deepcopy(self.risk_adj_lst),
        }

        return info_dict

    def save_profile(self, invest_profile):
        # basic data
        for fname in self.profile_hist_field_lst:
            if fname in list(invest_profile.keys()):
                self.profile_hist_ep[fname].append(invest_profile[fname])
            else:
                raise ValueError('Cannot find the field [{}] in invest profile..'.format(fname))
        phist_df = pd.DataFrame(self.profile_hist_ep, columns=self.profile_hist_field_lst)
        phist_df.to_csv(os.path.join(self.config.res_dir, '{}_profile.csv'.format(self.mode)), index=False)

        cputime_avg = np.mean(phist_df['cputime'])
        systime_avg = np.mean(phist_df['systime'])

        bestmodel_dict = {}
        v = np.max(phist_df['final_capital'])
        v_ep = list(phist_df[phist_df['final_capital']==v]['ep'])[-1]
        bestmodel_dict['max_capital_ep'] = v_ep
        bestmodel_dict['max_capital'] = v

        if self.mode != 'test':
            print("-"*30)
            log_str = "Mode: {}, Ep: {}, Current epoch capital: {}, historical best captial ({} ep): {}, cputime cur: {} s, avg: {} s, system time cur: {} s/ep, avg: {} s/ep..".format(self.mode, self.epoch, self.cur_capital, v_ep, v, np.round(np.array(phist_df['cputime'])[-1], 2), np.round(cputime_avg, 2), np.round(np.array(phist_df['systime'])[-1], 2), np.round(systime_avg, 2))
            print(log_str)
            print("-"*30)

        bestmodel_df = pd.DataFrame([bestmodel_dict])
        bestmodel_df.to_csv(os.path.join(self.config.res_dir, '{}_bestmodel.csv'.format(self.mode)), index=False)

        # save data of each step in 1st/best/last model
        fpath = os.path.join(self.config.res_dir, '{}_stepdata.csv'.format(self.mode))
        if not os.path.exists(fpath):
            step_data = {'capital_policy_1': invest_profile['asset_lst'], 'dailyReturn_policy_1': invest_profile['daily_return_lst'],
                        'reward_policy_1': invest_profile['reward_lst'], 'strategyVolatility_policy_1': invest_profile['stg_vol_lst'],
                        'risk_policy_1': invest_profile['risk_lst'], 'risk_wocbf_policy_1': invest_profile['risk_wocbf_lst'], 'capital_wocbf_policy_1': invest_profile['capital_wocbf_lst'],
                        'dailySR_policy_1': invest_profile['daily_sr_lst'], 'dailySR_wocbf_policy_1': invest_profile['daily_sr_wocbf_lst'], 
                        'riskAccepted_policy_1': invest_profile['risk_adj_lst']}
            step_data = pd.DataFrame(step_data)
        else:
            step_data = pd.DataFrame(pd.read_csv(fpath, header=0))
            
        if bestmodel_dict['{}_ep'.format(self.config.trained_best_model_type)] == invest_profile['ep']:
            step_data['capital_policy_best'] = invest_profile['asset_lst']
            step_data['dailyReturn_policy_best'] = invest_profile['daily_return_lst']
            step_data['reward_policy_best'] = invest_profile['reward_lst']
            step_data['strategyVolatility_policy_best'] = invest_profile['stg_vol_lst']  
            step_data['risk_policy_best'] = invest_profile['risk_lst']
            step_data['risk_wocbf_policy_best'] = invest_profile['risk_wocbf_lst']
            step_data['capital_wocbf_policy_best'] = invest_profile['capital_wocbf_lst']
            step_data['dailySR_policy_best'] = invest_profile['daily_sr_lst']
            step_data['dailySR_wocbf_policy_best'] = invest_profile['daily_sr_wocbf_lst']
            step_data['riskAccepted_policy_best'] = invest_profile['risk_adj_lst']

        # Record the test set performance on valid_best_policy
        if self.mode == 'test':
            valid_fpath = os.path.join(self.config.res_dir, 'valid_bestmodel.csv')
            if os.path.exists(valid_fpath):
                valid_records = pd.DataFrame(pd.read_csv(valid_fpath, header=0))
                if int(valid_records['{}_ep'.format(self.config.trained_best_model_type)][0]) == invest_profile['ep']:
                    step_data['capital_policy_validbest'] = invest_profile['asset_lst']
                    step_data['dailyReturn_policy_validbest'] = invest_profile['daily_return_lst']
                    step_data['reward_policy_validbest'] = invest_profile['reward_lst']
                    step_data['strategyVolatility_policy_validbest'] = invest_profile['stg_vol_lst']
                    step_data['risk_policy_validbest'] = invest_profile['risk_lst']
                    step_data['risk_wocbf_policy_validbest'] = invest_profile['risk_wocbf_lst']
                    step_data['capital_wocbf_policy_validbest'] = invest_profile['capital_wocbf_lst']
                    step_data['dailySR_policy_validbest'] = invest_profile['daily_sr_lst']
                    step_data['dailySR_wocbf_policy_validbest'] = invest_profile['daily_sr_wocbf_lst']
                    step_data['riskAccepted_policy_validbest'] = invest_profile['risk_adj_lst']
                print("-"*30)
                log_str = "Mode: {}, Ep: {}, Capital on test set (by using the best validation model, {} ep): {}, cputime cur: {} s, avg: {} s, system time cur: {} s/ep, avg: {} s/ep..".format(self.mode, self.epoch, int(valid_records['{}_ep'.format(self.config.trained_best_model_type)][0]), np.array(step_data['capital_policy_validbest'])[-1], np.round(np.array(phist_df['cputime'])[-1], 2), np.round(cputime_avg, 2), np.round(np.array(phist_df['systime'])[-1], 2), np.round(systime_avg, 2))
                print(log_str)
                print("-"*30)

        if invest_profile['ep'] == self.config.num_epochs:
            step_data['capital_policy_last'] = invest_profile['asset_lst']
            step_data['dailyReturn_policy_last'] = invest_profile['daily_return_lst']
            step_data['reward_policy_last'] = invest_profile['reward_lst']
            step_data['strategyVolatility_policy_last'] = invest_profile['stg_vol_lst']
            step_data['risk_policy_last'] = invest_profile['risk_lst']
            step_data['risk_wocbf_policy_last'] = invest_profile['risk_wocbf_lst']
            step_data['capital_wocbf_policy_last'] = invest_profile['capital_wocbf_lst']
            step_data['dailySR_policy_last'] = invest_profile['daily_sr_lst']
            step_data['dailySR_wocbf_policy_last'] = invest_profile['daily_sr_wocbf_lst']
            step_data['riskAccepted_policy_last'] = invest_profile['risk_adj_lst']         
        step_data.to_csv(fpath, index=False)
 


class StockPortfolioEnv_cash(StockPortfolioEnv):

    def step(self, actions):
        self.terminal = self.curTradeDay >= (self.totalTradeDay - 1)
        if self.terminal:
            self.end_cputime = time.process_time()
            self.end_systime = time.perf_counter()
            self.model_save_flag = True
            invest_profile = self.get_results()
            self.save_profile(invest_profile=invest_profile)
            return self.state, self.reward, self.terminal, {}
        else:
            actions = np.reshape(actions, (-1)) # [1, num_of_stocks] or [num_of_stocks, ]
            weights = self.weights_normalization(actions=actions) # Unnormalized weights -> normalized weights 
            self.actions_memory.append(weights[1:]) # Remove cash
            lastDayData = self.curData            
            # Jump to next day
            self.curTradeDay = self.curTradeDay + 1
            self.curData = copy.deepcopy(self.rawdata.loc[self.curTradeDay, :])
            self.curData.sort_values(['stock'], ascending=True, inplace=True)
            self.curData.reset_index(drop=True, inplace=True)
            self.covs = np.array(self.curData['cov'].values[0])
            self.state = np.append(self.covs, np.transpose(self.curData[self.tech_indicator_lst_wocov].values), axis=0)
            self.state = self.state.flatten()
            self.ctl_state = {k:np.array(list(self.curData[k].values)) for k in self.config.otherRef_indicator_lst} # State data for the controller

            curDay_ClosePrice_withSlippage = np.array(self.curData['close'].values) * (1 + (np.random.random(self.stock_num) * (self.slippage * 2) - self.slippage))
            sigDayReturn = ((curDay_ClosePrice_withSlippage / np.array(lastDayData['close'].values)) - 1) * weights[1:] # [s1_pct, s2_pct, .., px_pct_returns]
            poDayReturn = np.sum(sigDayReturn)
            poDayReturn_withcost = poDayReturn - self.transaction_cost
            updatePoValue = self.cur_capital * (1 + poDayReturn_withcost) # = cur_capital * (1 + poDayReturn_withcost + w_cash*0 )
            self.cur_capital = updatePoValue
            self.state = np.append(self.state, [np.log((self.cur_capital/self.initial_asset))], axis=0) # Account observation
            
            self.profit_lst.append(poDayReturn_withcost) 
            cur_date = self.curData['date'].unique()[0]
            self.date_memory.append(cur_date)
            self.asset_lst.append(self.cur_capital)

            # Adaptive Risk Strategy (ARS) module
            if self.config.is_dynamic_risk_bound:
                if len(self.profit_lst) <= self.config.ref_return_lookback:
                    dynamic_risk = self.config.risk_default
                else:
                    ma_r_daily = np.mean(self.profit_lst[-self.config.ref_return_lookback:])
                    ma_r_annual = np.power((1 + ma_r_daily), (self.config.tradeDays_per_year)) - 1

                    dynamic_risk = (self.linearF_a * ma_r_annual) + self.linearF_b
                    dynamic_risk = np.clip(dynamic_risk, self.config.risk_accepted, self.config.risk_accepted_max)

                self.risk_adj_lst.append(dynamic_risk)
            else:
                self.risk_adj_lst.append(self.config.risk_default)

            # For performance analysis. Record the risk without using risk controllrt during the validation/test period.
            daily_return_ay = np.array(list(self.curData['DAILYRETURNS-{}'.format(self.config.dailyRetun_lookback)].values))
            cur_cov = np.cov(daily_return_ay) 
            self.risk_cbf_lst.append(np.sqrt(np.matmul(np.matmul(weights[1:], cur_cov), weights[1:].T)))
            w_rl = weights[1:] - self.action_cbf_memeory[-1]
            self.risk_raw_lst.append(np.sqrt(np.matmul(np.matmul(w_rl, cur_cov), w_rl.T)))
            return_raw = self.return_raw_lst[-1] * (1 + (np.sum(((curDay_ClosePrice_withSlippage / np.array(lastDayData['close'].values)) - 1) * w_rl) - self.transaction_cost))
            self.return_raw_lst.append(return_raw)

            if self.cur_capital >= self.mdd_curHigh:
                self.mdd_curHigh = self.cur_capital
                self.mdd_curHighTimepoint = cur_date
                self.mdd_curLow = self.cur_capital
                self.mdd_curLowTimepoint = cur_date
            else:
                if self.cur_capital <= self.mdd_curLow:
                    self.mdd_curLow = self.cur_capital
                    self.mdd_curLowTimepoint = cur_date

                    if ((self.mdd_curHigh - self.mdd_curLow)/self.mdd_curHigh) >= self.mdd:
                        self.mdd = (self.mdd_curHigh - self.mdd_curLow)/self.mdd_curHigh
                        self.mdd_high = self.mdd_curHigh
                        self.mdd_low = self.mdd_curLow
                        self.mdd_highTimepoint = self.mdd_curHighTimepoint
                        self.mdd_lowTimepoint = self.mdd_curLowTimepoint

            self.reward = poDayReturn_withcost * 100 * self.reward_scaling
            self.reward_lst.append(self.reward)
            cur_stg_vol = np.sqrt((np.sum(np.power((self.profit_lst - np.mean(self.profit_lst)), 2)) * self.config.tradeDays_per_year / (len(self.profit_lst) - 1)))
            self.stg_vol_lst.append(cur_stg_vol)
            self.model_save_flag = False

            # Dynamic Contribution Mechanism (DCM) module
            if self.config.is_switch_weighting:
                if len(self.profit_lst) <= self.config.sw_ref_return_lookback:
                    self.sw_weight_lst.append(1.0)
                else:
                    ma_r_daily = np.mean(self.profit_lst[-self.config.sw_ref_return_lookback:])
                    risk_free = self.config.mkt_rf[self.config.market_name] / 100
                    # |Rs - rf|
                    rs_norm = ma_r_daily - risk_free
                    if rs_norm < 0:
                        ma_abs = np.abs(rs_norm)
                        ma_abs = np.min([ma_abs, self.config.sw_loss_lb])
                        if self.config.sw_is_linear:
                            sw_update_val = 1/self.config.sw_loss_lb * ma_abs # Linear
                        else:
                            # Non-linear
                            sw_update_val = np.power((self.config.sw_m + (ma_abs/self.config.sw_loss_lb)), (1-(ma_abs/self.config.sw_loss_lb)))
                        sw_update_val = np.min([sw_update_val, 1.0])
                        self.sw_weight_lst.append(sw_update_val)
                    else:
                        self.sw_weight_lst.append(self.config.sw_m)
            else:
                self.sw_weight_lst.append(1.0)

            return self.state, self.reward, self.terminal, {}
