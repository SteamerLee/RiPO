#！/usr/bin/python
# -*- coding: utf-8 -*-#

'''
---------------------------------
 Name:         config.py
 Description:  configuration file 
 Author:       RiPO

Referenced Repository in Github
Imple. of compared methods: https://github.com/ZhengyaoJiang/PGPortfolio/blob/48cc5a4af5edefd298e7801b95b0d4696f5175dd/pgportfolio/tdagent/tdagent.py#L7
RL-based agent (TD3 imple.): Baselines3 (https://stable-baselines3.readthedocs.io/en/master/modules/td3.html)
Trading environment: FinRL (https://github.com/AI4Finance-Foundation/FinRL)
Technical indicator imple.: TA-Lib (https://github.com/TA-Lib/ta-lib-python)
Second-order cone programming solver: CVXOPT (http://cvxopt.org/) 
---------------------------------
'''

import numpy as np
import os
import pandas as pd
import time
import datetime

class Config():

    def __init__(self, seed_num=2022, benchmark_algo=None, current_date=None, period_mode=None, topK=None):
        self.notes = 'RiPO'

        # When the selected base algorithm is 'TD3', setting self.mode = 'RLcontroller' will run the RiPO algorithm with TD3-based trading agents, OR setting self.mode = 'RLonly' will run the TD3 algorithm only.
        # When the selected algorithm is the conventional benchmark algorithm, pls set self.mode = 'Benchmark', and register benchmark algorithms after implementing them. 
        self.mode = 'RLcontroller' # 'RLonly', 'RLcontroller', 'Benchmark'
        self.rl_model_name = 'TD3'

        if period_mode is None:
            self.period_mode = 2 # 1: MS-1 (uptrend market), 2: MS-2 (downtrend market)
        else:
            self.period_mode = period_mode
        
        if topK is None:
            self.topK = 10 
        else:
            self.topK = topK

        self.is_dynamic_risk_bound = True # Enable Adaptive Risk Strategy (ARS).
        self.is_switch_weighting = True # Enable Dynamic Contribution Mechanism (DCM).
        self.controller_obj = 1 # 1: minimize the losses of profits. Not applicable.
        self.obj_weight_1 = 1 # Not applicable
        self.obj_weight_2 = 1 # Not applicable

        self.reward_scaling = 1 
        self.learning_rate = 0.0001 
        self.batch_size = 50
        self.train_freq = 400 
        self.sw_m = 0.8 # m in DCM, [0, 1], default 0 in MS-1 and 0.8 in MS-2.
        self.sw_loss_lb = 0.005 # v in DCM, default 0.5 in MS-1 and 0.005 in MS-2.
        self.mulrf = 1 # \mu in ARS, default 2 in MS-1 and 1 in MS-2.
        self.ref_return_lookback = 5 # Window size of observing recent strategy performance to estimate the expected returns. Default 3 in MS-1 and 5 in MS-2.

        # 
        self.risk_default = 0.01 # Default risk boundary
        self.risk_accepted = 0.01 # Lower bound
        self.risk_accepted_max = 0.015 # Upper bound, default 0.02 in MS-1 and 0.015 in MS-2

        self.market_name = 'SP500'
        self.dataDir = './data'
        self.num_epochs = 500 
        self.pricePredModel = 'MA' # Moving average method is applied to predict the stock price.

        self.cov_lookback = 21 # 21 days (i.e., one month) is the lookback period to calculate the covariance of stocks.
        self.norm_method = 'sum' # Normalize the weights. Option: softmax, sum 
        if benchmark_algo is None:
            self.benchmark_algo = 'CORN' # 'PAMR', 'CORN', 'CRP', 'EG', 'OLMAR', 'RAT', 'EIIE', 'PPN'
        else:
            self.benchmark_algo = benchmark_algo # For batch evaluation
            self.rl_model_name = benchmark_algo

        self.tmp_name = 'MS2' # Summary csv file name.
        self.trained_best_model_type = 'max_capital'

        self.risk_market = 0.001 # \Sigma_beta
        self.cbf_gamma = 0.3
        self.ref_sr = 3 # Not applicable

        self.sw_ref_return_lookback = self.ref_return_lookback
        self.sw_is_linear = False # For selecting the linear transformation or Non-linearn transformation.

        if self.mode == 'Benchmark':
            self.rl_model_name = self.benchmark_algo
        if current_date is None:
            self.cur_datetime = datetime.datetime.now().strftime('%Y-%m-%d-%H-%M-%S')
        else:
            self.cur_datetime = current_date
        self.res_dir = os.path.join('./res', self.mode, self.rl_model_name, '{}-{}'.format(self.market_name, self.topK), self.cur_datetime)
        os.makedirs(self.res_dir, exist_ok=True)
        
        self.res_model_dir = os.path.join(self.res_dir, 'model')
        os.makedirs(self.res_model_dir, exist_ok=True)
        self.res_img_dir = os.path.join(self.res_dir, 'graph')
        os.makedirs(self.res_img_dir, exist_ok=True)

        self.tradeDays_per_year = 252
        self.seed_num = seed_num
        
        date_split_dict = {            
            1: {'train_date_start': '2015-01-01', # MS-1
                'train_date_end': '2017-12-31',
                'valid_date_start': '2018-01-01',
                'valid_date_end': '2018-12-31',
                'test_date_start': '2019-01-01',
                'test_date_end': '2019-12-31'},

            2: {'train_date_start': '2015-01-01', # MS-2
                'train_date_end': '2019-12-31',
                'valid_date_start': '2020-01-01', 
                'valid_date_end': '2020-12-31',
                'test_date_start': '2021-01-01',
                'test_date_end': '2022-10-31'},
        }
        
        self.train_date_start = pd.Timestamp(date_split_dict[self.period_mode]['train_date_start'])
        self.train_date_end = pd.Timestamp(date_split_dict[self.period_mode]['train_date_end'])
        if (date_split_dict[self.period_mode]['valid_date_start'] is not None) and (date_split_dict[self.period_mode]['valid_date_end'] is not None):
            self.valid_date_start = pd.Timestamp(date_split_dict[self.period_mode]['valid_date_start'])
            self.valid_date_end = pd.Timestamp(date_split_dict[self.period_mode]['valid_date_end'])
        else:
            self.valid_date_start = None
            self.valid_date_end = None
        if (date_split_dict[self.period_mode]['test_date_start'] is not None) and (date_split_dict[self.period_mode]['test_date_end'] is not None):
            self.test_date_start = pd.Timestamp(date_split_dict[self.period_mode]['test_date_start'])
            self.test_date_end = pd.Timestamp(date_split_dict[self.period_mode]['test_date_end'])
        else:
            self.test_date_start = None
            self.test_date_end = None        

        # Technical indicator list        
        self.tech_indicator_talib_lst = ["AD", "ADOSC", "ADX", "ADXR", "APO", "AROONOSC", "ATR-14", "ATR-6", "BOP", "CCI-5", "CCI-10", "CCI-20", "CCI-88", 
                                            "CMO-14-close", "CMO-14-open", "DEMA-6", "DEMA-12", "DEMA-26", "DX", "EMA-6", "EMA-12", "EMA-26", "KAMA", "MACD", 
                                            "MEDPRICE", "MiNUS_DI", "MiNUS_DM", "MOM", "NATR", "OBV", "PLUS_DI", "PLUS_DM", "PPO", "ROC-6", "ROC-20", "ROCP-6",
                                            "ROCP-20", "ROC-6-volume", "ROC-20-volume", "ROCP-6-volume", "ROCP-20-volume", "RSI", "SAR", "TEMA-6", "TEMA-12", 
                                            "TEMA-26", "TRANGE", "TYPPRICE", "TSF", "ULTOSC", "WILLR", "AROON", "BBANDS"]
        self.tech_indicator_extra_lst = ['CHANGE']
        self.tech_indicator_input_lst = self.tech_indicator_talib_lst + self.tech_indicator_extra_lst
        self.dailyRetun_lookback = self.cov_lookback # Observation window size of Covariance Matrix
        self.otherRef_indicator_ma_window = 5 
        self.otherRef_indicator_std_window = 5

        self.otherRef_indicator_lst = ['MA-{}'.format(self.otherRef_indicator_ma_window), 'STDDEV-{}'.format(self.otherRef_indicator_std_window), 'DAILYRETURNS-{}'.format(self.dailyRetun_lookback)]

        self.mkt_rf = { 
            'SP500': 1.6575,
            'CSI300': 3.037,
        }

        self.invest_env_para = {
            'max_shares': 100, 'initial_asset': 1000000, 'reward_scaling': self.reward_scaling, 'norm_method': self.norm_method, 
            'transaction_cost': 0.001, 'slippage': 0.001, 'seed_num': self.seed_num
        }


        # if self.topK >= self.cov_lookback:
        #     # Tashman, L. J. 2000. Out-of-sample tests of forecasting accuracy: an analysis and review. International journal of forecasting, 16(4): 437–450.
        #     raise ValueError("The lookback period of covariance should be larger than the number of stocks. Suggested value: {} days (i.e., {} months)".format(((self.topK//21)+1)*21, (self.topK//21)+1))

        if self.risk_accepted <= self.risk_market:
            raise ValueError("The boundary of safe risk[{}] should not be less than/ equal to the market risk[{}].".format(self.risk_accepted, self.risk_market))

        self.load_para() # after seed_num, rl_model_name

    def load_para(self):
        
        base_para = {
            'policy': "MlpPolicy", 'learning_rate': self.learning_rate, 'buffer_size': 1000000,
            'learning_starts': 100, 'batch_size': self.batch_size, 'tau': 0.005, 'gamma': 0.99, 'train_freq': (self.train_freq, "step"),
            'gradient_steps': -1, 'action_noise': None,  'replay_buffer_class': None, 'replay_buffer_kwargs': None, 
            'optimize_memory_usage': False, 'tensorboard_log': None, 'policy_kwargs': None, 
            'verbose': 1, 'seed': self.seed_num, 'device': 'auto', '_init_setup_model': True,
        }

        # Para. for sepecific algorithm.
        algo_para = {
            'TD3': {'policy_delay': 2, 'target_policy_noise': 0.2, 'target_noise_clip': 0.5,},
            'SAC': {'ent_coef': 'auto', 'target_update_interval': 1, 'target_entropy': 'auto', 'use_sde': False, 'sde_sample_freq': -1, 'use_sde_at_warmup': False,},
            'PPO': {'n_steps': 100},
        }
        
        algo_para_rm_from_base = {
            'PPO': ['buffer_size', 'learning_starts', 'tau', 'train_freq', 'gradient_steps', 'action_noise', 'replay_buffer_class', 'replay_buffer_kwargs', 'optimize_memory_usage']
        }

        if self.rl_model_name in algo_para.keys():
            self.model_para = {**base_para, **algo_para[self.rl_model_name]}
        else:
            self.model_para = base_para
        
        if self.rl_model_name in algo_para_rm_from_base.keys():
            for rm_field in algo_para_rm_from_base[self.rl_model_name]:
                del self.model_para[rm_field]


    def print_config(self):
        log_str = '=' * 30 + '\n'
        para_str = '{} \n'.format(self.notes)
        log_str = log_str + para_str
        para_str = 'mode: {}, rl_model_name: {}, market_name: {}, topK: {}, dataDir: {} \n'.format(self.mode, self.rl_model_name, self.market_name, self.topK, self.dataDir)
        log_str = log_str + para_str
        para_str = 'period_mode: {}, num_epochs: {}, cov_lookback: {}, norm_method: {}, benchmark_algo: {}, trained_best_model_type: {}, pricePredModel: {}, \n'.format(self.period_mode, self.num_epochs, self.cov_lookback, self.norm_method, self.benchmark_algo, self.trained_best_model_type, self.pricePredModel)
        log_str = log_str + para_str
        para_str = 'is_dynamic_risk_bound: {}, risk_market: {}, risk_default: {}, risk_accepted: {}, risk_accepted_max: {}, cbf_gamma: {}, ref_return_lookback: {}, ref_sr: {}, mulrf: {}, \n'.format(self.is_dynamic_risk_bound, self.risk_market, self.risk_default, self.risk_accepted, self.risk_accepted_max, self.cbf_gamma, self.ref_return_lookback, self.ref_sr, self.mulrf)
        log_str = log_str + para_str
        para_str = 'cur_datetime: {}, res_dir: {}, tradeDays_per_year: {}, seed_num: {}, \n'.format(self.cur_datetime, self.res_dir, self.tradeDays_per_year, self.seed_num)
        log_str = log_str + para_str
        para_str = 'train_date_start: {}, train_date_end: {}, valid_date_start: {}, valid_date_end: {}, test_date_start: {}, test_date_end: {}, \n'.format(self.train_date_start, self.train_date_end, self.valid_date_start, self.valid_date_end, self.test_date_start, self.test_date_end)
        log_str = log_str + para_str
        para_str = 'tech_indicator_input_lst: {}, \n'.format(self.tech_indicator_input_lst)
        log_str = log_str + para_str
        para_str = 'otherRef_indicator_lst: {}, \n'.format(self.otherRef_indicator_lst)
        log_str = log_str + para_str
        para_str = 'is_switch_weighting: {}, sw_loss_lb: {}, sw_m: {}, sw_ref_return_lookback: {}, sw_is_linear: {} \n'.format(self.is_switch_weighting, self.sw_loss_lb, self.sw_m, self.sw_ref_return_lookback, self.sw_is_linear)
        log_str = log_str + para_str
        para_str = 'controller_obj: {}, obj_weight_1: {}, obj_weight_2: {}, tmp_name: {}, \n'.format(self.controller_obj, self.obj_weight_1, self.obj_weight_2, self.tmp_name)
        log_str = log_str + para_str
        para_str = 'mkt_rf: {}, \n'.format(self.mkt_rf)
        log_str = log_str + para_str
        para_str = 'invest_env_para: {}, \n'.format(self.invest_env_para)
        log_str = log_str + para_str
        para_str = 'model_para: {}, \n'.format(self.model_para)
        log_str = log_str + para_str
        log_str = log_str + '=' * 30 + '\n'

        print(log_str, flush=True)
