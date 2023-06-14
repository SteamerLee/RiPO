#！/usr/bin/python
# -*- coding: utf-8 -*-#

'''
---------------------------------
 Name: entrance.py  
 Author: RiPO
--------------------------------
'''

import os
os.environ['CUDA_DEVICE_ORDRE'] = 'PCI_BUS_ID'
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
import random 
import numpy as np
import torch as th
import datetime
import copy
th.backends.cudnn.deterministic = True
th.backends.cudnn.benchmark = False
os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
th.use_deterministic_algorithms(True)

import pandas as pd
import time
from config import Config
from utils.featGen import FeatureProcesser
from utils.tradeEnv import StockPortfolioEnv, StockPortfolioEnv_cash
from utils.model_pool import model_select, benchmark_algo_select
from utils.callback_func import PoCallback
from utils.utils import post_process

def RLonly(config):
    # Get dataset
    mkt_name = config.market_name
    fpath = os.path.join(config.dataDir, '{}_{}.csv'.format(mkt_name, config.topK))
    if not os.path.exists(fpath):
        raise ValueError("Cannot load the data from {}".format(fpath))
    data = pd.DataFrame(pd.read_csv(fpath, header=0))
    
    # Preprocess features
    featProc = FeatureProcesser(config=config)
    data_dict = featProc.preprocess_feat(data=data)
    tech_indicator_lst = featProc.techIndicatorLst 
    stock_num = data_dict['train']['stock'].nunique()
    print("Data has been processed..")

    # Initialize environment
    trainInvest_env_para = config.invest_env_para 
    env_train = StockPortfolioEnv(
        config=config, rawdata=data_dict['train'], mode='train', stock_num=stock_num, action_dim=stock_num, 
        tech_indicator_lst=tech_indicator_lst, **trainInvest_env_para
    )
    if (config.valid_date_start is not None) and (config.valid_date_end is not None):
        validInvest_env_para = config.invest_env_para 
        env_valid = StockPortfolioEnv(
            config=config, rawdata=data_dict['valid'], mode='valid', stock_num=stock_num, action_dim=stock_num, 
            tech_indicator_lst=tech_indicator_lst, **validInvest_env_para
        )
    else:
        env_valid = None

    if (config.test_date_start is not None) and (config.test_date_end is not None):
        testInvest_env_para = config.invest_env_para 
        env_test = StockPortfolioEnv(
            config=config, rawdata=data_dict['test'], mode='test', stock_num=stock_num, action_dim=stock_num, 
            tech_indicator_lst=tech_indicator_lst, **testInvest_env_para
        )
    else:
        env_test = None

    # Load RL model
    ModelCls = model_select(model_name=config.rl_model_name, mode=config.mode)
    model_para_dict = config.model_para
    po_model = ModelCls(env=env_train, **model_para_dict) # Create instance 
    total_timesteps = int(config.num_epochs * env_train.totalTradeDay)
    print('Training Start', flush=True)
    # Start to train the model.
    po_model.learn(total_timesteps=total_timesteps, log_interval=10, 
                    callback=PoCallback(config=config, train_env=env_train, valid_env=env_valid, test_env=env_test))
    del po_model
    print("Training Done...", flush=True)


def RLcontroller(config):

    # Get dataset
    mkt_name = config.market_name
    fpath = os.path.join(config.dataDir, '{}_{}.csv'.format(mkt_name, config.topK))
    if not os.path.exists(fpath):
        raise ValueError("Cannot load the data from {}".format(fpath))
    data = pd.DataFrame(pd.read_csv(fpath, header=0))
    
    # Preprocess features
    featProc = FeatureProcesser(config=config)
    data_dict = featProc.preprocess_feat(data=data)
    tech_indicator_lst = featProc.techIndicatorLst
    stock_num = data_dict['train']['stock'].nunique()
    print("Data has been processed..")
    
    # Initialize environment
    trainInvest_env_para = config.invest_env_para 
    env_train = StockPortfolioEnv(
        config=config, rawdata=data_dict['train'], mode='train', stock_num=stock_num, action_dim=stock_num, 
        tech_indicator_lst=tech_indicator_lst, **trainInvest_env_para
    )
    if (config.valid_date_start is not None) and (config.valid_date_end is not None):
        validInvest_env_para = config.invest_env_para 
        env_valid = StockPortfolioEnv(
            config=config, rawdata=data_dict['valid'], mode='valid', stock_num=stock_num, action_dim=stock_num, 
            tech_indicator_lst=tech_indicator_lst, **validInvest_env_para
        )
    else:
        env_valid = None
    if (config.test_date_start is not None) and (config.test_date_end is not None):
        testInvest_env_para = config.invest_env_para 
        env_test = StockPortfolioEnv(
            config=config, rawdata=data_dict['test'], mode='test', stock_num=stock_num, action_dim=stock_num, 
            tech_indicator_lst=tech_indicator_lst, **testInvest_env_para
        )
    else:
        env_test = None

    # Load RL model
    ModelCls = model_select(model_name=config.rl_model_name, mode=config.mode)
    model_para_dict = config.model_para
    po_model = ModelCls(env=env_train, **model_para_dict) 
    total_timesteps = int(config.num_epochs * env_train.totalTradeDay)
    print('Training Start', flush=True)
    # Start to train the model.
    po_model.learn(total_timesteps=total_timesteps, log_interval=10, 
                    callback=PoCallback(config=config, train_env=env_train, valid_env=env_valid, test_env=env_test))
    del po_model
    print("Training Done...", flush=True)

def benchmark_test(config):
    # Get dataset
    mkt_name = config.market_name
    fpath = os.path.join(config.dataDir, '{}_{}.csv'.format(mkt_name, config.topK))
    if not os.path.exists(fpath):
        raise ValueError("Cannot load the data from {}".format(fpath))
    data = pd.DataFrame(pd.read_csv(fpath, header=0))
    
    # Preprocess features
    featProc = FeatureProcesser(config=config)
    data_dict = featProc.preprocess_feat(data=data)
    tech_indicator_lst = featProc.techIndicatorLst 
    stock_num = data_dict['train']['stock'].nunique()
    print("Data has been processed..")

    if (config.valid_date_start is not None) and (config.valid_date_end is not None):
        validInvest_env_para = config.invest_env_para 
        if config.benchmark_algo in ['RAT', 'EIIE', 'PPN']:
            env_valid = StockPortfolioEnv_cash(
                config=config, rawdata=data_dict['valid'], mode='valid', stock_num=stock_num, action_dim=stock_num, 
                tech_indicator_lst=tech_indicator_lst, **validInvest_env_para
            )
        else:
            env_valid = StockPortfolioEnv(
                config=config, rawdata=data_dict['valid'], mode='valid', stock_num=stock_num, action_dim=stock_num, 
                tech_indicator_lst=tech_indicator_lst, **validInvest_env_para
            )
    else:
        env_valid = None

    if (config.test_date_start is not None) and (config.test_date_end is not None):
        testInvest_env_para = config.invest_env_para 
        if config.benchmark_algo in ['RAT', 'EIIE', 'PPN']:
            env_test = StockPortfolioEnv_cash(
                config=config, rawdata=data_dict['test'], mode='test', stock_num=stock_num, action_dim=stock_num, 
                tech_indicator_lst=tech_indicator_lst, **testInvest_env_para
            )
        else:
            env_test = StockPortfolioEnv(
                config=config, rawdata=data_dict['test'], mode='test', stock_num=stock_num, action_dim=stock_num, 
                tech_indicator_lst=tech_indicator_lst, **testInvest_env_para
            )
    else:
        env_test = None

    ModelCls = benchmark_algo_select(model_name=config.benchmark_algo)
    po_model = ModelCls(env=env_test, env_valid=env_valid)
    po_model.run()
    del po_model

def entrance():

    runtimes = 1
    
    if runtimes <= 1:
        # For single run test. Pls configure the para. in config.py
        benchmark_algo = None
        p_mode = None
        stock_x = None

        current_date = datetime.datetime.now().strftime('%Y-%m-%d-%H-%M-%S')
        rand_seed = int(time.mktime(datetime.datetime.strptime(current_date, '%Y-%m-%d-%H-%M-%S').timetuple()))
        random.seed(rand_seed)
        os.environ['PYTHONHASHSEED'] = str(rand_seed)
        np.random.seed(rand_seed)
        th.manual_seed(rand_seed)
        th.cuda.manual_seed(rand_seed)
        th.cuda.manual_seed_all(rand_seed)

        start_cputime = time.process_time()
        start_systime = time.perf_counter()
        config = Config(seed_num=rand_seed, benchmark_algo=benchmark_algo, current_date=current_date, period_mode=p_mode, topK=stock_x) 
        config.print_config()
        if config.mode == 'RLonly':
            RLonly(config=config)

        elif config.mode == 'RLcontroller':
            RLcontroller(config=config)
        
        elif config.mode == 'Benchmark':
            benchmark_test(config=config)
        else:
            raise ValueError('Unexpected mode {}'.format(config.mode))

        end_cputime = time.process_time()
        end_systime = time.perf_counter()
        print("Total cputime: {} s, system time: {} s".format(np.round(end_cputime - start_cputime, 2), np.round(end_systime - start_systime, 2)))
        # post_process(config=config) # Please dont use this function when debugging.
        # print("[End Sig] stock_num: {}, period_mode: {}, benchmark_algo: {}, run_no: {}".format(stock_x, p_mode, benchmark_algo, run_no))

    else:
        for p_mode in [2]: # [1, 2]:
            stock_x = 10  
            for benchmark_algo in ['TD3']: # ['TD3']:
                # When the selected base algorithm is 'TD3', setting self.mode = 'RLcontroller' will run the RiPO algorithm with TD3-based trading agents, OR setting self.mode = 'RLonly' will run the TD3 algorithm only.
                # When the selected algorithm is the conventional benchmark algorithm, pls set self.mode = 'Benchmark', and register benchmark algorithms after implementing them. 
                for run_no in range(1, runtimes+1):
                    current_date = datetime.datetime.now().strftime('%Y-%m-%d-%H-%M-%S')
                    rand_seed = int(time.mktime(datetime.datetime.strptime(current_date, '%Y-%m-%d-%H-%M-%S').timetuple()))

                    random.seed(rand_seed)
                    os.environ['PYTHONHASHSEED'] = str(rand_seed)
                    np.random.seed(rand_seed)
                    th.manual_seed(rand_seed)
                    th.cuda.manual_seed(rand_seed)
                    th.cuda.manual_seed_all(rand_seed)

                    start_cputime = time.process_time()
                    start_systime = time.perf_counter()
                    config = Config(seed_num=rand_seed, benchmark_algo=benchmark_algo, current_date=current_date, period_mode=p_mode, topK=stock_x) 
                    config.print_config()
                    if config.mode == 'RLonly':
                        RLonly(config=config)

                    elif config.mode == 'RLcontroller':
                        RLcontroller(config=config)
                    
                    elif config.mode == 'Benchmark':
                        benchmark_test(config=config)
                    else:
                        raise ValueError('Unexpected mode {}'.format(config.mode))

                    end_cputime = time.process_time()
                    end_systime = time.perf_counter()
                    print("Run No.: {}, Total cputime: {} s, system time: {} s".format(run_no, np.round(end_cputime - start_cputime, 2), np.round(end_systime - start_systime, 2)))
                    post_process(config=config)

                    print("[End] stock_num: {}, period_mode: {}, benchmark_algo: {}, run_no: {}".format(stock_x, p_mode, benchmark_algo, run_no))

def main():
    entrance()

if __name__ == '__main__':
    main()
