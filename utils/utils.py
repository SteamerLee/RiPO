#！/usr/bin/python
# -*- coding: utf-8 -*-#
'''
---------------------------------
 Name: utils.py  
 Author: RiPO
--------------------------------
'''
import numpy as np
import os
import pandas as pd
import copy

class LossRecord(object):
    def __init__(self, batch_size, mode='avg'):
        self.rec_loss = 0
        self.count = 0
        if mode == 'avg':
            self.num_of_samples = 1
        elif mode == 'sum':
            self.num_of_samples = batch_size
        else:
            raise ValueError("Unexpected mode [{}] in LossRecord.".format(mode))
    def update(self, loss):
        if isinstance(loss, list):
            avg_loss = sum(loss)
            avg_loss = avg_loss / (len(loss)*self.num_of_samples)
            self.rec_loss += avg_loss
            self.count += 1
        if isinstance(loss, float):
            self.rec_loss += loss/self.num_of_samples
            self.count += 1

    def get_val(self, init=False):
        pop_loss = self.rec_loss / self.count
        if init:
            self.rec_loss = 0
            self.count = 0
        return pop_loss
    
def post_process(config):
    # post-process after training
    fdir = os.path.join('./res', 'tmp_{}'.format(config.tmp_name))
    os.makedirs(fdir, exist_ok=True)
    # Extract the annuralized return, MDD, sharpe ratio, final capital from the profile files.
    if config.mode == 'Benchmark':
        correspond_profile = pd.DataFrame(pd.read_csv(os.path.join(config.res_dir, 'test_profile.csv'), header=0))
    elif (config.mode == 'RLcontroller') or (config.mode == 'RLonly'):
        valid_best_file = pd.DataFrame(pd.read_csv(os.path.join(config.res_dir, 'valid_bestmodel.csv'), header=0))
        best_ep = int(valid_best_file['max_capital_ep'].values[0])
        test_profile = pd.DataFrame(pd.read_csv(os.path.join(config.res_dir, 'test_profile.csv'), header=0))
        correspond_profile = test_profile[test_profile['ep'] == best_ep]
    else:
        raise ValueError('Unexpected mode {}'.format(config.mode))
    final_capital_corr = float(correspond_profile['final_capital'].values[0])
    ar_corr = float(correspond_profile['annualReturn_pct'].values[0])
    mdd_corr = float(correspond_profile['mdd'].values[0])
    sr_corr = float(correspond_profile['sharpeRatio'].values[0])
    # check wether the path exists.
    # info: mode, market, period_mode, rl_model, stock_num, 
    # Record the current date, final capital, AR, MDD, SR
    sigdata = {'current_date': [config.cur_datetime], 'final_capital': [final_capital_corr], 'annualReturn_pct': [ar_corr], 'mdd': [mdd_corr], 'sharpeRatio': [sr_corr], 'seed': [config.seed_num]} 
    fname = '{}_{}_{}_{}_{}_{}_{}_{}_{}.csv'.format(config.mode, config.market_name, config.period_mode, config.rl_model_name, config.topK, config.sw_m, config.sw_loss_lb, config.is_dynamic_risk_bound, config.is_switch_weighting)
    fpath = os.path.join(fdir, fname)
    if os.path.exists(fpath):
        rec_data = pd.DataFrame(pd.read_csv(fpath, header=0))
        # merge dataframes
        rec_data = pd.concat([rec_data, pd.DataFrame(sigdata)], axis=0, join='outer', ignore_index=True)
        rec_data.sort_values(by=['seed'], ascending=True, inplace=True)
        rec_data.reset_index(drop=True, inplace=True)
    else:
        rec_data = pd.DataFrame(sigdata)
    rec_data = copy.deepcopy(rec_data[['current_date', 'final_capital', 'annualReturn_pct', 'mdd', 'sharpeRatio', 'seed']])
    rec_data.to_csv(fpath, index=False)

    # calculate the average/std/max/min/median of the final capital, AR, MDD, SR, the folder name(current date) of exp list.
    date_lst = str(rec_data['current_date'].values.tolist())
    fc_avg = rec_data['final_capital'].mean()
    fc_std = rec_data['final_capital'].std() # ddof=1
    fc_best = rec_data['final_capital'].max()
    fc_worst = rec_data['final_capital'].min()
    fc_median = rec_data['final_capital'].median()

    ar_avg = rec_data['annualReturn_pct'].mean()
    ar_std = rec_data['annualReturn_pct'].std() # ddof=1
    ar_best = rec_data['annualReturn_pct'].max()
    ar_worst = rec_data['annualReturn_pct'].min()
    ar_median = rec_data['annualReturn_pct'].median()

    mdd_avg = rec_data['mdd'].mean()
    mdd_std = rec_data['mdd'].std() # ddof=1
    mdd_best = rec_data['mdd'].min()
    mdd_worst = rec_data['mdd'].max()
    mdd_median = rec_data['mdd'].median()

    sr_avg = rec_data['sharpeRatio'].mean()
    sr_std = rec_data['sharpeRatio'].std() # ddof=1
    sr_best = rec_data['sharpeRatio'].max()
    sr_worst = rec_data['sharpeRatio'].min()
    sr_median = rec_data['sharpeRatio'].median()

    num_of_cur_runs = len(rec_data)
    fpath = os.path.join('./res', 'summary_ref_{}.csv'.format(config.tmp_name)) # Just for reference, not accurate when conducting the hyper-para/ablation experiments as all those results will be summarized in the same record.
    if os.path.exists(fpath):
        summary_data = pd.DataFrame(pd.read_csv(fpath, header=0))
        sig_summarydata = summary_data[(summary_data['mode']==config.mode) & (summary_data['market_name']==config.market_name) & (summary_data['period_mode']==config.period_mode) & (summary_data['algorithm']==config.rl_model_name) & (summary_data['stock_num']==config.topK) 
                                    & (summary_data['m']==config.sw_m) & (summary_data['v']==config.sw_loss_lb) & (summary_data['ars']==config.is_dynamic_risk_bound) & (summary_data['dcm']==config.is_switch_weighting)]
        if len(sig_summarydata) == 0:
            sig_summarydata = {'num_of_runs': [num_of_cur_runs], 'mode': [config.mode], 'market_name': [config.market_name], 'period_mode': [config.period_mode], 'algorithm': [config.rl_model_name], 'stock_num': [config.topK], 'm': [config.sw_m], 'v': [config.sw_loss_lb], 'ars': [config.is_dynamic_risk_bound], 'dcm': [config.is_switch_weighting],
                            'final_capital_avg': [fc_avg], 'final_capital_std': [fc_std], 'final_capital_best': [fc_best], 'final_capital_worst': [fc_worst], 'final_capital_median': [fc_median],
                            'annualReturn_pct_avg': [ar_avg], 'annualReturn_pct_std': [ar_std], 'annualReturn_pct_best': [ar_best], 'annualReturn_pct_worst': [ar_worst], 'annualReturn_pct_median': [ar_median],
                            'mdd_avg': [mdd_avg], 'mdd_std': [mdd_std], 'mdd_best': [mdd_best], 'mdd_worst': [mdd_worst], 'mdd_median': [mdd_median],
                            'sharpeRatio_avg': [sr_avg], 'sharpeRatio_std': [sr_std], 'sharpeRatio_best': [sr_best], 'sharpeRatio_worst': [sr_worst], 'sharpeRatio_median': [sr_median],
                            'date_list': [date_lst]}
            sig_summarydata = pd.DataFrame(sig_summarydata)
            summary_data = pd.concat([summary_data, sig_summarydata], axis=0, join='outer', ignore_index=True)

        elif len(sig_summarydata) == 1:
            summary_data.loc[(summary_data['mode']==config.mode) & (summary_data['market_name']==config.market_name) & (summary_data['period_mode']==config.period_mode) & (summary_data['algorithm']==config.rl_model_name) & (summary_data['stock_num']==config.topK) & (summary_data['m']==config.sw_m) & (summary_data['v']==config.sw_loss_lb) & (summary_data['ars']==config.is_dynamic_risk_bound) & (summary_data['dcm']==config.is_switch_weighting), 'num_of_runs'] = num_of_cur_runs
            summary_data.loc[(summary_data['mode']==config.mode) & (summary_data['market_name']==config.market_name) & (summary_data['period_mode']==config.period_mode) & (summary_data['algorithm']==config.rl_model_name) & (summary_data['stock_num']==config.topK) & (summary_data['m']==config.sw_m) & (summary_data['v']==config.sw_loss_lb) & (summary_data['ars']==config.is_dynamic_risk_bound) & (summary_data['dcm']==config.is_switch_weighting), 'final_capital_avg'] = fc_avg
            summary_data.loc[(summary_data['mode']==config.mode) & (summary_data['market_name']==config.market_name) & (summary_data['period_mode']==config.period_mode) & (summary_data['algorithm']==config.rl_model_name) & (summary_data['stock_num']==config.topK) & (summary_data['m']==config.sw_m) & (summary_data['v']==config.sw_loss_lb) & (summary_data['ars']==config.is_dynamic_risk_bound) & (summary_data['dcm']==config.is_switch_weighting), 'final_capital_std'] = fc_std
            summary_data.loc[(summary_data['mode']==config.mode) & (summary_data['market_name']==config.market_name) & (summary_data['period_mode']==config.period_mode) & (summary_data['algorithm']==config.rl_model_name) & (summary_data['stock_num']==config.topK) & (summary_data['m']==config.sw_m) & (summary_data['v']==config.sw_loss_lb) & (summary_data['ars']==config.is_dynamic_risk_bound) & (summary_data['dcm']==config.is_switch_weighting), 'final_capital_best'] = fc_best
            summary_data.loc[(summary_data['mode']==config.mode) & (summary_data['market_name']==config.market_name) & (summary_data['period_mode']==config.period_mode) & (summary_data['algorithm']==config.rl_model_name) & (summary_data['stock_num']==config.topK) & (summary_data['m']==config.sw_m) & (summary_data['v']==config.sw_loss_lb) & (summary_data['ars']==config.is_dynamic_risk_bound) & (summary_data['dcm']==config.is_switch_weighting), 'final_capital_worst'] = fc_worst
            summary_data.loc[(summary_data['mode']==config.mode) & (summary_data['market_name']==config.market_name) & (summary_data['period_mode']==config.period_mode) & (summary_data['algorithm']==config.rl_model_name) & (summary_data['stock_num']==config.topK) & (summary_data['m']==config.sw_m) & (summary_data['v']==config.sw_loss_lb) & (summary_data['ars']==config.is_dynamic_risk_bound) & (summary_data['dcm']==config.is_switch_weighting), 'final_capital_median'] = fc_median

            summary_data.loc[(summary_data['mode']==config.mode) & (summary_data['market_name']==config.market_name) & (summary_data['period_mode']==config.period_mode) & (summary_data['algorithm']==config.rl_model_name) & (summary_data['stock_num']==config.topK) & (summary_data['m']==config.sw_m) & (summary_data['v']==config.sw_loss_lb) & (summary_data['ars']==config.is_dynamic_risk_bound) & (summary_data['dcm']==config.is_switch_weighting), 'annualReturn_pct_avg'] = ar_avg
            summary_data.loc[(summary_data['mode']==config.mode) & (summary_data['market_name']==config.market_name) & (summary_data['period_mode']==config.period_mode) & (summary_data['algorithm']==config.rl_model_name) & (summary_data['stock_num']==config.topK) & (summary_data['m']==config.sw_m) & (summary_data['v']==config.sw_loss_lb) & (summary_data['ars']==config.is_dynamic_risk_bound) & (summary_data['dcm']==config.is_switch_weighting), 'annualReturn_pct_std'] = ar_std
            summary_data.loc[(summary_data['mode']==config.mode) & (summary_data['market_name']==config.market_name) & (summary_data['period_mode']==config.period_mode) & (summary_data['algorithm']==config.rl_model_name) & (summary_data['stock_num']==config.topK) & (summary_data['m']==config.sw_m) & (summary_data['v']==config.sw_loss_lb) & (summary_data['ars']==config.is_dynamic_risk_bound) & (summary_data['dcm']==config.is_switch_weighting), 'annualReturn_pct_best'] = ar_best
            summary_data.loc[(summary_data['mode']==config.mode) & (summary_data['market_name']==config.market_name) & (summary_data['period_mode']==config.period_mode) & (summary_data['algorithm']==config.rl_model_name) & (summary_data['stock_num']==config.topK) & (summary_data['m']==config.sw_m) & (summary_data['v']==config.sw_loss_lb) & (summary_data['ars']==config.is_dynamic_risk_bound) & (summary_data['dcm']==config.is_switch_weighting), 'annualReturn_pct_worst'] = ar_worst
            summary_data.loc[(summary_data['mode']==config.mode) & (summary_data['market_name']==config.market_name) & (summary_data['period_mode']==config.period_mode) & (summary_data['algorithm']==config.rl_model_name) & (summary_data['stock_num']==config.topK) & (summary_data['m']==config.sw_m) & (summary_data['v']==config.sw_loss_lb) & (summary_data['ars']==config.is_dynamic_risk_bound) & (summary_data['dcm']==config.is_switch_weighting), 'annualReturn_pct_median'] = ar_median

            summary_data.loc[(summary_data['mode']==config.mode) & (summary_data['market_name']==config.market_name) & (summary_data['period_mode']==config.period_mode) & (summary_data['algorithm']==config.rl_model_name) & (summary_data['stock_num']==config.topK) & (summary_data['m']==config.sw_m) & (summary_data['v']==config.sw_loss_lb) & (summary_data['ars']==config.is_dynamic_risk_bound) & (summary_data['dcm']==config.is_switch_weighting), 'mdd_avg'] = mdd_avg
            summary_data.loc[(summary_data['mode']==config.mode) & (summary_data['market_name']==config.market_name) & (summary_data['period_mode']==config.period_mode) & (summary_data['algorithm']==config.rl_model_name) & (summary_data['stock_num']==config.topK) & (summary_data['m']==config.sw_m) & (summary_data['v']==config.sw_loss_lb) & (summary_data['ars']==config.is_dynamic_risk_bound) & (summary_data['dcm']==config.is_switch_weighting), 'mdd_std'] = mdd_std
            summary_data.loc[(summary_data['mode']==config.mode) & (summary_data['market_name']==config.market_name) & (summary_data['period_mode']==config.period_mode) & (summary_data['algorithm']==config.rl_model_name) & (summary_data['stock_num']==config.topK) & (summary_data['m']==config.sw_m) & (summary_data['v']==config.sw_loss_lb) & (summary_data['ars']==config.is_dynamic_risk_bound) & (summary_data['dcm']==config.is_switch_weighting), 'mdd_best'] = mdd_best
            summary_data.loc[(summary_data['mode']==config.mode) & (summary_data['market_name']==config.market_name) & (summary_data['period_mode']==config.period_mode) & (summary_data['algorithm']==config.rl_model_name) & (summary_data['stock_num']==config.topK) & (summary_data['m']==config.sw_m) & (summary_data['v']==config.sw_loss_lb) & (summary_data['ars']==config.is_dynamic_risk_bound) & (summary_data['dcm']==config.is_switch_weighting), 'mdd_worst'] = mdd_worst
            summary_data.loc[(summary_data['mode']==config.mode) & (summary_data['market_name']==config.market_name) & (summary_data['period_mode']==config.period_mode) & (summary_data['algorithm']==config.rl_model_name) & (summary_data['stock_num']==config.topK) & (summary_data['m']==config.sw_m) & (summary_data['v']==config.sw_loss_lb) & (summary_data['ars']==config.is_dynamic_risk_bound) & (summary_data['dcm']==config.is_switch_weighting), 'mdd_median'] = mdd_median

            summary_data.loc[(summary_data['mode']==config.mode) & (summary_data['market_name']==config.market_name) & (summary_data['period_mode']==config.period_mode) & (summary_data['algorithm']==config.rl_model_name) & (summary_data['stock_num']==config.topK) & (summary_data['m']==config.sw_m) & (summary_data['v']==config.sw_loss_lb) & (summary_data['ars']==config.is_dynamic_risk_bound) & (summary_data['dcm']==config.is_switch_weighting), 'sharpeRatio_avg'] = sr_avg
            summary_data.loc[(summary_data['mode']==config.mode) & (summary_data['market_name']==config.market_name) & (summary_data['period_mode']==config.period_mode) & (summary_data['algorithm']==config.rl_model_name) & (summary_data['stock_num']==config.topK) & (summary_data['m']==config.sw_m) & (summary_data['v']==config.sw_loss_lb) & (summary_data['ars']==config.is_dynamic_risk_bound) & (summary_data['dcm']==config.is_switch_weighting), 'sharpeRatio_std'] = sr_std
            summary_data.loc[(summary_data['mode']==config.mode) & (summary_data['market_name']==config.market_name) & (summary_data['period_mode']==config.period_mode) & (summary_data['algorithm']==config.rl_model_name) & (summary_data['stock_num']==config.topK) & (summary_data['m']==config.sw_m) & (summary_data['v']==config.sw_loss_lb) & (summary_data['ars']==config.is_dynamic_risk_bound) & (summary_data['dcm']==config.is_switch_weighting), 'sharpeRatio_best'] = sr_best
            summary_data.loc[(summary_data['mode']==config.mode) & (summary_data['market_name']==config.market_name) & (summary_data['period_mode']==config.period_mode) & (summary_data['algorithm']==config.rl_model_name) & (summary_data['stock_num']==config.topK) & (summary_data['m']==config.sw_m) & (summary_data['v']==config.sw_loss_lb) & (summary_data['ars']==config.is_dynamic_risk_bound) & (summary_data['dcm']==config.is_switch_weighting), 'sharpeRatio_worst'] = sr_worst
            summary_data.loc[(summary_data['mode']==config.mode) & (summary_data['market_name']==config.market_name) & (summary_data['period_mode']==config.period_mode) & (summary_data['algorithm']==config.rl_model_name) & (summary_data['stock_num']==config.topK) & (summary_data['m']==config.sw_m) & (summary_data['v']==config.sw_loss_lb) & (summary_data['ars']==config.is_dynamic_risk_bound) & (summary_data['dcm']==config.is_switch_weighting), 'sharpeRatio_median'] = sr_median

            summary_data.loc[(summary_data['mode']==config.mode) & (summary_data['market_name']==config.market_name) & (summary_data['period_mode']==config.period_mode) & (summary_data['algorithm']==config.rl_model_name) & (summary_data['stock_num']==config.topK) & (summary_data['m']==config.sw_m) & (summary_data['v']==config.sw_loss_lb) & (summary_data['ars']==config.is_dynamic_risk_bound) & (summary_data['dcm']==config.is_switch_weighting), 'date_list'] = date_lst

        else:
            raise ValueError('Unexpected length of sig_summarydata {}'.format(len(sig_summarydata)))

    else:
        sig_summarydata ={'num_of_runs': [num_of_cur_runs], 'mode': [config.mode], 'market_name': [config.market_name], 'period_mode': [config.period_mode], 'algorithm': [config.rl_model_name], 'stock_num': [config.topK], 'm': [config.sw_m], 'v': [config.sw_loss_lb], 'ars': [config.is_dynamic_risk_bound], 'dcm': [config.is_switch_weighting],
                            'final_capital_avg': [fc_avg], 'final_capital_std': [fc_std], 'final_capital_best': [fc_best], 'final_capital_worst': [fc_worst], 'final_capital_median': [fc_median],
                            'annualReturn_pct_avg': [ar_avg], 'annualReturn_pct_std': [ar_std], 'annualReturn_pct_best': [ar_best], 'annualReturn_pct_worst': [ar_worst], 'annualReturn_pct_median': [ar_median],
                            'mdd_avg': [mdd_avg], 'mdd_std': [mdd_std], 'mdd_best': [mdd_best], 'mdd_worst': [mdd_worst], 'mdd_median': [mdd_median],
                            'sharpeRatio_avg': [sr_avg], 'sharpeRatio_std': [sr_std], 'sharpeRatio_best': [sr_best], 'sharpeRatio_worst': [sr_worst], 'sharpeRatio_median': [sr_median],
                            'date_list': [date_lst]}
        summary_data = pd.DataFrame(sig_summarydata)
    summary_data = summary_data[['num_of_runs', 'mode', 'market_name', 'period_mode', 'algorithm', 'stock_num', 'm', 'v', 'ars', 'dcm', 
                                    'final_capital_avg', 'final_capital_std', 'annualReturn_pct_avg', 'annualReturn_pct_std', 'mdd_avg', 'mdd_std', 'sharpeRatio_avg', 'sharpeRatio_std', 
                                    'final_capital_best', 'final_capital_worst', 'final_capital_median', 'annualReturn_pct_best', 'annualReturn_pct_worst', 'annualReturn_pct_median', 'mdd_best', 'mdd_worst', 'mdd_median', 'sharpeRatio_best', 'sharpeRatio_worst', 'sharpeRatio_median', 
                                    'date_list']]
    summary_data.to_csv(fpath, index=False)

    print("Done [post-process]")

