#！/usr/bin/python
# -*- coding: utf-8 -*-#
'''
---------------------------------
 Name:         featGen.py
 Description:  Technical feature generation.
 Author:       RiPO
---------------------------------
'''
import numpy as np
import pandas as pd
import copy
import os
from talib import abstract

class FeatureProcesser:
    def __init__(self, config):
        self.config = config
    
    def preprocess_feat(self, data):
        data = self.gen_feat(data=data)
        data = self.scale_feat(data=data)
        return data
    
    def gen_feat(self, data):
        data['date'] = pd.to_datetime(data['date'])
        data.sort_values(['stock', 'date'], ascending=True, inplace=True)
        data.reset_index(drop=True, inplace=True)        
        # ['date', 'stock', 'open', 'high', 'low', 'close', 'volume']
        self.rawColLst = list(data.columns)
        datax = copy.deepcopy(data)
        stock_lst = datax['stock'].unique()
        for indidx, sigIndicatorName in enumerate(list(self.config.tech_indicator_input_lst) + list(self.config.otherRef_indicator_lst)):
            if sigIndicatorName.split('-')[0] in ['DAILYRETURNS']: 
                continue
            ind_df = pd.DataFrame()
            for sigStockName in stock_lst:
                dataSig = copy.deepcopy(data[data['stock']==sigStockName])
                dataSig.sort_values(['date'], ascending=True, inplace=True)
                dataSig.reset_index(drop=True, inplace=True)
                
                if sigIndicatorName == 'CHANGE':
                    open_ay = np.array(dataSig['open'])
                    diff_ay = np.diff(open_ay)
                    open_pct = np.divide(diff_ay, open_ay[:-1], out=np.zeros_like(diff_ay), where=open_ay[:-1]!=0)
                    open_pct = np.append([0], open_pct, axis=0)
                    open_pct = open_pct * 100

                    high_ay = np.array(dataSig['high'])
                    diff_ay = np.diff(high_ay)
                    high_pct = np.divide(diff_ay, high_ay[:-1], out=np.zeros_like(diff_ay), where=high_ay[:-1]!=0)
                    high_pct = np.append([0], high_pct, axis=0)
                    high_pct = high_pct * 100

                    low_ay = np.array(dataSig['low'])
                    diff_ay = np.diff(low_ay)
                    low_pct = np.divide(diff_ay, low_ay[:-1], out=np.zeros_like(diff_ay), where=low_ay[:-1]!=0)                  
                    low_pct = np.append([0], low_pct, axis=0)
                    low_pct = low_pct * 100

                    close_ay = np.array(dataSig['close'])
                    diff_ay = np.diff(close_ay)
                    close_pct = np.divide(diff_ay, close_ay[:-1], out=np.zeros_like(diff_ay), where=close_ay[:-1]!=0)
                    close_pct = np.append([0], close_pct, axis=0)
                    close_pct = close_pct * 100

                    volume_ay = np.array(dataSig['volume'])
                    diff_ay = np.diff(volume_ay)
                    volume_pct = np.divide(diff_ay, volume_ay[:-1], out=np.zeros_like(diff_ay), where=volume_ay[:-1]!=0)
                    volume_pct = np.append([0], volume_pct, axis=0)
                    volume_pct = volume_pct * 100
                    
                    # logarithm normalization
                    close_ay = np.array(dataSig['close'])
                    close_log = np.log10(close_ay[1:]) - np.log10(close_ay[:-1])
                    close_log = np.append([0], close_log, axis=0)
                    temp = {'CHANGEOPEN': open_pct, 'CHANGEHIGH': high_pct, 'CHANGELOW': low_pct, 'CHANGECLOSE': close_pct, 'CHANGEVOLUME': volume_pct, "CHANGELOGCLOSE": close_log}

                elif (sigIndicatorName in self.config.tech_indicator_talib_lst) or (sigIndicatorName in self.config.otherRef_indicator_lst):
                    indNameLst = sigIndicatorName.split('-')
                    if len(indNameLst) == 1:
                        iname = sigIndicatorName
                        window_size = None
                        indFunc = abstract.Function(iname)
                        ifield = None
                    elif len(indNameLst) == 2:
                        iname = indNameLst[0]
                        window_size = int(indNameLst[1])
                        indFunc = abstract.Function(iname)
                        ifield = None
                    elif len(indNameLst) == 3:
                        iname = indNameLst[0]
                        window_size = int(indNameLst[1])
                        indFunc = abstract.Function(iname)
                        ifield = indNameLst[2]
                        
                    else:
                        raise ValueError("Unexpect indicator {}".format(sigIndicatorName))
                    
                    input_ay = []
                    if ifield is None:
                        if 'price' in indFunc.input_names.keys():
                            col = indFunc.input_names['price']
                            sig_ay = np.transpose(np.array(dataSig[col]))
                            if len(np.shape(sig_ay)) == 1:
                                sig_ay = [sig_ay]
                            input_ay = [*input_ay, *sig_ay]
                            
                        if 'prices' in indFunc.input_names.keys():
                            cols = indFunc.input_names['prices']
                            mul_ay = np.transpose(np.array(dataSig[cols]))
                            input_ay = [*input_ay, *mul_ay]

                    else:
                        sig_ay = np.transpose(np.array(dataSig[ifield]))
                        if len(np.shape(sig_ay)) == 1:
                            sig_ay = [sig_ay]
                        input_ay = [*input_ay, *sig_ay]               
                    
                    if window_size is None:
                        ind_val = indFunc(*input_ay)
                    else:
                        ind_val = indFunc(*input_ay, window_size)

                    if len(np.shape(ind_val)) == 1:
                        temp = {sigIndicatorName: ind_val}
                    else:
                        if sigIndicatorName == 'MACD':
                            temp = {sigIndicatorName: ind_val[0]}
                        elif sigIndicatorName == 'AROON':
                            temp = {'AROONDOWN': ind_val[0], 'AROONUP': ind_val[1]}
                        elif sigIndicatorName == 'BBANDS':
                            temp = {'BOLLUP': ind_val[0], 'BOLLMID': ind_val[1], 'BOLLLOW': ind_val[2]}
                        else:
                            raise ValueError("Please specify the features of indicator {}..".format(sigIndicatorName))
                else:
                    raise ValueError("Please specify the category of the indicator: {}".format(sigIndicatorName))               
                temp = pd.DataFrame(temp)
                temp['stock'] = sigStockName
                temp['date'] = np.array(dataSig['date'])
                ind_df = pd.concat([ind_df, temp], axis=0, join='outer')
            datax = pd.merge(datax, ind_df, how='outer', on=['stock', 'date'])
        
        datax.sort_values(['stock', 'date'], ascending=True, inplace=True)
        datax.reset_index(drop=True, inplace=True)
        cur_cols =list(datax.columns)
        self.techIndicatorLst = sorted(list(set(cur_cols) - set(self.rawColLst) - set(self.config.otherRef_indicator_lst)))
        return datax

    def scale_feat(self, data):
        data['date'] = pd.to_datetime(data['date'])
        datax = copy.deepcopy(data)

        # covariance calculation
        datax.sort_values(['date', 'stock'], ascending=True, inplace=True)
        datax.reset_index(drop=True, inplace=True)
        datax.index = datax.date.factorize()[0]
        cov_lst = []
        date_lst = []
        for idx in range(self.config.cov_lookback, datax['date'].nunique()):
            sigPeriodData = datax.loc[idx-self.config.cov_lookback:idx, :]
            sigPeriodClose = sigPeriodData.pivot_table(index = 'date',columns = 'stock', values = 'close')
            sigPeriodClose.sort_values(['date'], ascending=True, inplace=True)
            sigPeriodReturn = sigPeriodClose.pct_change().dropna()
            covs = sigPeriodReturn.cov().values 
            cov_lst.append(covs)
            date_lst.append(datax.loc[idx, 'date'].values[0])
        
        cov_pd = pd.DataFrame({'date': date_lst, 'cov': cov_lst})
        datax = pd.merge(datax, cov_pd, how='inner', on=['date'])

        # [t-T, t-T+1, .., t-1, t]
        if 'DAILYRETURNS-{}'.format(self.config.dailyRetun_lookback) in self.config.otherRef_indicator_lst:
            r_lst = []
            stockNo_lst = []
            date_lst = []
            datax.sort_values(['date', 'stock'], ascending=True, inplace=True)
            datax.reset_index(drop=True, inplace=True)
            datax.index = datax.date.factorize()[0]    
            for idx in range(self.config.dailyRetun_lookback, datax['date'].nunique()):
                sigPeriodData = datax.loc[idx-self.config.dailyRetun_lookback:idx, :][['date', 'stock', 'close']]
                sigPeriodClose = sigPeriodData.pivot_table(index = 'date',columns = 'stock', values = 'close')
                sigPeriodClose.sort_values(['date'], ascending=True, inplace=True)
                sigPeriodReturn = sigPeriodClose.pct_change().dropna() # without percentage
                sigPeriodReturn.sort_values(['date'], ascending=True, inplace=True)
                sigStockName_lst = np.array(sigPeriodReturn.columns)
                stockNo_lst = stockNo_lst + list(sigStockName_lst)
                r_lst = r_lst + list(np.transpose(sigPeriodReturn.values))
                date_lst = date_lst + [datax.loc[idx, 'date'].values[0]] * len(sigStockName_lst)
            r_pd = pd.DataFrame({'date': date_lst, 'stock': stockNo_lst, 'DAILYRETURNS-{}'.format(self.config.dailyRetun_lookback): r_lst})
            datax = pd.merge(datax, r_pd, how='inner', on=['date', 'stock'])

        datax.reset_index(drop=True, inplace=True)
        if self.config.test_date_end is None:
            if self.config.valid_date_end is None:
                data_date_end = self.config.train_date_end
            else:
                data_date_end = self.config.valid_date_end
        else:
            data_date_end = self.config.test_date_end
        datax = copy.deepcopy(datax[(datax['date'] >= self.config.train_date_start) & (datax['date'] <= data_date_end)])
        datax.sort_values(['date', 'stock'], ascending=True, inplace=True) 
        datax.reset_index(drop=True, inplace=True)

        for sigIndicatorName in self.techIndicatorLst:
            # Normalization
            nan_cnt = len(np.argwhere(np.isnan(np.array(datax[sigIndicatorName]))))
            inf_cnt = len(np.argwhere(np.isinf(np.array(datax[sigIndicatorName]))))
            if (nan_cnt > 0) or (inf_cnt > 0):
                raise ValueError("Indicator: {}, nan count: {}, inf count: {}".format(sigIndicatorName, nan_cnt, inf_cnt))
            
            if sigIndicatorName in ['CHANGELOGCLOSE', 'cov']:
                # No need to be normalized.
                continue
            
            train_ay = np.array(datax[(datax['date'] >= self.config.train_date_start) & (datax['date'] <= self.config.train_date_end)][sigIndicatorName])
            ind_mean = np.mean(train_ay)
            ind_std = np.std(train_ay, ddof=1)
            datax[sigIndicatorName] = (np.array(datax[sigIndicatorName]) - ind_mean) / ind_std
        self.techIndicatorLst = list(self.techIndicatorLst) + ['cov']
        cols_order = list(self.rawColLst) + list(self.config.otherRef_indicator_lst) + list(sorted(self.techIndicatorLst)) 
        datax = datax[cols_order]
        
        dataset_dict = {}
        train_data = copy.deepcopy(datax[(datax['date'] >= self.config.train_date_start) & (datax['date'] <= self.config.train_date_end)])
        train_data.sort_values(['date', 'stock'], ascending=True, inplace=True)
        train_data.reset_index(drop=True, inplace=True)
        dataset_dict['train'] = train_data
        
        if (self.config.valid_date_start is not None) and (self.config.valid_date_end is not None):
            valid_data = copy.deepcopy(datax[(datax['date'] >= self.config.valid_date_start) & (datax['date'] <= self.config.valid_date_end)])
            valid_data.sort_values(['date', 'stock'], ascending=True, inplace=True)
            valid_data.reset_index(drop=True, inplace=True)
            dataset_dict['valid'] = valid_data

        if (self.config.test_date_start is not None) and (self.config.test_date_end is not None):
            test_data = copy.deepcopy(datax[(datax['date'] >= self.config.test_date_start) & (datax['date'] <= self.config.test_date_end)])
            test_data.sort_values(['date', 'stock'], ascending=True, inplace=True)
            test_data.reset_index(drop=True, inplace=True)
            dataset_dict['test'] = test_data

        print(datax)
        # ['date', 'stock', 'open', 'high', 'low', 'close', 'volume'] + [{technical_indicators}]
        return dataset_dict
        