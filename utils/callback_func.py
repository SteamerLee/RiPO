#！/usr/bin/python
# -*- coding: utf-8 -*-#
'''
---------------------------------
 Name: callback_func.py  
 Author: RiPO
--------------------------------
'''
import numpy as np
import os
import pandas as pd
import time
import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt
from stable_baselines3.common.callbacks import BaseCallback
from .model_pool import model_select
import sys
sys.path.append('..')
from RL_controller.controllers import RL_withoutController, RL_withController

class PoCallback(BaseCallback):

    def __init__(self, config, train_env, valid_env=None, test_env=None, verbose=0):
        super(PoCallback, self).__init__(verbose)
        self.train_env = train_env
        self.valid_env = valid_env
        self.test_env = test_env
        self.config = config
        if self.config.mode == 'RLonly': 
            self.risk_controller = RL_withoutController
        elif self.config.mode == 'RLcontroller':
            self.risk_controller = RL_withController
        else:
            raise ValueError("Unexpected mode [{}]..".format(self.config.mode))
    def _on_training_start(self) -> None:
        """
        This method is called before the first rollout starts.
        """
        pass

    def _on_rollout_start(self) -> None:
        """
        A rollout is the collection of environment interaction
        using the current policy.
        This event is triggered before collecting new samples.
        """
        pass

    def _on_step(self) -> bool:
        """
        This method will be called by the model after each call to `env.step()`.

        For child callback (of an `EventCallback`), this will be called
        when the event is triggered.

        :return: (bool) If the callback returns False, training is aborted early.
        """
        # Save model
        if self.train_env.model_save_flag:
            cur_ep = self.train_env.epoch - 1 # self.train_env.epoch is the epoch number after reset().
            env_type = 'train'
            fpath = os.path.join(self.config.res_dir, '{}_bestmodel.csv'.format(env_type))
            model_records = pd.DataFrame(pd.read_csv(fpath, header=0))
            if cur_ep == int(model_records['max_capital_ep'][0]):
                mpath = os.path.join(self.config.res_model_dir, '{}_max_capital'.format(env_type))
                self.model.save(mpath)

            mpath = os.path.join(self.config.res_model_dir, 'current_model')
            self.model.save(mpath)
            # Evaluate model in validation set and test set
            ModelCls = model_select(model_name=self.config.rl_model_name,  mode=self.config.mode)
            trained_model = ModelCls.load(mpath)
            if self.valid_env is not None:
                obs_valid = self.valid_env.reset()
                while True:
                    a_rlonly, _ = trained_model.predict(obs_valid)
                    a_rlonly = np.reshape(a_rlonly, (-1))
                    a_rl = a_rlonly
                    if np.sum(a_rl) == 0:
                        a_rl = np.array([1/len(a_rl)]*len(a_rl))
                    else:
                        a_rl = a_rl / np.sum(a_rl)
                    a_cbf  = self.risk_controller(a_rl=a_rl, env=self.valid_env)
                    a_final = a_rl + a_cbf
                    a_final = np.array([a_final])
                    obs_valid, rewards, terminal_flag, _ = self.valid_env.step(a_final) 
                    if terminal_flag:
                        break    
            
                cur_ep = self.valid_env.epoch # self.valid_env.epoch is the epoch number before reset().
                env_type = 'valid'
                fpath = os.path.join(self.config.res_dir, '{}_bestmodel.csv'.format(env_type))
                model_records = pd.DataFrame(pd.read_csv(fpath, header=0))
                if cur_ep == int(model_records['max_capital_ep'][0]):
                    mpath = os.path.join(self.config.res_model_dir, '{}_max_capital'.format(env_type))
                    trained_model.save(mpath)
                
            if self.test_env is not None:
                obs_test = self.test_env.reset()
                while True:
                    a_rlonly, _ = trained_model.predict(obs_test)
                    a_rlonly = np.reshape(a_rlonly, (-1))
                    a_rl = a_rlonly
                    if np.sum(a_rl) == 0:
                        a_rl = np.array([1/len(a_rl)]*len(a_rl))
                    else:
                        a_rl = a_rl / np.sum(a_rl)
                    a_cbf  = self.risk_controller(a_rl=a_rl, env=self.test_env)
                    a_final = a_rl + a_cbf
                    a_final = np.array([a_final])
                    obs_test, rewards, terminal_flag, _ = self.test_env.step(a_final) 
                    if terminal_flag:
                        break    
                
                cur_ep = self.test_env.epoch # self.test_env.epoch is the epoch number before reset().
                env_type = 'test'
                fpath = os.path.join(self.config.res_dir, '{}_bestmodel.csv'.format(env_type))
                model_records = pd.DataFrame(pd.read_csv(fpath, header=0))
                if cur_ep == int(model_records['max_capital_ep'][0]):
                    mpath = os.path.join(self.config.res_model_dir, '{}_max_capital'.format(env_type))
                    trained_model.save(mpath)

            del trained_model

            # Compared Graph
            # x-axis: Episode, y-axis: final_capital, rewards_sum, vol_max, volatility, sharpeRatio, mdd
            fpath = os.path.join(self.config.res_dir, 'train_profile.csv')
            trainHist = pd.DataFrame(pd.read_csv(fpath, header=0))
            trainHist.sort_values(['ep'], ascending=True, inplace=True)
            trainHist.reset_index(drop=True, inplace=True)

            if self.valid_env is not None:
                fpath = os.path.join(self.config.res_dir, 'valid_profile.csv')
                validHist = pd.DataFrame(pd.read_csv(fpath, header=0))
                validHist.sort_values(['ep'], ascending=True, inplace=True)
                validHist.reset_index(drop=True, inplace=True)

            if self.test_env is not None:
                fpath = os.path.join(self.config.res_dir, 'test_profile.csv')
                testHist = pd.DataFrame(pd.read_csv(fpath, header=0))
                testHist.sort_values(['ep'], ascending=True, inplace=True)
                testHist.reset_index(drop=True, inplace=True)

            dataLen = len(trainHist)
            x = np.arange(dataLen)
            
            target_dict = { 
                'final_capital': {'ylabel': 'Returns'},
                'reward_sum': {'ylabel': 'Rewards'},
            }             
            for k, v in target_dict.items():
                fig = plt.figure(figsize=(12, 6))
                ax1 = fig.add_subplot(111)
                lns0 = ax1.plot(x, np.array(trainHist[k]), 'r', label='Train')
                if (self.valid_env is not None) or (self.test_env is not None):
                    ax2 = ax1.twinx()
                if self.valid_env is not None:
                    lns1 = ax2.plot(x, np.array(validHist[k]), 'b', label='Valid')
                    lns0 = lns0 + lns1
                if self.test_env is not None:
                    lns2 = ax2.plot(x, np.array(testHist[k]), 'k', label='Test')
                    lns0 = lns0 + lns2
                labs = [lnx.get_label() for lnx in lns0] 
                ax1.legend(lns0, labs, loc=0)
                ax1.set_xlabel("Episode")  #, fontsize=fontsize+4)
                ax1.set_xlabel(v['ylabel']) #, fontsize=fontsize+4)
                ax2.set_xlabel(v['ylabel']) #, fontsize=fontsize+4)
                plt.title('Trend of {}'.format(v['ylabel']))
                plt.savefig(os.path.join(self.config.res_img_dir, 'Comp_{}.png'.format(v['ylabel'].replace(' ', '_'))), bbox_inches='tight') # dpi=dpi
                plt.close('all')
        self.train_env.model_save_flag = False
        return True

    def _on_rollout_end(self) -> None:
        """
        This event is triggered before updating the policy.
        """
        pass

    def _on_training_end(self) -> None:
        """
        This event is triggered before exiting the `learn()` method.
        """
        pass
