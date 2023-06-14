# RiPO: Combining Reinforcement Learning and Barrier Functions for Adaptive Risk Management in Portfolio Optimization

Reinforcement learning (RL) based investment strategies have been widely adopted in portfolio management (PM) in recent years. Nevertheless, most RL-based approaches may often emphasize on pursuing returns while ignoring the risks of the underlying trading strategies that may potentially lead to great losses especially under high market volatility. Therefore, a risk-manageable PM investment framework integrating both RL and barrier functions (BF) is proposed to carefully balance the needs for high returns and acceptable risk exposure in PM applications. Up to our understanding, this work represents the first attempt to combine BF and RL for financial applications. While the involved RL approach may aggressively search for more profitable trading strategies, the BF-based risk controller will continuously monitor the market states to dynamically adjust the investment portfolio as a controllable measure for avoiding potential losses particularly in downtrend markets. Additionally, two adaptive mechanisms are provided to dynamically adjust the impact of risk controllers such that the proposed framework can be flexibly adapted to uptrend and downtrend markets. The empirical results of our proposed framework clearly reveal such advantages against most well-known RL-based approaches on real-world data sets. More importantly, our proposed framework shed lights on many possible directions for future investigation.

This reportsitory is the official implementation of RiPO. The detailed introduction of RiPO has been released online: [RiPO Paper](https://arxiv.org/pdf/2306.07013.pdf).

## Requirements

Please run on Python 3.x, and install the libraries by running the command:
```
python -m pip install -r requirements.txt
```

## Entrance Script

Algorithms and trading settings can be configured in ```config.py```. After that, run the command to start training.
```
python entrance.py
```

## Acknowledgement
- Compared Algorithm Implementation: [PGPortfolio](https://github.com/ZhengyaoJiang/PGPortfolio/blob/48cc5a4af5edefd298e7801b95b0d4696f5175dd/pgportfolio/tdagent/tdagent.py#L7)
- Trading Environment: [FinRL](https://github.com/AI4Finance-Foundation/FinRL)
- TD3 Implementation: [Baselines3](https://stable-baselines3.readthedocs.io/en/master/modules/td3.html)
- Financial Indicator Implementation: [TA-Lib](https://github.com/TA-Lib/ta-lib-python)
- Second-order Cone Programming Solver: [CVXOPT](http://cvxopt.org/) 

We appreciate that they share their amazing works and implementations. This project would not have been finished without their works.

## Others
Should you have any questions, please do not hesitate to contact me: lzlong@hku.hk


Thanks!

RiPO
