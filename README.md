# Simulation and Empirical Study Implementation with ArbitraryRectangle-range Generalized Elastic Net

This repo. contains python implementation for the paper [Variable Selection and regularization via ArbitraryRectangle-range Generalized Elastic Net]().

Authors:
- Yujia Ding* <yujia.ding@cgu.edu>
- Hansen Chen <hansenchen1@gmail.com>
- Qidi Peng <qidi.peng@cgu.edu>
- Zhengming Song <zhengming.song@cgu.edu>

## Folder Structure

```
.
├── README.md
├── index_tracking                          -> implementations for chapter 6
│   ├── __init__.py                                 
│   ├── docker                                   
│   │   └── stack.yml                       -> docker compose/swarm config file
│   ├── notebook                                
│   │   ├── get_sp500_data.ipynb            -> retrive s&p 500 constituents stock prices
│   │   └── result_analysis.ipynb           -> conduct analysis and generate experienments results
│   ├── sandp500                            -> data folder
│   │   ├── WIKI_metadata.csv
│   │   ├── __init__.py
│   │   ├── failed_queries.txt
│   │   ├── sp500_index.csv
│   │   └── sp500_pct.csv
│   └── scripts                             
│       ├── ARGEN.py                        -> solver class defination
│       ├── __init__.py                     
│       ├── analysis_utils.py               -> helper functions for emperical analysis
│       ├── data_utils.py                   -> helper functions for retriving data
│       └── hyperparameter_study.py         -> main script to conduct hyper parameter tunning
├── requirements.txt                        -> packges used in implementations
└── simulation                              -> implementations for chapter 5
    ├── __init__.py
    ├── arbitrary_signal.py
    ├── driver.py                           -> simulation utility class definition
    ├── simple_signal.py
    └── simulation_of_other_dataset.py      -> main script to generate 8 example results
```


