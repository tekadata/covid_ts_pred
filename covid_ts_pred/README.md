# Package: `covid_ts_pred`
- Document here the package: covid_ts_pred

Please document the package the better you can.

# Functions library

```
covid_ts_pred git:(master) ✗ tree
.
├── README.md
├── __init__.py
|
├── a_data
|   |
│   ├── country_data.py
│   ├── data.py
│   └── out_csv
│       ├── index_<country_name>.csv
│       ...
│       ├── indicator_<country_name>.csv
│       ...
│       └── world_population
│           └── world_population.csv
├── b_data_prep
│   ├── __init__.py
│   ├── ba_preproc
│   │   ├── __init__.py
│   ├── bb_feat_eng
│   │   └── __init__.py
│   ├── bc_feat_scal
│   │   └── __init__.py
│   ├── indicator.py
│   ├── model_Spain.pkl
│   ├── prediction.py
│   └── visualization.py
├── c_model
│   ├── __init__.py
│   ├── ca_models
│   │   ├── __init__.py
│   │   ├── best_models
│   │   │   └── best_1
│   │   ├── model_Brazil.pkl
│   │   ├── model_France.pkl
│   │   └── model_Mexico.pkl
│   ├── cb_linear_reg
│   │   └── SVR
│   │       ├── SVN_model.py
│   │       └── grid_search_SVR.py
│   ├── cc_mach_learn
│   │   ├── __init__.py
│   │   ├── arima
│   │   │   └── arima_model.py
│   │   └── sequencing.py
│   ├── cd_deep_learn
│   │   ├── RNN_model.py
│   │   └── __init__.py
│   ├── model.py
│   ├── model_index.py
│   ├── model_indicator.py
│   ├── prediction.py
│   ├── preprocessor.py
│   ├── preprocessing.py
├── d_eng
│   ├── __init__.py
│   ├── app-sumedha.py
│   ├── app.py
│   └── app_alb.py
└── z_utils
    ├── __init__.py
    └── project.py
```
## from covid_ts_pred.a_data.country_data import

### get_csv() -> list:
    function that returns the list of raw df loaded with the CSVs: `df_raw_list` (list).


### clean_data() -> list:
    function that returns the list of clean df: `df_clean_list` (list).

###  country_output(country) -> tuple:
    function that takes in params:
     - a country name: `country name` (string)
    and returns a tuple with:
     - an index df: `country_index` (pd.DataFrame),
     - an indicator df: `country_indicator` (pd.DataFrame).

### get_country_raw_data(country = 'Germany', is_index = False) -> pd.DataFrame:
    get_country_raw_data function get the raw_data either from index or indicator
    Drop column: 'Unamed:0' and
    Add column 'date' (pd.to_datetime)
    params:
    - country (string) country name
    - index_or_indicator = 'index' by default or could be 'indicator')
    return:
    -> pd.DataFrame

## from covid_ts_pred.a_data.country_data_2 import

### data_cleaning_all_index() -> pd.DataFrame:
    function that takes in params:
    - the table with the name of the data df: `name_data_table`(pd.DataFrame)
    and returns the transformed df of all cleaning index df: `trans_table` (pd.DataFrame).

###  data_cleaning_all_indicator(name_data_table) -> pd.DataFrame:
    function that takes in params:
    - the table with the name of the data df: `name_data_table`(pd.DataFrame)
    and returns the transformed df of all cleaning indicator df: `trans_table` (pd.DataFrame).

### country_output_2(country) -> tuple:
    function that takes in params:
     - a country name: `country name` (string)
    and returns a tuple with:
     - an index df: `country_index` (pd.DataFrame),
     - an indicator df: `country_indicator` (pd.DataFrame).

## from covid_ts_pred.a_data.data import

### function that take in parameter:
     - a root URL (string) to get the CSV data,
     - a list of CSV files,
     - a path (string) to store CSV in local,
     - a grid (list of list) to add in the CSV filename, URL, local path.
    and returns the gird updated with the CSVs of the list





# Install the package: `covid_ts_pred`
