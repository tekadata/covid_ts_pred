# covid_ts_pred package project

## Data analysis
- Document here the project: covid_time_series_prediction
- Description: Deep Learning Time Series Prediction of Daily COVID-19 Cases according to Government Responses
- Data Source: https://www.bsg.ox.ac.uk/research/research-projects/covid-19-government-response-tracker
- Type of analysis: Machine Learning ARIMA model + Deep Learning RNN model

Please document the project the better you can.

##  package project install & setup

```bash
mkdir ~/code/Teky-Teka && cd Teky-Teka
git clone git@github.com:Teky-Teka/covid_ts_pred.git
cd covid_ts_pred
```

### Create/Activate virtualenv

```bash
pyenv virtualenv 3.8.12 covid_tsp_env
pyenv local covid_tsp_env
```
restart a new terminal
```bash
pyenv virtualenvs
```
check `covid_ts_env` is activated: `* covid_ts_env`
if not activate `covid_ts_env`:
```bash
pyenv activate covid_ts_env
```

### install the minimal project dependencies
install the project packages (editable for hot-reloading)
```bash
pip install --upgrade pip; pip install -r requirements.txt
pip install -e covid_ts_pred
pip freeze | grep covid_ts_pred
```
check `covid_ts_env` version is in pip: `covid_ts_pred==0.1`

### Test:
```bash
make clean install test
```

### Create a new project on `gitlab.com/<YOUR-GITHUB-NAME>/covid_ts_pred`
### Populate it:

```bash
###   e.g. if group is "Teky-Teka" and project_name is "covid_ts_pred"
git remote add origin git@github.com:Teky-Teka/covid_ts_pred.git
git push -u origin master
git push -u origin --tags
```

Functionnal test with a script:

```bash
cd
mkdir tmp
cd tmp
covid_ts_pred-run
```

## Installation of the project

The clone setup (only for contributor x1).

Go to `https://github.com/Teky-Teka/covid_ts_pred` to see the project, manage issues,
setup you ssh public key, ...

Create a python3 virtualenv and activate it:

```bash
pyenv virtualenv covid_tsp_env
cd ~/code/Teky-Teka/covid_ts_pred
pyenv local covid_tsp_env
pip install --upgrade pip; pip install -r requirements.txt
pip freeze
```

Clone the project and install it:

```bash
git clone git@github.com:Teky-Teka/covid_ts_pred.git
cd covid_ts_pred
pip install -r requirements.txt
make clean install test                # install and test
```
Functionnal test with a script:

```bash
cd
mkdir tmp
cd tmp
covid_ts_pred-run
```

## Setup

### 1 - Project Structure
Go to your local `~/code/Teky-Teka/covid_ts_pred` folder.
Display the project structure.

```bash
. # covid_ts_pred project "package" (root/py_path)
├── MANIFEST.in
├── Makefile  # Make constructor
├── README.md # project documentation
├── __init__.py  # turns the root folder into a "package"
├── build
├── covid_ts_pred
│   ├── README.md # python package documentation
│   ├── __init__.py # turns the covid_ts_pred folder into a "package"
│   ├── a_data # Data functions python files
│   │   ├── __init__.py # turns the a_data folder into a "package"
│   │   ├── preprocessing.py
│   │   ├── sequencing.py
│   │   ├── sourcing.py
│   │   └── visualizing.py
│   ├── b_model # Model functions
│   │   ├── __init__.py # turns the b_model folder into a "package"
│   │   ├── best_models
│   │   │   ├── best_1
│   │   │   └── model_Spain.pkl
│   │   ├── dl_rnn # Deep Learning Recurrent Neural Network functions
│   │   │   ├── __init__.py # turns the dl_rnn folder into a "package"
│   │   │   └── dl_rnn_modeling.py
│   │   ├── ml_svm_svr  # Machine Learning Support Vectors Machine functions
│   │   │   ├── __init__.py # turns the ml_svm_svr folder into a "package"
│   │   │   ├── ml_svr_grid_searching.py
│   │   │   ├── ml_svr_modeling.py
│   │   │   └── models
│   │   │       ├── SVR_MAPE_10d_Brazil_model.pkl
│   │   │       ├── SVR_MAPE_10d_France_model.pkl
│   │   │       └── SVR_MAPE_10d_Mexico_model.pkl
│   │   ├── training.py
│   │   └── ts_arima
│   │       ├── __init__.py # turns the ts_arima folder into a "package"
│   │       └── ts_arima_modeling.py
│   └── c_eng
│       ├── __init__.py # turns the c_eng folder into a "package"
│       ├── app-sumedha.py
│       ├── app.py
│       ├── app_alb.py
│       └── engineering.py
├── data_files
│   ├── out_csv # data output
│   │   ├── index_Brazil.csv
│   │   ├── index_France.csv
│   │   ├── index_Mexico.csv
│   │   ├── indicator_Brazil.csv
│   │   ├── indicator_France.csv
│   │   ├── indicator_Mexico.csv
│   │   └── world_population
│   │       └── world_population.csv
│   └── raw_data # data source (git ignored)
│       ├── OxCGRT_timeseries_all copy.csv
│       ├── OxCGRT_timeseries_all.csv
│       ├── OxCGRT_timeseries_all.xlsx
│       ├── api_fetch_data_USA_2020-02-14.xlsx
│       ├── c1m_school_closing copy.csv
│       ├── c1m_school_closing.csv
│       ├── c2m_workplace_closing.csv
│       ├── c3m_cancel_public_events.csv
│       ├── c4m_restrictions_on_gatherings.csv
│       ├── c5m_close_public_transport.csv
│       ├── c6m_stay_at_home_requirements.csv
│       ├── c7m_movementrestrictions.csv
│       ├── c8ev_internationaltravel.csv
│       ├── cm5_close_public_transport.csv
│       ├── confirmed_cases.csv
│       ├── confirmed_deaths.csv
│       ├── confirmed_deaths_save.csv
│       ├── containment_health_index_avg.csv
│       ├── data_United States.csv
│       ├── e1_income_support.csv
│       ├── e2_debtrelief.csv
│       ├── economic_support_index.csv
│       ├── government_response_index_avg.csv
│       ├── h1_public_information_campaigns.csv
│       ├── h2_testing_policy.csv
│       ├── h3_contact_tracing.csv
│       ├── h6m_facial_coverings.csv
│       ├── h7_vaccination_policy.csv
│       ├── h8m_protection_of_elderly_ppl.csv
│       ├── index
│       │   ├── data_Brazil.csv
│       │   ├── data_France.csv
│       │   ├── data_Mexico.csv
│       ├── stringency_index_avg.csv
│       ├── vaccinations-by-age-group.csv
│       └── vaccinations.csv
├── notebooks # Data Analysis & Science notebooks
│   ├── README.md # notebooks documentation
│   ├── a_eda # Exploratory Data Analysis notebooks
│   │   ├── Data_analysis-multiple_data.ipynb
│   │   ├── EDA_TeKa.ipynb
│   │   ├── USA.ipynb
│   │   ├── visualization_alb.ipynb
│   │   └── visualization_to_save.ipynb
│   └── b_model # Models notebooks
│       ├── dl_rnn # Deep Learning Recurrent Neural Network notebooks
│       │   ├── EDA_RNN_TeKa.ipynb
│       │   ├── RNN_TeKa.ipynb
│       │   ├── optimizing_rnn_kim.ipynb
│       │   └── time_series_prediction_covid_usa.ipynb
│       ├── ml_svm_svr # Machine Learning Support Vectors Machine notebooks
│       │   ├── model--SVR-index.ipynb
│       │   ├── model-Brazil-SVR-index.ipynb
│       │   ├── model-France-SVR-index.ipynb
│       │   ├── model-France-pickle.ipynb
│       │   ├── model-India-SVR-index-.ipynb
│       │   ├── model-Italy-SVR-index.ipynb
│       │   ├── model-Mexico-SVR-index.ipynb
│       │   ├── model-Russia-SVR-index.ipynb
│       │   ├── model-Spain-SVR-index.ipynb
│       │   ├── model-UK-SVR-index.ipynb
│       │   └── model-UK-SVR.ipynb
│       └── ts_arima # Time Series AutoRegressive Integrated Moving Average notebooks
│           └── model-UK-Arima.ipynb
├── pytest.ini
├── requirements.txt # dependencies versioning to install
├── scripts
│   └── covid_vn_pred-run
├── setup.py # settings to install
└── tests
    ├── __init__.py
    └── training_outputs # data engineering outputs
```
### 2 - Edit the `PYTHONPATH`

Add `covid_ts_pred` path to your `PYTHONPATH`.

This will allow you to easily import modules defined in `covid_ts_pred` in your notebooks throughout the week.

Open your terminal and navigate to your home directory by running:

```bash
cd
```

Now you'll need to open your `.zshrc` file. As you might have noticed the file starts with a dot which means it's a hidden file. To be able to see this file in your terminal you'll need to run the command below, the flag `-a` will allow you to see hidden files:

```bash
ls -a
```

Next lets open the file using your text editor:

```bash
code .zshrc
```

Now in your terminal run:
```bash
cd ~/code/Teky-Teka/covid_ts_pred/ && echo "export PYTHONPATH=\"$(pwd):\$PYTHONPATH\""
```

👉 Copy the resulting output line from your terminal and paste it at the bottom of your ~/.zshrc file. Don't forget to save and restart all your terminal windows to take this change into account.



### 🔥 Check your setup

Go to your `covid_ts_pred` sub-folder (the one with the Python .py files) and run an `ipython` session:

```bash
cd ~/code/Teky-Teka/covid_ts_pred
ipython
```

Then type the following to check that the setup phase from the previous exercise worked:

```python
from indicator import Indicator
Indicator().ping()
# => pong
```

If you get something else than `pong`, ask teammates or raise a ticket to get some help from a TA. You might have a problem with the `$PYTHONPATH`.
