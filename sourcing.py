import pandas as pd
from covid_ts_pred.c_eng.engineering import *

def get_database_to_csv(url, csv_list, db_grid=[]) -> list:
    """
    function that take in parameter:
     - a root URL (string) to get the CSV data,
     - a list of CSV files,
     - a grid (list of list) to add in the CSV filename, URL, local path.
    and returns the gird updated with the CSVs of the list

    """

    ### Create a database grid (list of list) with all CSVs and associated URLs
    # print('db_grid', db_grid)
    #### Data project directory (if empty do not store CSV in local)
    # print('path', path)
    ### Website CSV datasets URL
    # print('url', url)
    #### List of CSVs of Website to retrieve
    # print('csv_list', csv_list)

    #### Length of grid aka number of CSVs already stored in grid
    len_grid = len(db_grid)

    for l in range(len(csv_list)):
        sub_list = []
        sub_list.append(csv_list[l]) ## 1st pos°: CSV filename
        sub_list.append(url + csv_list[l]) ## 2nd pos°: URL + CSV
        sub_list.append(get_raw_data_path(csv_list[l])) ## 3rd pos°: local data path + CSV
        cmd_list = ['curl',
                 '-o',
                 f'{get_raw_data_path(csv_list[l])}',
                 f'{url + csv_list[l]}']
        # print('cmd_list', cmd_list)
        err = execute_command(cmd_list
                              , debug=False)
        sub_list.append(cmd_list)
        # err = execute_command(cmd_s, debug=)
        # print('err', err)
        db_grid.append(sub_list)

    ### Return a database grid (list of list) with all CSVs and associated URLs
    return db_grid


def get_cleaned_data_from_raw_csv(refresh=True) -> list:
    """
    get_cleaned_data_from_raw_csv() -> list:
    function that returns the list of raw df loaded with the CSVs
    params:
        - refresh data from Oxford database: `refresh` (boolean), by default True
    returns:
        - list of raw df: `df_raw_list` (list).
    """
    if refresh == True:
        # Refresh all data from Oxford database
        ### Oxford Master data time series URL
        url_root_oxford = 'https://raw.githubusercontent.com/OxCGRT/covid-policy-tracker/master/data/timeseries/'

        #### List of CSVs of Oxford database Feel free to add more feature...
        ## 'E3;Fiscal measures;' missed!
        ## 'E4;International support;' missed!
        ## 'H3;Contact tracing;' missed!
        ## 'H4;Emergency investment in healthcare!
        ## 'H5;Investment in vaccines missed!
        ## 'V1;Vaccine Prioritisation missed!
        ## 'V2;Vaccine Availability missed!
        ## 'V3;Vaccine Financial Support!
        ## 'V4;Mandatory Vaccination missed!
        csv_list = ['confirmed_cases.csv', 'confirmed_deaths.csv',
                    'government_response_index_avg.csv', 'stringency_index_avg.csv',
                    'containment_health_index_avg.csv', 'economic_support_index.csv',
                    'c1m_school_closing.csv', 'c2m_workplace_closing.csv',
                    'c3m_cancel_public_events.csv', 'c4m_restrictions_on_gatherings.csv',
                    'c5m_close_public_transport.csv', 'c6m_stay_at_home_requirements.csv',
                    'c7m_movementrestrictions.csv', 'c8ev_internationaltravel.csv',
                    'e1_income_support.csv', 'e2_debtrelief.csv',
                    'h1_public_information_campaigns.csv', 'h2_testing_policy.csv',
                    'h3_contact_tracing.csv', 'h6m_facial_coverings.csv',
                    'h7_vaccination_policy.csv', 'h8m_protection_of_elderly_ppl.csv'
                ] ## ; print('csv_list', csv_list, 'len(csv_list)', len(csv_list))

        ### Vacinations Dataset URLs
        url_root_vaccinations = 'https://raw.githubusercontent.com/owid/covid-19-data/master/public/data/vaccinations/'

        #### List of CSVs of Vaccinations database
        csv_list_vax = ['vaccinations.csv', 'vaccinations-by-age-group.csv'] ## ; print('csv_list', csv_list_vax, 'len(csv_list)', len(csv_list_vax))

        ### Create a database grid all CSVs and associated URLs from Oxford website
        db_grid = get_database_to_csv(url_root_oxford, csv_list)
        ### Insert into database grid all CSVs and associated URLs from vaccinations website
        db_grid = get_database_to_csv(url_root_vaccinations, csv_list_vax, db_grid)
        # print('db_grid', db_grid)

        # Stack all csv in the list
        csv_list += csv_list_vax ## ; print('csv_list', csv_list)

        # transform list into dict:
        csv = dict(zip(csv_list, [v[0] for v in enumerate(csv_list)])) ## ; print ('csv', csv) ## if v[1] == 'containment_health_index_avg.csv'

        print('Global Data Refresh')

    # Load all data from local raw CSV database

    # DataFrame Index
    df_gov_response = clean_data(pd.read_csv(get_raw_data_path('government_response_index_avg.csv')), is_fillna=True)
    df_strigency = clean_data(pd.read_csv(get_raw_data_path('stringency_index_avg.csv')), is_fillna=True)
    df_health = clean_data(pd.read_csv(get_raw_data_path('containment_health_index_avg.csv')), is_fillna=True)
    df_economic = clean_data(pd.read_csv(get_raw_data_path('economic_support_index.csv')), is_fillna=True)
    print('Local Data Read')

    # DataFrames Indicator
    # C sub-indicators
    df_c1 = clean_data(pd.read_csv(get_raw_data_path('c1m_school_closing.csv')), is_fillna=True)
    df_c2 = clean_data(pd.read_csv(get_raw_data_path('c2m_workplace_closing.csv')), is_fillna=True)
    df_c3 = clean_data(pd.read_csv(get_raw_data_path('c3m_cancel_public_events.csv')), is_fillna=True)
    df_c4 = clean_data(pd.read_csv(get_raw_data_path('c4m_restrictions_on_gatherings.csv')), is_fillna=True)
    df_c5 = clean_data(pd.read_csv(get_raw_data_path('c5m_close_public_transport.csv')), is_fillna=True)
    df_c6 = clean_data(pd.read_csv(get_raw_data_path('c6m_stay_at_home_requirements.csv')), is_fillna=True)
    df_c7 = clean_data(pd.read_csv(get_raw_data_path('c7m_movementrestrictions.csv')), is_fillna=True)
    df_c8 = clean_data(pd.read_csv(get_raw_data_path('c8ev_internationaltravel.csv')), is_fillna=True)
     # E sub-indicators
    df_e1 = clean_data(pd.read_csv(get_raw_data_path('e1_income_support.csv')), is_fillna=True)
    df_e2 = clean_data(pd.read_csv(get_raw_data_path('e2_debtrelief.csv')), is_fillna=True)
    # H sub-indicators
    df_h1 = clean_data(pd.read_csv(get_raw_data_path('h1_public_information_campaigns.csv')), is_fillna=True)
    df_h2 = clean_data(pd.read_csv(get_raw_data_path('h2_testing_policy.csv')), is_fillna=True)
    df_h3 = clean_data(pd.read_csv(get_raw_data_path('h3_contact_tracing.csv')), is_fillna=True)
    df_h6 = clean_data(pd.read_csv(get_raw_data_path('h6m_facial_coverings.csv')), is_fillna=True)
    df_h7 = clean_data(pd.read_csv(get_raw_data_path('h7_vaccination_policy.csv')), is_fillna=True)
    df_h8 = clean_data(pd.read_csv(get_raw_data_path('h8m_protection_of_elderly_ppl.csv')), is_fillna=True)

    # DataFrame Vaccination
    df_vaccination = pd.read_csv(get_raw_data_path('vaccinations.csv'))
    df_vaccination = df_vaccination[['date','location','people_vaccinated_per_hundred', 'people_fully_vaccinated_per_hundred', 'total_boosters_per_hundred']]

    # Data Frame target
    df_cases = clean_data(pd.read_csv(get_raw_data_path('confirmed_cases.csv')), is_fillna=True)
    df_deaths = clean_data(pd.read_csv(get_raw_data_path('confirmed_deaths.csv')), is_fillna=True)

    # Crear lista con los df_raw

    df_cleaned_list = [df_gov_response, df_strigency, df_health,
                   df_economic, df_c1, df_c2, df_c3,
                   df_c4, df_c5, df_c6, df_c7, df_c8,
                   df_e1, df_e2, df_h1, df_h2, df_h3,
                   df_h6, df_h7, df_h8, df_vaccination,
                   df_cases, df_deaths]
    return df_cleaned_list


def get_country_data(self, country_name) -> pd.DataFrame:
    country_index, country_indicator = country_output(country=country_name)
    return country_index, country_indicator


def country_output(country, refresh=True) -> tuple:
    """
        country_output(country) -> tuple:
        function that takes in params:
        - a country name: `country name` (string),
        - refresh data from Oxford database: `refresh` (boolean), by default True,
        and returns a tuple with:
        - an index df: `country_index` (pd.DataFrame),
        - an indicator df: `country_indicator` (pd.DataFrame).
    """
     # Cleaning Index and Indicators
    df_gov_response, df_strigency, df_health,
    df_economic, df_c1, df_c2, df_c3,
    df_c4, df_c5, df_c6, df_c7, df_c8,
    df_e1, df_e2, df_h1, df_h2, df_h3,
    df_h6, df_h7, df_h8, df_vaccination,
    df_cases, df_deaths = get_cleaned_data_from_raw_csv(refresh=refresh)


    df_clean_list = [df_gov_response, df_strigency, df_health,df_economic,
               df_c1,df_c2,df_c3,df_c4,df_c5,df_c6,df_c7,df_c8,df_e1,df_e2,
               df_h1,df_h2,df_h3,df_h6,df_h7,df_h8,
               df_vaccination,
               df_cases,df_deaths]

    # INDEX FEATURES
    country_index = df_gov_response.copy()
    country_index = pd.DataFrame(country_index[[country]].iloc[:,0])
    country_index.index.name = country
    country_index.columns = ['gov_response']
    country_index['containment_and_health'] = df_health[[country]].iloc[:,0]
    country_index['stringency'] = df_strigency[[country]].iloc[:,0]
    country_index['economics_sup'] = df_economic[[country]].iloc[:,0]

    # INDICATOR FEATURES
    df = pd.DataFrame(df_c1[[country]].rename(columns = {country:'school_closing'}).iloc[:,0])
    df.index.name = country
    df['workplace_closing'] = df_c2[[country]].iloc[:,0]
    df['cancel_public_events'] = df_c3[[country]].iloc[:,0]
    df['restrictions_on_gatherings'] = df_c4[[country]].iloc[:,0]
    df['close_public_transport'] = df_c5[[country]].iloc[:,0]
    df['stay_at_home_requirements'] = df_c6[[country]].iloc[:,0]
    df['restrictions_on_internal_movement'] = df_c7[[country]].iloc[:,0]
    df['international_travel_controls'] = df_c8[[country]].iloc[:,0]
    df['income_support'] = df_e1[[country]].iloc[:,0]
    df['debt/contract_relief'] = df_e2[[country]].iloc[:,0]
    df['public_information_campaigns'] = df_h1[[country]].iloc[:,0]
    df['testing_policy'] = df_h2[[country]].iloc[:,0]
    df['contact_tracing'] = df_h3[[country]].iloc[:,0]
    df['facial_coverings'] = df_h6[[country]].iloc[:,0]
    df['vaccination_policy'] = df_h7[[country]].iloc[:,0]
    df['protection_of_elderly_people'] = df_h8[[country]].iloc[:,0]
    df = df.fillna(method = 'ffill')
    country_indicator = df

    # POPULATION VACCINATED
    country_vaccination = df_vaccination.loc[df_vaccination['location']==country]
    country_vaccination = country_vaccination.fillna(method='ffill').drop(columns = 'location')
    country_vaccination.index.name = country
    country_vaccination = country_vaccination.fillna(0)

    # TARGET
    country_target = df_cases.copy()
    country_target = pd.DataFrame(country_target[[country]].iloc[:,0])
    country_target.index.name = country
    country_target.columns = ['total_cases']
    country_target['new_cases'] = country_target - country_target.shift(1)
    country_target['total_deaths'] = df_deaths[[country]].iloc[:,0]
    country_target['new_deaths'] = df_deaths[[country]].iloc[:,0] - df_deaths[[country]].iloc[:,0].shift(1)

    country_target['new_cases'].loc[country_target['new_cases'] < 0] = 0
    country_target['new_deaths'].loc[country_target['new_deaths'] < 0] = 0

    # DAYS NO UPDATED DATA
    counter = non_update(country_target)

    # LAST UPDATED DATA
    country_index = country_index[:-counter]
    country_indicator = country_indicator[:-counter]
    country_vaccination = country_vaccination[:-counter]
    country_target = country_target[:-counter]

    # JOIN INDEX-TARGET AND INDICATOR-TARGET
    country_index = country_index.join(country_target)

    country_indicator = country_indicator.join(country_target)

    # JOIN INDEX AND VACCINATION
    country_vaccination.reset_index(inplace=True)
    country_vaccination['date'] = pd.to_datetime(country_vaccination['date'])

    country_index.reset_index(inplace=True)
    country_index[country] = country_index[country].apply(lambda x: pd.to_datetime( x, format='%y%b%d', infer_datetime_format=True))

    country_index.rename(columns = {country: 'date'}, inplace = True)
    country_index = country_index.merge(country_vaccination, how = 'left' , on = 'date')

    country_index.fillna(method = 'ffill', inplace=True)
    country_index.fillna(0, inplace=True)
    country_index.drop(columns = country, inplace=True)

    # AVERAGE CASES AND DEATHS 7 DAYS FOR INDEX
    country_index['new_deaths'] = country_index['new_deaths'].rolling(window=7).mean().fillna(0)
    country_index['new_cases'] = country_index['new_cases'].rolling(window=7).mean().fillna(0)

    # JOIN INDICATOR AND VACCINATION
    country_indicator.reset_index(inplace=True)
    country_indicator[country] = country_indicator[country].apply(lambda x: pd.to_datetime( x, format='%y%b%d', infer_datetime_format=True))

    country_indicator.rename(columns = {country: 'date'}, inplace = True)
    country_indicator = country_indicator.merge(country_vaccination, how = 'left' , on = 'date')

    country_indicator.fillna(method = 'ffill', inplace=True)
    country_indicator.fillna(0, inplace=True)
    country_indicator.drop(columns = country, inplace=True)

    # AVERAGE CASES AND DEATHS 7 DAYS FOR INDICATOR
    country_indicator['new_deaths'] = country_indicator['new_deaths'].rolling(window=7).mean().fillna(0)
    country_indicator['new_cases'] = country_indicator['new_cases'].rolling(window=7).mean().fillna(0)


    # START THE SERIES WITH THE FIRST COVID CASE REPORT IT
    country_index = country_index.loc[country_index['total_cases'] > 0].reset_index(drop=True)
    country_indicator = country_indicator.loc[country_indicator['total_cases'] > 0].reset_index(drop=True)

    return country_index, country_indicator


def get_country_raw_data(country, is_index = False) -> pd.DataFrame:
    """
    get_country_raw_data function get the raw_data either from index or indicator
    Drop column: 'Unamed:0' and
    Add column 'date' (pd.to_datetime)
    params:
    - country (string) country name
    - index_or_indicator = 'index' by default or could be 'indicator')
    return:
    -> pd.DataFrame
    """
    path = ''
    if is_index == True:
        path = f'index/'
    path += f"data_{country}.csv"
    print(path)
    country_data = pd.read_csv(get_raw_data_path(path),index_col=False)

    country_data.drop(columns = 'Unnamed: 0', inplace=True)

    country_data['date']=pd.to_datetime(country_data['date'])

    return country_data


def non_update(country_target) -> int:
    """
    non_update(country_target) -> int
    function that returns the number of days with no updated date
    params:
    - country target df: `country_target`(pd.DataFrame),
    returns:
    - nb of days with no updated data: `counter` (int).
    """
    counter = 0
    x = 1
    while country_target['total_deaths'][-x] == 0:
        counter += 1
        x += 1
    return counter


def clean_data(df, is_fillna=False) -> pd.DataFrame:
    """
        clean_data(df) -> pd.DataFrame:
        function that returns the cleaned df
        params:
        - `df` (pd.DataFrame),
        return:
        - cleaned df: `df` (pd.DataFrame).
    """
    drop_columns = ['Unnamed: 0','country_code','region_code','region_name','jurisdiction']

    df = df.drop(columns = drop_columns)
    df.set_index(keys='country_name', inplace=True)
    df = df.T
    if is_fillna == True:
        df = df.fillna(0)

    return


def data_cleaning_all_index(name_data_table) -> pd.DataFrame:
    """
        data_cleaning_all_index() -> pd.DataFrame:
        function that takes in params:
        - the table with the name of the data df: `name_data_table`(pd.DataFrame)
        and returns the transformed df of all cleaning index df: `trans_table` (pd.DataFrame).
    """
    trans_table=name_data_table.groupby('country_code').sum().T.drop('Unnamed: 0')
    trans_table.index = pd.to_datetime(trans_table.index)

    return trans_table


def data_cleaning_all_indicator(name_data_table) -> pd.DataFrame:
    """
        data_cleaning_all_indicator(name_data_table) -> pd.DataFrame:
        function that takes in params:
        - the table with the name of the data df: `name_data_table`(pd.DataFrame)
        and returns the transformed df of all cleaning indicator df: `trans_table` (pd.DataFrame).
    """
    trans_table=name_data_table.groupby('country_code').mean().round(decimals = 0).T.drop('Unnamed: 0')
    trans_table.index = pd.to_datetime( trans_table.index)

    return trans_table


def generate_country_code(country):
    code=df_cases_raw[df_cases_raw['country_name']==country]['country_code']
    code=code.iloc[0]
    return code

generate_country_code('France')


# def country_output_2(country) -> tuple:
# """
#     country_output_2(country) -> tuple:
#     function that takes in params:
#      - a country name: `country name` (string)
#     and returns a tuple with:
#      - an index df: `country_index` (pd.DataFrame),
#      - an indicator df: `country_indicator` (pd.DataFrame).
# """
#     df_gov_response_usa=df_gov_response[country]
#     country_index=df_gov_response_usa
#     country_index=pd.DataFrame(country_index)
#     country_index.columns = ['gov_response']
#     country_index.insert(0, 'containment_and_health', df_health[country])
#     country_index.insert(1, 'stringency', df_strigency[country])
#     country_index.insert(2,'economics_sup',df_economic[country])
#     country_index.insert(3,'total_cases',df_cases[country])
#     country_index.insert(4,'new_cases',df_cases[country]-df_cases[country].shift(1))
#     country_index.insert(5,'total_deaths',df_deaths[country])
#     country_index.insert(6,'new_deaths',df_deaths[country] - df_deaths[country].shift(1))
#     country_index.index.name='date'
#     country_index['new_cases'].loc[country_index['new_cases'] < 0] = 0
#     country_index['new_deaths'].loc[country_index['new_deaths'] < 0] = 0
#     country_index['gov_response'] = (country_index['gov_response'] / country_index['gov_response'].sum()) * 100
#     country_index['containment_and_health'] = (country_index['containment_and_health'] / country_index['containment_and_health'].sum()) * 100
#     country_index['stringency'] = (country_index['stringency'] / country_index['stringency'].sum()) * 100
#     country_index['economics_sup'] = (country_index['economics_sup'] / country_index['economics_sup'].sum()) * 100
#     country_index['economics_sup'] = (country_index['economics_sup'] / country_index['stringency'].sum()) * 100




#     #vaccination
#     country_vaccination=df_vaccination[df_vaccination['iso_code']==country]
#     country_vaccination=country_vaccination[['total_vaccinations', 'people_vaccinated','people_fully_vaccinated', 'total_boosters']]

#     #indicator
#     df_c2_usa=df_c2[country]
#     country_indicator= df_c2_usa
#     country_indicator=pd.DataFrame(country_indicator)
#     country_indicator.columns = ['workplace_closing']
#     country_indicator.insert(0, 'cancel_public_events', df_c3[country])
#     country_indicator.insert(1, 'school_closing', df_c1[country])
#     country_indicator.insert(2, 'restrictions_on_gathering', df_c4[country])
#     country_indicator.insert(3,'close_public_transport',df_c5[country])
#     country_indicator.insert(4,'stay_at_home_requirements',df_c6[country])
#     country_indicator.insert(5,'restrictions_on_internal_movement',df_c7[country])
#     country_indicator.insert(6,'international_travel_controls',df_c8[country])
#     country_indicator.insert(7,'income_support',df_e1[country])
#     country_indicator.insert(8,'debt/contract_relief',df_e2[country])
#     country_indicator.insert(9,'public_information_campaigns',df_h1[country])
#     country_indicator.insert(10,'testing_policy',df_h2[country])
#     country_indicator.insert(11,'contact_tracing',df_h3[country])
#     country_indicator.insert(12,'facial_coverings',df_h6[country])
#     country_indicator.insert(13,'vaccination_policy',df_h7[country])
#     country_indicator.insert(14,'protection_of_elderly_people',df_h8[country])
#     country_indicator.insert(15,'total_cases',df_cases[country])
#     country_indicator.insert(16,'new_cases',df_cases[country]-df_cases[country].shift(1))
#     country_indicator.insert(17,'total_deaths',df_deaths[country])
#     country_indicator.insert(18,'new_deaths',df_deaths[country] - df_deaths[country].shift(1))
#     country_indicator.index.name='date'
#     country_indicator['new_cases'].loc[country_indicator['new_cases'] < 0] = 0
#     country_indicator['new_deaths'].loc[country_indicator['new_deaths'] < 0] = 0


#     country_index = country_index.merge(country_vaccination, how = 'left' , on = 'date')
#     country_indicator=country_indicator.merge(country_vaccination, how = 'left' , on = 'date')


#     indicator_death=country_indicator[country_indicator['total_deaths']>0]
#     first_death_date=indicator_death.index[0]
#     last_death_date=indicator_death.index[-1]
#     country_indicator=country_indicator[~(country_indicator.index < first_death_date)]
#     country_indicator=country_indicator[~(country_indicator.index > last_death_date)]

#     index_death=country_index[country_index['total_deaths']>0]
#     first_death_date_index=index_death.index[0]
#     last_death_date_index=index_death.index[-1]
#     country_index=country_index[~(country_index.index < first_death_date_index)]
#     country_index=country_index[~(country_index.index > last_death_date_index)]

#     country_indicator = country_indicator.fillna(0)
#     country_index = country_index.fillna(0)

#     return country_index,country_indicator


# def main_country_data():
