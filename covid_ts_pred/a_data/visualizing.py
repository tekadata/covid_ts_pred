from covid_ts_pred.b_model.country_data import  country_output

import matplotlib.pyplot as plt

# Visualization Indexs

def index_plot(country):

    country_index = country_output(country, refresh=False)[0]

    x = country_index['date']
    img = plt.figure(figsize=(25,10))
    plt.plot(x, country_index[['gov_response']], color = 'r')
    plt.plot(x, country_index[['containment_and_health']], color = 'b')
    plt.plot(x, country_index[['stringency']], color = 'g')
    plt.plot(x, country_index[['economics_sup']], color = 'y')
    plt.title(country, fontsize=30)
    plt.legend(['Gov_response','Containment_and_health','Stringency','Economics_sup'], fontsize='xx-large')

    return

def indicators(country):

    return

def cases(country):

    country_index = country_output(country, refresh=False)[0]

    # Visualization new cases
    x = country_index['date']
    img = plt.figure(figsize=(25,10))
    plt.plot(x, country_index[['new_cases']])
    plt.title(f'New covid cases in {country}', fontsize=30)

    # Visualization total cases
    x = country_index['date']
    img = plt.figure(figsize=(25,10))
    plt.plot(x, country_index[['total_cases']])
    plt.title(f'Total covid cases in {country}', fontsize=30)

    return


def deaths(country):

    country_index = country_output(country, refresh=False)[0]

    # Visualization new deaths
    x = country_index['date']
    img = plt.figure(figsize=(25,10))
    plt.plot(x, country_index[['new_deaths']])
    plt.title(f'New covid deaths in {country}', fontsize=30)

    # Visualization total deaths
    x = country_index['date']
    img = plt.figure(figsize=(25,10))
    plt.plot(x, country_index[['total_deaths']])
    plt.title(f'Total covid deaths in {country}', fontsize=30)

    return

def vaccination(country):

    country_index = country_output(country, refresh=False)[0]

    # Visualization Vaccination
    x = country_index['date']
    img = plt.figure(figsize=(25,10))
    plt.plot(x, country_index[['people_vaccinated_per_hundred']], color = 'r')
    plt.plot(x, country_index[['people_fully_vaccinated_per_hundred']], color = 'b')
    plt.plot(x, country_index[['total_boosters_per_hundred']], color = 'g')
    plt.title(country, fontsize=30)
    plt.legend(['people_vaccinated_per_hundred','people_fully_vaccinated_hundred','total_boosters_per_hundred'], fontsize='xx-large');

    return
