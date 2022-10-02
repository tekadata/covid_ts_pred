from setuptools import find_packages
from setuptools import setup

with open('requirements.txt') as f:
    content = f.readlines()
requirements = [x.strip() for x in content if 'git+' not in x]

setup(name='covid_ts_pred',
      version="0.1",
      description="Deep Learning Time Series Prediction of Daily COVID-19 Cases \
          according to Government Responses",
      packages=find_packages(),
      install_requires=requirements,
      test_suite='tests',
      # include_package_data: to install data from MANIFEST.in
      include_package_data=True,
      scripts=['scripts/covid_ts_pred-run'],
      zip_safe=False)
