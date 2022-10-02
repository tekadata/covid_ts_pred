from os.path import dirname, isfile, join
from dotenv import load_dotenv

env_path = join(dirname(dirname(__file__)),'.env') # ../.env
print('project env_path', env_path)
load_dotenv(dotenv_path=env_path)

version_file = '{}/version.txt'.format(dirname(__file__))

if isfile(version_file):
    with open(version_file) as version_file:
        __version__ = version_file.read().strip()
