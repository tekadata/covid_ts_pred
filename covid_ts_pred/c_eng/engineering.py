import subprocess
import time
import tracemalloc
from  os.path import abspath, normpath, join, dirname
# you can check your pythonpath specifically
import os;
## To use this library inside the project covid_ts_pred uncomment:
# import covid_ts_pred.z_utils.project as uz
## Then call insight function with uz.get_<my_function>("dir1", "dir2")
## With optional parameter list of chained directories (use ".." to go to parent)

def get_py_path(*args) -> str:
    """
        get_py_path(*args) -> str:
        function to get the python path and concat *args as an abs path
        params:
        - a list of optional args: `*args` (list),
        return:
        -> python path plus args path (str)
    """
    if ':' in os.environ['PYTHONPATH']:
        py_path = os.environ['PYTHONPATH'][:[i for i, v in enumerate(os.environ['PYTHONPATH']) if os.environ['PYTHONPATH'][i] == ':'][0]]
        return get_abs_path(join(py_path, *args))
    else:
        return get_abs_path(join(os.environ['PYTHONPATH'], *args))


def get_abs_path(*args) -> str:
    """
        get_abs_path(*args) -> str:
        function to get the file path and concat from *args the absolute path
        params:
        - a list of optional args: `*args` (list),
        return:
        -> absolute path from file and *args (str)
    """
    return abspath(join(dirname(__file__), *args))


def get_csv_out_path(*args) -> str:
    """
        get_csv_out_path(*args) -> str:
        function to get the csv_out path
        params:
        - a list of optional args: `*args` (list),
        return:
        -> (str)
    """
    return get_abs_path(join(get_py_path(), "data_files", "out_csv", *args))


def get_raw_data_path(*args) -> str:
    """
        get_raw_data_path(*args) -> str:
        function to get the raw_data path
        params:
        - a list of optional args: `*args` (list),
        return:
        -> (str)
    """
    return abspath(get_py_path("data_files", "raw_data", *args))


def get_best_models_path(*args) -> str:
    """
        get_best_models_path(*args) -> str:
        function to get the best_models path
        params:
        - a list of optional args: `*args` (list),
        return:
        -> (str)
    """
    return abspath(get_py_path("covid_ts_pred", "b_model", "best_models", *args))


def simple_time_and_memory_tracker(method):

    # ### Log Level
    # 0: Nothing
    # 1: Print Time and Memory usage of functions
    LOG_LEVEL = 1

    def method_with_trackers(*args, **kw):
        ts = time.time()
        tracemalloc.start()
        result = method(*args, **kw)
        _, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        te = time.time()
        duration = te - ts
        if LOG_LEVEL > 0:
            output = f"{method.__qualname__} executed in {round(duration, 2)} seconds, using up to {round(peak / 1024**2,2)}MB of RAM"
            print(output)
        return result

    return method_with_trackers


def execute_command(cmd, debug = False):
   '''
   Excecute a command and return the stdiout and errors.
   cmd: list of the command. e.g.: ['ls', '-la']
   '''
   try:
      cmd_data = subprocess.Popen(cmd) #, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
      output,error = cmd_data.communicate()
      if debug:
         if error:
            print('Error:', error)
         if output:
            print('Output:', output)
      return output, error
   except:
      return 'Error in command:', cmd

# execute_command(['curl',
# '-k', '-H', '"Authorization: Bearer xxxxxxxxxxxxxxxx"', '-H', '"hawkular-tenant: test"', '-X', 'GET', 'https://www.example.com/test', '|', 'python', '-m', 'json.tool'])


## Memory Optimization
def compress(df, **kwargs):
    """
    Reduces size of dataframe by downcasting numerical columns
    """
    input_size = df.memory_usage(index=True).sum()/ 1024
    print("new dataframe size: ", round(input_size,2), 'kB')

    in_size = df.memory_usage(index=True).sum()
    for type in ["float", "integer"]:
        l_cols = list(df.select_dtypes(include=type))
        for col in l_cols:
            df[col] = pd.to_numeric(df[col], downcast=type)
    out_size = df.memory_usage(index=True).sum()
    ratio = (1 - round(out_size / in_size, 2)) * 100

    print("optimized size by {} %".format(round(ratio,2)))
    print("new dataframe size: ", round(out_size / 1024,2), " kB")

    return df

# compress(df) ; print('df.memory_usage', df.memory_usage, 'df.dtypes', df.dtypes)
