from pathlib import Path
import pandas as pd

parameter_path = Path(__file__).parent.parent / "data" / "parameters.csv"

def load_parameters(role = None):
    df = pd.read_csv(parameter_path, skipinitialspace=True)
    if role is not None:
        df = df[df['role'] == role]
    df = df.set_index('name')
    params = {}
    for name in df.index:
        params[name] = tuple(df.loc[name, ["min", "likely", "max"]])
    return params