from typing import Dict, Any
import pickle
import json

def save_params(params: Dict, path: str) -> None:
    """[Save parameters of the experiment.]

    Args:
        params (Dict): [parameters]
        path (str): [path to the file]
    """

    with open(path, 'w') as file:
        json.dump(params, file)


def load_params(path: str) -> Dict:
    """[Loads parameters of the experiment.]

    Args:
        path (str): [path to the file]

    Returns:
        Dict: [parameters]
    """
    with open(path) as file:
        return json.load(file)


def save_data(data: Any, path: str) -> None:
    """[Saves data]

    Args:
        data (Any)
        path (str)
    """

    with open(path, 'wb') as file:
        pickle.dump(data, file)


def load_data(path: str) -> Any:
    """[Load data]

    Args:
        path (str)

    Returns:
        Any: [data]
    """

    with open(path, 'rb') as file:
        return pickle.load(file)
