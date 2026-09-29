import numpy as np


def load_accident_data(filepath):
    """
    Load the raw accident dataset.

    Parameters
    ----------
    filepath : str
        Path to the accident CSV file.

    Returns
    -------
    numpy.ndarray
        Structured NumPy array containing the raw accident records.
    """

    data = np.genfromtxt(
        filepath,
        delimiter=",",
        dtype=None,
        encoding="utf-8",
        names=True
    )

    return data


