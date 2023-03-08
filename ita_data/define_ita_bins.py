# Copyright 2021-2022, The Johns Hopkins University Applied Physics Laboratory LLC
# All rights reserved.

import json
import pdb
from config import ITA_JSON

def create_bin_eval_func(low, high):
    def bool_func(v):
        if high is None:
            a = low
            return a < v
        elif low is None:
            b = high
            return v <= b
        else:
            a, b = low, high
            return a < v <= b

    return bool_func


def get_ita_data():
    """
    Gets the necessary ITA data from the JSON
    :return: dict containing ITA data from the JSON. The keys are each of the ITA bin abbreviations, with the associated
    values being dicts containing the full name of the bin (key is "Skin Tone"), the min and max values in the bin
    (key is "Cutoffs", stored as a list of [low, high]), and a function that takes an ITA value as an argument and
    checks if it belongs in the associated bin (key is 'check_in_bin')
    """
    with open(str(ITA_JSON)) as f:
        data_dict = json.load(f)

    for k in data_dict.keys():
        low, high = data_dict[k]["Cutoffs"]
        low = float(low) if low != None else None
        high = float(high) if high != None else None
        data_dict[k]['check_in_bin'] = create_bin_eval_func(low, high)

    return data_dict
