# -*- coding: utf-8 -*-
"""
Created by: zhengmingsong
Created on: 2019-05-19 21:02
Utility functions
"""

import numpy as np

def tracking_error(r_a, r_b, period='d'):
    """

    Args:
        r_a: (n, 1) np array
        r_b: (n, 1) np array
        period: 'd'(default): daily, 'y': yearly, 'm': monthly

    Returns:
        scalar: tracking error

    """
    if period=='y':
        return np.sqrt(252)*np.std(r_a - r_b)
    elif period=='m':
        return np.sqrt(30)*np.std(r_a - r_b)
    else:
        return np.std(r_a - r_b)

def mse(r_a, r_b):
    """

    Args:
        r_a: (n, 1) np array
        r_b: (n, 1) np array

    Returns:
        mean squared error

    """
    return np.average((r_a - r_b)**2, axis=0)