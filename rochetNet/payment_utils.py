"""
Utility functions for payment analysis in visualization notebooks.
"""

import numpy as np


def get_payment(x, pay, v1_val, v2_val):
    """
    Get payment for specific v1 and v2 values.
    
    Parameters:
    -----------
    x : numpy.ndarray
        Grid array (e.g., np.linspace(0, v_max, D))
    pay : numpy.ndarray
        Payment array of shape (D, D)
    v1_val : float
        Desired v1 value
    v2_val : float
        Desired v2 value
    
    Returns:
    --------
    tuple : (idx1, idx2, actual_v1, actual_v2, payment_val)
        idx1, idx2: Indices in the grid
        actual_v1, actual_v2: Actual grid values closest to requested values
        payment_val: Payment value at the closest grid point
    """
    # Find the closest indices
    idx1 = np.argmin(np.abs(x - v1_val))
    idx2 = np.argmin(np.abs(x - v2_val))
    actual_v1 = x[idx1]
    actual_v2 = x[idx2]
    payment_val = pay[idx1, idx2]
    return idx1, idx2, actual_v1, actual_v2, payment_val

