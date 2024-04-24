def ExponentToUnit(e, o):
    """
    Converts an exponent to a unit scaling factor and a unit string.

    Parameters:
    e (int): The exponent of the largest absolute value of the data.
    o (int): The exponent after the data is scaled by the unit exponent.

    Returns:
    tuple: A tuple containing the scaling factor and the unit string.
    """

    poss_strings = ['p', 'n', 'µ', 'm', '', 'k', 'M', 'G']
    poss_exp = [-12, -9, -6, -3, 0, 3, 6, 9]

    # Finding the appropriate index
    i = next((idx for idx, val in enumerate(poss_exp) if val <= e), None)
    if i is None:
        i = 0

    # Calculating the scaling factor
    fact = 10 ** (o % 3 - o)

    # Getting the unit string
    unit_string = poss_strings[i]

    return fact, unit_string
