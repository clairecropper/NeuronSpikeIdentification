def coordinates2ID(coordinates, num_rows, num_cols):
    """
    Converts 2D grid coordinates to a linear index.

    Parameters:
    coordinates (tuple): A tuple of two integers (row, column).
    num_rows (int): The number of rows in the grid.
    num_cols (int): The number of columns in the grid.

    Returns:
    int: The linear index corresponding to the coordinates.

    Raises:
    ValueError: If the calculated ID is out of the range of the grid.
    """

    row, col = coordinates
    id_ = num_rows * (row - 1) + col
    if id_ > (num_rows * num_cols):
        raise ValueError('ID out of range!')

    return id_
