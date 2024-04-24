def ID2coordinates(id_, num_rows, num_cols):
    """
    Converts a linear index to 2D grid coordinates.

    Parameters:
    id_ (int): The linear index to be converted.
    num_rows (int): The number of rows in the grid.
    num_cols (int): The number of columns in the grid.

    Returns:
    tuple: A tuple of two integers representing the 2D grid coordinates.

    Raises:
    ValueError: If the calculated coordinates are out of the range of the grid.
    """

    id_ -= 1  # Adjust for 0-based indexing
    x = id_ // num_rows
    y = id_ % num_rows

    # Adjust for 0-based indexing
    coordinates = (x + 1, y + 1)

    # Check if coordinates are within the grid
    if coordinates[0] > num_rows or coordinates[1] > num_cols or coordinates[0] < 1 or coordinates[1] < 1:
        raise ValueError('Single sensor coordinates exceed total sensor size!')

    return coordinates
