def checkParameter(cfg, fieldname, default):
    """
    Helper function to set default parameters if necessary.

    Checks if a key with name 'fieldname' exists in the dictionary 'cfg'. If not, 
    or if it is None, sets it to 'default', otherwise leaves it unchanged.
    Returns a tuple of the updated cfg dictionary and a boolean 'is_default' 
    which is True if the default settings have been set in cfg, otherwise False 
    if the key is unchanged.

    Parameters:
    cfg (dict): Configuration dictionary to check and update.
    fieldname (str): Key name to check in the cfg dictionary.
    default: Default value to set if the key is not present or its value is None.

    Returns:
    (dict, bool): Tuple containing the updated cfg dictionary and the is_default flag.
    """

    is_default = False

    if cfg is None:
        cfg = {}
        cfg[fieldname] = default
        is_default = True
    elif fieldname not in cfg or cfg[fieldname] is None:
        cfg[fieldname] = default
        is_default = True

    return cfg, is_default