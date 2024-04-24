def verbosity(level):

    global McsHDF5_verbosity

    if level != 'verbose' and level != 'quiet':
        raise ValueError('Only verbose and quiet are allowed as verbosity levels!')

    McsHDF5_verbosity = level