def print(s):

    global McsHDF5_verbosity

    if McsHDF5_verbosity is None:
        McsHDF5_verbosity = 'quiet'

    if McsHDF5_verbosity != 'quiet':
        print(s)