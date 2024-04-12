
# https://doi.org/10.1016/j.neuroimage.2019.02.065 mag: 5 picoTesla , Grad: 400 pT

rejectCriteria = dict(
                     mag = 3000e-15,
                     grad = 3000e-13)

flatCriteria = dict(
                    mag=1e-15,
                    grad=1e-13)



# Define frequency bands
freqBands = {
    'Theta': (4, 8),
    'Alpha': (8, 13),
    'Beta': (13, 30),
    'Gamma': (30, 40),
    'Broadband': (3, 40)
}

# Define individualized frequency range over main peaks in each freq band
bandSubRanges = {
    'Theta': (-1, 1),
    'Alpha': (-2, 4),
    'Beta': (-5, 5),
    'Gamma': (-5, 5),
}

channelsOfInterest = []