

rejectCriteria = dict(
                     mag = 3000e-15,
                     grad = 3000e-13)

flatCriteria = dict(
                    mag=1e-15,
                    grad=1e-13)



# Define frequency bands
freq_bands = {      # individualized peak
    'theta': (4, 8), # (-1, +1)
    'alpha': (8, 13), # (-2, +2)
    'beta': (13, 30), # (-5, +5)
    'gamma': (30, 40), #(-5, +5)
}