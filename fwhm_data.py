import numpy as np

def find_index(x, val):
    # finds the index that is closest to the given value in the array
    return (np.abs(x - val)).argmin()

def find_max(x, y, range):
    #finds the max y value in a given range of x values
    # get index range of the x_range
    i_low = find_index(x, range[0])
    i_high = find_index(x, range[1])
    # get maximum in index range
    y_max = max(y[i_low:i_high])
    # get index of y maximum
    i_max = np.where(y == y_max)[0][0]
    # get x maximum
    x_max = x[i_max]
    # return index, x and y of maximum
    return x_max, y_max

def line_interp(x, y, i, half_y):
    # https://stackoverflow.com/questions/49100778
    # linear interpolation if half maximum doesn't coincide with data points
    # it's just geometry (not too difficult)
    return x[i] + (x[i+1] - x[i]) * ((half_y - y[i]) / (y[i+1] - y[i]))

def half_max_only(x, y, max_y):
    # https://stackoverflow.com/questions/49100778
    # get half max
    half_y = max_y / 2
    # get array that displays if value is above or below half max
    signs = np.sign(np.add(y, - half_y))
    # get the zero crossings
    zero_crossings = (signs[0:-2] != signs[1:-1])
    zero_crossings_i = np.where(zero_crossings)[0]
    # might have multiple crossings, get the ones that are closest to the peak
    i_max = find_index(y, max_y)
    zero_crossings_ii = [zero_crossings_i[zero_crossings_i < i_max].max(), zero_crossings_i[zero_crossings_i > i_max].min()]
    # return the interpolated x values which have half max y as value
    return [line_interp(x, y, zero_crossings_ii[0], half_y), line_interp(x, y, zero_crossings_ii[1], half_y)]

def peak_fwhm(x, y, range):
    # finds the peak and the fwhm in a given range
    x_max, y_max = find_max(x, y, range)
    cross1, cross2 = half_max_only(x, y, y_max)
    fwhm = cross2 - cross1
    return x_max, y_max, fwhm
