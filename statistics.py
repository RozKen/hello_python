#encoding: utf-8
'''
statistics.py
@description Helper for Analyzing Time-series Data
@author Kenichi Yorozu
@email rozken@gmail.com
@notice Source files on this repository is provided as-is and no guarantee
        or warranty is provided for any damage that my arise from using it.
        This code is free for your own use, the only thing I ask is small
        credit somewhere for my work. An e-mail saying you found it useful
        would also be much appreciated by myself.
'''

'''
@fn summary
@brief calculate summary statistics
@param data : 2D NumPy array : vertical:date, horizontal:asset class, economic indicators, or etc.
@return summary(0:max, 1:min, 2:average, 3:25%tile, 4:Median, 5: 75%tile, 6:Variance, 7:Standard Deviation)
'''
def summary(data):
    import numpy as np
    from scipy import stats
    
    axis = 0
    
    _, assets = data.shape
    summary = np.zeros((8, assets))
    
    #MAX
    summary[0] = np.max(data, axis=axis)
    #MIN
    summary[1] = np.min(data, axis=axis)
    #Average
    summary[2] = np.average(data, axis=axis)
    #1-Quartile
    summary[3] = stats.scoreatpercentile(data, 25, axis=axis)
    #Median
    summary[4] = np.median(data, axis=axis)
    #3-Quartile
    summary[5] = stats.scoreatpercentile(data, 75, axis = axis)
    #Variance
    summary[6] = np.var(data, axis=axis)
    #Standard Deviation
    summary[7] = np.std(data, axis=axis)
    
    return summary

'''
@fn corMat
@brief calculate correlation coefficents matrix
@param data : 2D NumPy array : vertical:date, horizontal:asset class, economic indicators, or etc.
@param days : days used for calculation
@return matrix
'''
def corMat(data, days = 120):
    import numpy as np
       
    data = data[:-days]
    matrix = np.corrcoef(data.T)

    return matrix

'''
@fn z_scores
@brief calculate z-scores of series
@description z_scores = (data - mean) / standard deviation
@param data : 2D NumPy array : vertical:date, horizontal:asset class, economic indicators or  etc.
@return z_scores
'''
def z_scores(data):
    import numpy as np
    mean = np.mean(data, axis = 0)
    stdev = np.nanstd(data, axis = 0)
    z_scores = (data - mean) / stdev
    return z_scores
'''
@fn rolling_z
@brief calculate rolling z-scores of series
@description rolling_z compiles z_scores for each term (days)
@param data : 2D NumPy array : vertical:date, horizontal: asset class, ecoomic indicators or etc.
@param days : calculating term (default = 120)
@return rolling_z
'''
def rolling_z(data, days = 120):
    import numpy as np
    end, _ = data.shape
    rolling_z = np.array([])
    for term in range(days, end):
        if term == days:
            rolling_z = np.hstack((rolling_z,z_scores(data[term - days:term])[0]))
        else:
            rolling_z = np.vstack((rolling_z,z_scores(data[term - days:term])[0]))
    return rolling_z
