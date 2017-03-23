'''
analize.py
@description Main Procedure for Executing Principle Component Analysis on Time-series Data
@author Kenichi Yorozu
@email rozken@gmail.com
@notice Source files on this repository is provided as-is and no guarantee
        or warranty is provided for any damage that my arise from using it.
        This code is free for your own use, the only thing I ask is small
        credit somewhere for my work. An e-mail saying you found it useful
        would also be much appreciated by myself.
'''
from numpy import matrix

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
    
#   _, assets = data.shape
#    matrix = np.zeros((assets, assets))
    
    data = data[:-days]
    #print data[:, 0]
    #print data[:, [1,2]]
    matrix = np.corrcoef(data.T)

    return matrix