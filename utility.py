#encoding: utf-8
'''
utility.py
@description Provide Utility Functions
@author Kenichi Yorozu
@email rozken@gmail.com
@notice Source files on this repository is provided as-is and no guarantee
        or warranty is provided for any damage that my arise from using it.
        This code is free for your own use, the only thing I ask is small
        credit somewhere for my work. An e-mail saying you found it useful
        would also be much appreciated by myself.
'''

'''
@fn argDate
@brief find arg of the date
@param timestamp : numPy Array (DateTime) : find from this array
@param date_str : string "yyyy/mm/dd" : find this
@return index : location of date("yyyy/mm/dd") in timestamp array
'''
def argDate(timestamp, date_str):
    from datetime import datetime as dt
    import numpy as np
    date = dt.strptime(date_str, "%Y/%m/%d")
    index = np.where(timestamp == date)
    return index[0][0]