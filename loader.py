#encoding: utf-8
'''
loader.py
@description Provide Methods for Reading Time-series CSV Data
@author Kenichi Yorozu
@email rozken@gmail.com
@notice Source files on this repository is provided as-is and no guarantee
        or warranty is provided for any damage that my arise from using it.
        This code is free for your own use, the only thing I ask is small
        credit somewhere for my work. An e-mail saying you found it useful
        would also be much appreciated by myself.
'''
from __future__ import unicode_literals

'''
@fn decode_array
@brief decode string array to unicode
@param string_array : 1D array of strings
@param codec : codec of the strings (default : cp932 : Japanese)
@return u_string_array : NumPy 1D Array : 1D NumPy Array of strings (unicode)
'''
def decode_array(string_array, codec='cp932'):
    import numpy as np
    #Convert Character Set to Unicode
    u_string_array = np.array([])
    for name in string_array:
        u_string_array = np.hstack((u_string_array, np.array(name.decode(codec))))
    return u_string_array

'''
@fn encode_array
@brief encode string array from unicode
@param u_string_array : 1D array of strings
@param codec : codec of the strings (default : cp932 : Japanese)
@return string_array : NumPy 1D Array : 1D NumPy Array of strings (unicode)
'''
def encode_array(u_string_array, codec='utf_8'):
    import numpy as np
    #Convert Character Set to Unicode
    string_array = np.array([])
    for name in u_string_array:
        string_array = np.hstack((string_array, np.array(name.encode(codec))))
    return string_array

'''
@fn csv_no_header
@brief load csv data with no header
@param filename : path_to_the_csv_file.csv : file path to the csv file
@param methods : loading methods : 0-NumPy(loadtxt), 1-NumPy (genfromtxt), 2-Pandas(WARNING), 3-CSV
@return data : NumPy 2D Array : data
'''
def csv_no_header(filename, methods = 0):
    import numpy as np  #necessary for @return data (NumPy 2D Array)
       
    print "===Loading Data from CSV file (with no headers)==="
       
    if methods == 0:
        #===Use Numpy (loadtxt)===
        csv_data = np.loadtxt(filename, delimiter=',', skiprows=0)
    elif methods == 1:
        #===Use Numpy (genfromtxt: substitute NaN to 0.0)===
        csv_data = np.genfromtxt(filename, delimiter=',', filling_values=0.0)
    elif methods == 2:
        #===Use Pandas=== WARNING: data is not translated from DataFrame to NumPy correctly so that PCA result could change.
        import pandas
        #read CSV with pandas -> convert pandas "dataframe" to numpy "array"
        csv_data = pandas.read_csv(filename, header=None, encoding='utf-8').fillna(0.0).as_matrix() #encoding='shift_jis' or 'utf-8'
    elif methods == 3:
        #===Use CSV===
        import csv
        csv_data = np.array([])
        dataFlag = True
        with open(filename, 'rU') as csv_file:
            reader = csv.reader(csv_file, quoting=csv.QUOTE_NONNUMERIC)
            for row in reader:
                if dataFlag:
                    csv_data = np.hstack((csv_data, np.array(row)))
                    dataFlag = False
                else:
                    csv_data = np.vstack((csv_data, np.array(row)))
    else:
        print "invalid value for methods"
        exit()
        
    return csv_data

'''
@fn csv
@brief load csv data with a header
@param filename : path_to_the_csv_file.csv : file path to the csv file
@param header_rows : number of rows used for header in the csv file (default = 1)
@return header : NumPy Array (String) : headers
@return timestamp : numPy Array (DateTime) : time stamp
@return data : NumPy 2D Array (float64) : data
'''
def csv(filename, header_rows = 1):
    import numpy as np  #necessary for @return data (NumPy 2D Array)
    import csv
    import codecs
    import sys
    from datetime import datetime as dt
    
    print "===Loading Data from CSV Files==="
    
    header = np.array([])
    timestamp = np.array([])
    csv_data = np.array([])
    dataFlag = True
    count = 0
    
    #Read CSV Data
    with codecs.open(filename, 'rU') as csv_file:
        reader = csv.reader(csv_file, quoting=csv.QUOTE_ALL) #QUOTE_ALL is required for headers & date column
        for row in reader:
            if dataFlag: #Read Headers and First Data
                if count < header_rows: #Read Headers
                    temp = decode_array(row, 'cp932')
                    if count == 0:
                        header = np.hstack((header, np.array(temp)))
                    else:
                        header = np.vstack((header, np.array(temp)))
                    count += 1
                else: #Read First Data
                    csv_data = np.hstack((csv_data, np.array(row)))
                    dataFlag = False
            else: #Read following Data
                csv_data = np.vstack((csv_data, np.array(row)))

    #split Time stamp Column
    timestamp_s = csv_data[:,0]
    header = header[1:]
    csv_data = csv_data[:,1:].astype(np.float64) #transform string to float64

    ###transform string to datetime###
    timestamp = np.hstack((timestamp, np.array(dt.strptime(timestamp_s[0], '%Y/%m/%d'))))
    for i in range(1, timestamp_s.size):
        timestamp = np.vstack((timestamp, np.array(dt.strptime(timestamp_s[i], '%Y/%m/%d'))))

    #sort data according to time stamp
    if timestamp.size > 1:
        if timestamp[0] > timestamp[1]:
            timestamp = timestamp[::-1]
            csv_data = csv_data[::-1]

    return header, timestamp, csv_data

'''
@fn settings
@brief load csv data for setting
@description setting file contains directions (1 or -1) used for analysis
            for now, 
             row[0] : PCA / Correlation Matrix
             row[1] : Heatmap
@param filename : path_to_the_csv_file.csv : file path to the csv file
@param header_rows : number of rows used for header in the csv file (default = 1)
@return settings : NumPy Array (float64) : settings
'''
def settings(filename, header_rows = 1):
    import numpy as np
    import csv
    
    print "===Loading Setting==="
    settings = np.array([])
    count = 0
    
    #Read CSV Data
    with open(filename, 'rU') as csv_file:
        reader = csv.reader(csv_file, quoting=csv.QUOTE_ALL) #QUOTE_ALL is required for headers & date column
        for row in reader:
            if count < header_rows:
                count = count + 1
            elif count == header_rows:
                settings = np.hstack((settings, np.array(row)))
                count = count + 1
            else:
                settings = np.vstack((settings, np.array(row)))
    
    #Remove 1st Column and Transform string to float64
    settings = settings[:,1:].astype(np.float64)
    
    return settings
