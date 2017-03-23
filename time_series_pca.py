'''
time_series_pca.py
@description A simple principle component analysis methods for time-series data
@author Kenichi Yorozu
@email rozken@gmail.com
@notice Source files on this repository is provided as-is and no guarantee
        or warranty is provided for any damage that my arise from using it.
        This code is free for your own use, the only thing I ask is small
        credit somewhere for my work. An e-mail saying you found it useful
        would also be much appreciated by myself.
'''
from pandas.tslib import string_types

'''
@TODO
- review methods for +/- direction of each PCA index
- review level adjustment on composite index with regression
- eliminate Manual Inputs

***TEMPOLARY ADJUSTMENT***
#4437 days until raw_data acquired (2015/7/22) *** should be timestamp.size
#MANUAL INPUT [12] (column index of S&P500 for standardization of compound index)
#assume 'settings' is 1D Column : load_settings
historical_start = 3407 #days since raw_data acquired (2012 ~ )
'''

'''
@fn PCA
@brief Compose principal components with 120-day data
@param data : 2D NumPy array : vertical:date, horizontal:asset class, economic indicators, or etc.
@param normalize_days : integer : days to be used in normalization of PCA
@return np.dot(evecs.T, z_score.T).T : 2D NumPy array : Time series of principal components (default is 3 series)
@return evals : NumPy array : Eigen Values of covariance matrix used for analysis
@return evecs : 2D NumPy array : Eigen Vectors of covariance matrix used for analysis
@return z_score : 2D NumPy array : Normalized data of input "data"
'''
def PCA(data, normalize_days=120):
    import numpy as np
    from scipy import linalg as la
    
    print "===Analizing Principal Components==="
    m, n = data.shape
    print "original data for PCA: ", m, "-days x", n, "-assets matrix"
    # Calculate Z-score
    data_for_normalize = data[m - normalize_days:]  #extract data used for normalization
    m_n, n_n = data_for_normalize.shape
    print "- on normalization: ", m_n, "-days x", n_n, "-assets matrix is used."
    z_score =(data - data_for_normalize.mean(axis = 0)) / data_for_normalize.std(axis = 0)
    z_score_for_normalize = z_score[m - normalize_days:]   #extract z-score used for normalization
        
    # calculate the covariance matrix (day's direction is vertical: rowvar = 0, unbiased: bias = 0)
    covMat = np.cov(z_score_for_normalize, rowvar = 0, bias = 0)
    
    # calculate eigenvectors & eigenvalues of the covariance matrix
    evals, evecs = la.eigh(covMat)
    
    #normalize eval so that evals.sum() always be 1
    evalsum = evals.sum()
    evals = evals/evalsum
    
    # sort eigenvalue in decreasing order
    idx = np.argsort(evals)[::-1]
    evals = evals[idx]
    
    # sort eigenvectors according to same index
    evecs = evecs[:,idx]
    
    # carry out the transformation on the data using eigenvectors
    # and return the re-scaled data, eigenvalues, and eigenvectors
    return np.dot(evecs.T, z_score.T).T, evals, evecs, z_score

'''
@fn plot
@brief show and save plotted time-series data
@param data : 2D NumPy array : vertical:date, horizontal:asset class, economic indicators, or etc.
@param filename : filename for save png
@param folder_path : folder path for save png
@param IsSave : whether or not save graph as png (default : True)
@param IsShow : whether or not show graph on display (default: True)
@return none
'''
def plot(data, filename ="plot", folder_path="graphs/", IsSave = True, IsShow = True):
    from mpl_toolkits.mplot3d import Axes3D
    ### FOR MAC ###
    #import matplotlib
    #matplotlib.use('TkAgg')
    from matplotlib import pyplot as plt
    
    print "===Plotting==="
    
    _, n = data.shape
    clr1 = '#202682'
    fig = plt.figure()
    if n > 2:
        ax1 = fig.add_subplot(221)
        ax1.plot(data[:, 0], data[:, 1], '.', mfc=clr1, mec=clr1)
        ax1.set_xlabel('1st')
        ax1.set_ylabel('2nd')
        ax2 = fig.add_subplot(222)
        ax2.plot(data[:, 1], data[:, 2], '.', mfc=clr1, mec=clr1)
        ax2.set_xlabel('2nd')
        ax2.set_ylabel('3rd')
        ax3 = fig.add_subplot(223)
        ax3.plot(data[:, 0], data[:, 2], '.', mfc=clr1, mec=clr1)
        ax3.set_xlabel('1st')
        ax3.set_ylabel('3rd')
        ax4 = fig.add_subplot(224, projection = '3d')
        ax4.scatter3D(data[:, 0], data[:, 1], data[:, 2])
        ax4.set_xlabel('1st')
        ax4.set_ylabel('2nd')
        ax4.set_zlabel('3rd')
    elif n == 2:
        ax = fig.add_subplot(111)
        ax.plot(data[:, 0], data[:, 1], '.', mfc=clr1, mec=clr1)
        ax.set_xlabel('1st')
        ax.set_ylabel('2nd')
    else:
        ax = fig.add_subplot(111)
    
    if IsSave:
        #Save Plot Image
        plt.savefig(folder_path + filename + ".png", dpi=300)
    
    if IsShow:
        #Show Plot Image
        plt.show()
    
    #Free Memory
    plt.close()
    
'''
@fn line_graph
@brief show and save line graph
@param data : 2D NumPy array : vertical:date
@param timestamp : numPy Array (DateTime) : time stamp
@param filename : filename for save png
@param folder_path : folder path for save png
@param IsSave : whether or not save graph as png (default : True)
@param IsShow : whether or not show graph on display (default: True)
@return none
'''    
def line_graph(data, timestamp, filename = "composite index", folder_path="graphs/", IsSave = True, IsShow = True):
    ### FOR MAC ###
    #import matplotlib
    #matplotlib.use('TkAgg')
    from matplotlib import pyplot as plt
    import matplotlib.dates as mdates
    import datetime as dt

    print "===Drawing Line Graph==="
    
    fig = plt.figure()
    ax = fig.add_subplot(111)

    #convert timestamp data for x-axis
    dates = mdates.date2num(timestamp)
    
    if data.ndim > 1:
        series, _ = data.shape
        for i in range(0, series):
            ax.plot(dates, data[i])
    else:
        ax.plot(dates, data)

    #format labels
    ax.set_xlabel('Date')
    ax.set_ylabel(filename)
    
    #format ticks
    years = mdates.YearLocator()    #every year
    months = mdates.MonthLocator()  #every month
    yearsFmt = mdates.DateFormatter('%y')
    
    ax.xaxis.set_major_locator(years)
    ax.xaxis.set_major_formatter(yearsFmt)
    ax.xaxis.set_minor_locator(months)
    
    datemin = dt.date(timestamp.min().year, 1, 1)
    datemax = dt.date(timestamp.max().year + 1, 1, 1)
    ax.set_xlim(datemin, datemax)
    ax.set_ylim(-150, 150)
    
    #format the coordinates message box
    ax.format_xdata = mdates.DateFormatter('%Y/%m/%d')
    #ax.grid(True)
    
    if IsSave:
        #Save Plot Image
        plt.savefig(folder_path + filename + ".png", dpi=300)
    
    if IsShow:
        #Show Plot Image
        plt.show()
    
    #Free Memory
    plt.close()

'''
@fn heatmap
@brief show and save heatmap
@param data : 2D NumPy array : vertical:date
@param timestamp : numPy Array (DateTime) : time stamp
@param header : name of each columns
@param filename : filename for save png
@param folder_path : folder path for save png
@param IsSave : whether or not save graph as png (default : True)
@param IsShow : whether or not show graph on display (default: True)
@return none
'''
def heatmap(data, timestamp, header, filename = "heatmap", folder_path = "graphs/", IsSave = True, IsShow = True):
    import seaborn as sns
    import matplotlib.pyplot as plt
    import matplotlib.dates as mdates
    import pandas as pd
    import datetime as dt
    
    print "===Drawing Heatmap==="
    
    #convert timestamp data for x-axis
    dates = mdates.date2num(timestamp)
    
    ### TEMPOLARY ADJUSTMENT ###
    header = np.array(['EUR','JPY',2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32,33])
    
    #build up data for heatmap
    df = []
    m, n = data.shape   #assume data is 2-dimensional
    for i in range(m):
        for j in range(n):
            df.append({
                    'Date'      : dates[i],
                    'Header'    : header[j],
                    'Value'     : data[i, j]
                })

    df = pd.DataFrame(df)
    #df['Date'] = mdates.num2date(df['Date'])
    df['Date'] = df['Date'].astype(int)
    df['Header'] = df['Header'].astype(str)
    df['Value'] = df['Value'].astype(float)
    print df
    
    fig = plt.figure()
    ax = fig.add_subplot(111)
    
    xlabel = pd.DataFrame(mdates.num2date(df['Date'])).astype(str)
    interval = 30
    #for i in (len(xlabel.column)):
    #    if i % interval != 0:
    #        xlabel[i] = ''
    
    df_pivot = pd.pivot_table(data=df, values='Value', columns='Date', index='Header', aggfunc=np.mean)
    sns.heatmap(df_pivot, xticklabels=interval, vmin=-2.0, vmax = 2.0, square=False, ax = ax)
    #xticklabels = xlabel
    
    #format labels
    ax.set_xlabel('Date')
    ax.set_ylabel(filename)
    
    #format ticks
    years = mdates.YearLocator()    #every year
    months = mdates.MonthLocator()  #every month
    yearsFmt = mdates.DateFormatter('%y')
    monthsFmt = mdates.DateFormatter('%m')
    
    #ax.xaxis.set_major_locator(years)
    #ax.xaxis.set_major_formatter(yearsFmt)
    #ax.xaxis.set_minor_locator(months)
    #ax.xaxis.set_major_locator(months)
    #ax.xaxis.set_major_formatter(monthsFmt)
    
    datemin = dt.date(timestamp.min().year, 1, 1)
    datemax = dt.date(timestamp.max().year + 1, 1, 1)
    #ax.set_xlim(datemin, datemax)
    #ax.set_ylim(-150, 150)
    
    #format the coordinates message box
    fig.autofmt_xdate()
    #ax.format_xdata = mdates.DateFormatter('%Y/%m/%d')
    #ax.grid(True)
    
    if IsSave:
        #Save Plot Image
        plt.savefig(folder_path + filename + ".png", dpi=300)
    
    if IsShow:
        #Show Plot Image
        plt.show()
    
    #Free Memory
    plt.close()

'''
@fn loadcsv_no_header
@brief load csv data with no header
@param filename : path_to_the_csv_file.csv : file path to the csv file
@param methods : loading methods : 0-NumPy(loadtxt), 1-NumPy (genfromtxt), 2-Pandas(WARNING), 3-CSV
@return data : NumPy 2D Array : data
'''
def loadcsv_no_header(filename, methods = 0):
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
@fn loadcsv
@brief load csv data with a header
@param filename : path_to_the_csv_file.csv : file path to the csv file
@param header_rows : number of rows used for header in the csv file (default = 1)
@return header : NumPy Array (String) : headers
@return timestamp : numPy Array (DateTime) : time stamp
@return data : NumPy 2D Array (float64) : data
'''
def loadcsv(filename, header_rows = 1):
    import numpy as np  #necessary for @return data (NumPy 2D Array)
    import csv
    from datetime import datetime as dt
    
    print "===Loading Data from CSV Files==="
    
    header = np.array([])
    timestamp = np.array([])
    csv_data = np.array([])
    dataFlag = True
    count = 0
    
    #Read CSV Data
    with open(filename, 'rU') as csv_file:
        reader = csv.reader(csv_file, quoting=csv.QUOTE_ALL) #QUOTE_ALL is required for headers & date column
        for row in reader:
            if dataFlag: #Read Headers and First Data
                if count < header_rows: #Read Headers
                    if count == 0:
                        header = np.hstack((header, np.array(row)))
                    else:
                        header = np.vstack((header, np.array(row)))
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

    ### ***TEMPOLARY ADJUSTMENT ###
    # 4437 days until raw_data acquired (2015/7/22) *** should be timestamp.size
    # 2015/07/22 : 4436, 07/21 : 4435, 07/20 : 4434
    # 2015/06/16 : 4410
    #csv_data = csv_data[4855-4410:]
    #timestamp = timestamp[4855-4410:]

    #sort data according to time stamp
    if timestamp.size > 1:
        if timestamp[0] > timestamp[1]:
            timestamp = timestamp[::-1]
            csv_data = csv_data[::-1]
    return header, timestamp, csv_data

'''
@fn load_settings
@brief load csv data for setting
@param filename : path_to_the_csv_file.csv : file path to the csv file
@param header_rows : number of rows used for header in the csv file (default = 1)
@return settings : NumPy Array (float64) : settings
'''
def load_settings(filename, header_rows = 1):
    import numpy as np  #necessary for @return data (NumPy 2D Array)
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
            else:
                settings = np.hstack((settings, np.array(row)))
    
    #Remove 1st Column and Transform string to float64
    #TEMPOLARY ADJUSTMENT *** assume 'settings' is 1D Column
    settings = settings[1:].astype(np.float64)
    
    return settings

'''
Main procedure
'''
import numpy as np
#from datetime import datetime as dt

pca_dimensions = 3

normalization_days = 120
### TEMPOLARY ADJUSTMENT ### 
historical_start = 3407 #days since raw_data acquired (2012 ~ )

print "===Program Initiated==="

#Path to folders
output_folder = "data_output/"
input_folder = "data_input/"
#folder_path = "graphs/"        #default value in plotting functions

#Read Settings
settings = load_settings(input_folder + "settings.csv")

#Read Data
#raw_data = loadcsv_no_header("series.csv", 0)
header, timestamp, raw_data = loadcsv(input_folder + "series_raw.csv")

#Calculate and Save Historical PCA results
index_historical = np.array([])
eval_historical = np.array([])
evec_historical = np.array([])

for history in range(historical_start, timestamp.size):
    #Principal Component Analysis
    _, n = raw_data.shape
    data_pca, evals, evecs, z_score = PCA(raw_data[:history + 1], normalization_days)
    
    #Determining +/- direction of each PCA index
    signs = np.array([])
    for i in range(0, evecs[0].size):
        signs = np.hstack((signs, np.array(settings * evecs[:,i]).sum()))
        signs[i] = signs[i] / abs(signs[i])
    
    # apply signs to evecs, data_pca
    for i in range(0, evecs[0].size):
        evecs[:, i] = signs[i] * evecs[:, i]
        data_pca[:, i] = signs[i] * data_pca[:, i]
        #Translate as Z-score    ###TEMPOLARY###
        pca_i_average = np.average(data_pca[:, i])
        pca_i_stdev = np.std(data_pca[:, i])
        data_pca[:, i] = (data_pca[:, i] - pca_i_average) / pca_i_stdev * 100.0
    
    ###how to deal with new PCA series which is not consistent with historical series
    methods = 201
    
    ###Output Images for Debug
    IsDebug = False
    
    if (methods - methods % 1000) / 1000 == 0:
        #Calculate Weighted Average
        composite_index = np.zeros(data_pca[:,0].size)
        for i in range(0, pca_dimensions):
            composite_index = composite_index + data_pca[:, i] * evals[i]
    else:   #if (methods - methods % 1000) / 1000 == 1:
        ### create historical sereis by averaged evecs
        
        #Number of Days for averaging evecs
        n_days = 10
        count = 0
        
        temp_eval = evals
        temp_evecs = evals * evecs
        count = count + 1
        
        if n_days > 1:
            temp_evecs = temp_evecs + evals * evecs
            count = count + 1
            
        if history > historical_start + 1 and min(n_days, evec_historical.shape[2]) > 2:
            
            for i in range(2, min(n_days, evec_historical.shape[2])):
                temp_evecs = temp_evecs + evals * evecs
                count = count + 1
            
        temp_evecs = temp_evecs / n_days
        
        averaged_pca = np.dot(temp_evecs.T, z_score.T).T
        
        #Calculate Weighted Average
        composite_index = np.zeros(averaged_pca[:, 0].size)
        for i in range(0, pca_dimensions):
            composite_index = composite_index + averaged_pca[:, i]
        
    if (methods % 1000 - methods % 100) / 100 == 1:  #100, 101, 110, 111
        ###    set -100.0 on minimum S&P500 day / average of normalization_days around minimum S&P 500 days for historical
        ###    fit new PCA to historical series

        #acquire argmin of column #13 : S&P500 : 2009/3/9 : 676.53 pt
        ###TEMPOLARY ADJUSTMENT - MANUAL INPUT [12]
        argmin = np.argmin(raw_data[:,12])

        
        if history == historical_start:
            if (methods - 100 - methods % 10)/10 == 0:  #100, 101
                # - Set -100.0 for minimum-S&P500 day (2009/3/9)
                composite_denom = composite_index[argmin]
            else:   #110, 111
                # - Set -100.0 for average of normalization_days around minimum-S&P500 day
                composite_denom = np.average(composite_index[argmin - normalization_days/2: argmin + normalization_days/2])
            composite_index = composite_index / composite_denom * (-100.0)
        
        #fit composite_index to index_historical
        if history != historical_start:
            p = np.poly1d(np.polyfit(composite_index[:-1], index_historical, 1))
            if IsDebug:###DEBUG###
                old = composite_index[:-1] 
            composite_index = p(composite_index)
            
        ###DEBUG: output graphs on regression###
        if IsDebug:
            if history > historical_start and history % 100 == 0:
                temp = np.vstack((index_historical, old, composite_index[:-1]))
                line_graph(temp, timestamp[:history], "regression result " + str(history), IsShow = False)

    elif (methods % 1000 - methods % 100) / 100 == 2:
        ###    scale historical series so as to (max, min) or (average of normalization_days around max, min)== (100.0, -100.0)
        ###    scale new PCA so as to (max, min) == (100.0, -100.0)
        
        #Fit new PCA
        if (methods - 100 - methods % 10)/10 == 0:  #200, 201
            composite_max = np.max(composite_index)
            composite_min = np.min(composite_index)
        else:   #210, 211
            peak = np.argmax(composite_index)
            trough = np.argmin(composite_index)
            composite_max = np.average(composite_index[max(peak - normalization_days/2, 0) : min(peak + normalization_days/2, composite_index.size)])
            composite_min = np.average(composite_index[max(trough - normalization_days/2, 0) : min(trough + normalization_days/2, composite_index.size)])
        
        composite_peak2trough = composite_max - composite_min
        if IsDebug: ###DEBUG###
            old = composite_index
        
        composite_index = (composite_index - composite_min) / composite_peak2trough * 200.0 -100.0
                            
        ###DEBUG: output graphs on regression###
        if IsDebug:
            if history > historical_start:  # and history % 100 == 0:
                temp = np.vstack((index_historical, old[:-1], composite_index[:-1]))
                line_graph(temp, timestamp[:history], "regression result " + str(history), IsShow = False)

    #Stack new PCA series to historical index
    if history == historical_start:
        index_historical = np.hstack((index_historical, composite_index))
        eval_historical = np.hstack((eval_historical, evals))
        evec_historical = evecs

    else:
        if methods % 10 == 0:
            #add latest value of new PCA to historical series
            index_historical = np.hstack((index_historical, composite_index[-1]))
        else:   #methods % 10 == 1:
            #add latest move of new PCA to historical series
            index_historical = np.hstack((index_historical, index_historical[-1] + composite_index[-1] - composite_index[-2]))
        eval_historical = np.vstack((eval_historical, evals))
        evec_historical = np.dstack((evec_historical, evecs))

### TEMPOLARY ADJUSTMENT
heatmap(z_score[4600:], timestamp[4600:], header)

#timestamp: transform datetime to String for csv output
csv_timestamp = np.array([])
for i in range(0, timestamp.size):
    csv_timestamp = np.hstack((csv_timestamp, np.array(timestamp[i,0].strftime('%Y/%m/%d'))))

#principal component index numbers
index_numbers = np.array([1])
for i in range(2, evals.size+1):
    index_numbers = np.hstack((index_numbers, np.array([i])))

#Principal Components
csv_series_result = np.vstack((csv_timestamp, composite_index, data_pca.T)).T
csv_series_result = np.vstack((np.hstack((np.array(['Date', 'Compound Index']), index_numbers)), csv_series_result))
np.savetxt(output_folder + "series_results.csv", csv_series_result, delimiter=',', fmt='%s')

#Z-scores
csv_header = np.hstack((np.array(['Date']), header))
csv_z_score = np.vstack((csv_header, np.vstack((csv_timestamp, z_score.T)).T))
np.savetxt(output_folder + "z-score.csv", csv_z_score, delimiter=',', fmt='%s')

#PCA_result
csv_header[0] = 'Proportion of Variance'    #rename
csv_pca_result = np.vstack((np.hstack((np.array(['Principal Components']), index_numbers)), np.vstack((csv_header, np.vstack((evals.T, evecs)).T)).T))
np.savetxt(output_folder + "pca_result.csv", csv_pca_result, delimiter=',', fmt='%s')

#Historical Index
csv_historical_header = np.array(['Date', 'Global Financial Markets Monitoring System'])
csv_historical = np.vstack((csv_timestamp, index_historical)).T
csv_historical = np.vstack((csv_historical_header, csv_historical))
np.savetxt(output_folder + "historical_series.csv", csv_historical, delimiter=',', fmt='%s')

#Historical Eigen Values (Proportions)
csv_eval_historical_header = np.hstack((np.array(['Date']), index_numbers))
csv_eval_historical = np.vstack((csv_timestamp[historical_start:], eval_historical.T)).T
csv_eval_historical = np.vstack((csv_eval_historical_header, csv_eval_historical))
np.savetxt(output_folder + "historical_evals.csv", csv_eval_historical, delimiter=',', fmt='%s')

#Historical Eigen Vectors
csv_evec_historical_header = np.hstack((np.array(['Date']), header))
for i in range(0, pca_dimensions):
    csv_evec_historical = np.vstack((csv_timestamp[historical_start:], evec_historical[:, i, :])).T
    csv_evec_historical = np.vstack((csv_evec_historical_header, csv_evec_historical))
    np.savetxt(output_folder + "historical_evecs-" + str(i) + ".csv", csv_evec_historical, delimiter=',', fmt='%s')

#Plot PCA result
#plot(data_pca)

#Draw Composite Results
line_graph(composite_index, timestamp)

#Draw Historical Cumulative Result
line_graph(index_historical, timestamp, "historical gfms")

print "=== Program Ended ==="