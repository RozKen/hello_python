'''
time_series_pca.py
@description A simple principle component analysis methods for time-series data
@author Kenichi Yorozu
@email rozken@gmail.com
@notice Sourcefiles on this repository is provided as-is and no gurantee
        or warranty is provided for any damage that my arise from using it.
        This code is free for your own use, the only thing I ask is small
        credit somewhere for my work. An e-mail saying you found it useful
        would also be much appreciated by myself.
'''

"""
@TODO
- sum up output csv files

***TEMPOLARY ADJUSTMENT***
=== avoid subdivision with zero (stdev could be 0) ===
csv_data = csv_data[:1900]
timestamp = timestamp[:1900]
"""

'''
@fn PCA
@brief Compose principal components with 120-day data
@param data : 2D NumPy array : vertical:date, horizontal:asset class, economic indicators, or etc.
@param pca_dimension : integer : dimension of principal components
@param normalize_days : integer : days to be used in normalization of PCA
@return np.dot(evecs.T, z_score.T).T : 2D NumPy array : Time series of principal components (default is 3 series)
@return evals : NumPy array : Eigen Values of covariance matrix used for analysis
@return evecs : 2D NumPy array : Eigen Vectors of covariance matrix used for analysis
@return z_score : 2D NumPy array : Normalized data of input "data"
'''
def PCA(data, pca_dimension=3, normalize_days=120):
    import numpy as np
    from scipy import linalg as la
    
    print "===Principal Components Analysis==="
    m, n = data.shape
    print "original data for PCA: ", m, "-days x", n, "-assets matrix"
    # Calculate Z-score
    data_for_normalize = data[m - normalize_days:]  #extract data used for normalization
    m_n, n_n = data_for_normalize.shape
    print "- on normalization: ", m_n, "-days x", n_n, "-assets matrix is used."
    z_score =(data - data_for_normalize.mean(axis = 0)) / data_for_normalize.std(axis = 0)
    z_score_for_normalize = z_score[m - normalize_days:]    #extract z-score used for normalization
        
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
    
    #select the first n eigenvalues & eigenvectors ( n is desired dimension
    # of rescaled data array, or dims_rescaled_data)
    #evals = evals[:pca_dimension]
    #evecs = evecs[:, :pca_dimension]
    # carry out the transformation on the data using eigenvectors
    # and return the re-scaled data, eigenvalues, and eigenvectors
    return np.dot(evecs.T, z_score.T).T, evals, evecs, z_score
'''
@fn plot
@brief show and save plotted time-series data
@param data : 2D NumPy array : vertical:date, horizontal:asset class, economic indicators, or etc.
@return none
'''
def plot(data):
    from mpl_toolkits.mplot3d import Axes3D
    import matplotlib
    matplotlib.use('TkAgg')
    from matplotlib import pyplot as plt
    
    print "===Plots==="
    
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
    
    #Save Plot Image
    plt.savefig("plot.png", dpi=300)
    
    #Show Plot Image
    plt.show()
'''
@fn line_graph
@brief show and save line graph
@param data : 2D NumPy array : vertical:date
@return none
'''    
def line_graph(data, timestamp):
    import matplotlib
    matplotlib.use('TkAgg')
    from matplotlib import pyplot as plt
    import matplotlib.dates as mdates
    import datetime

    print "===Draw Composite Index Graph==="
    
    fig = plt.figure()
    ax = fig.add_subplot(111)

    #convert timestamp data for x-axis
    dates = mdates.date2num(timestamp)
    
    ax.plot(dates, data)

    #format labels
    ax.set_xlabel('Date')
    ax.set_ylabel('composite index')
    
    #format ticks
    years = mdates.YearLocator()    #every year
    months = mdates.MonthLocator()  #every month
    yearsFmt = mdates.DateFormatter('%Y')
    
    ax.xaxis.set_major_locator(years)
    ax.xaxis.set_major_formatter(yearsFmt)
    ax.xaxis.set_minor_locator(months)
    
    datemin = datetime.date(timestamp.min().year, 1, 1)
    datemax = datetime.date(timestamp.max().year + 1, 1, 1)
    ax.set_xlim(datemin, datemax)
    
    #format the coordinates message box
    ax.format_xdata = mdates.DateFormatter('%Y/%m/%d')
    #ax.gird(True)
    
    #Save Plot Image
    plt.savefig("composite index.png", dpi=300)
        
    #Show Plot Image
    plt.show()
    
'''
@fn loadcsv_no_header
@brief load csv data with no header
@param filename : path_to_the_csv_file.csv : file path to the csv file
@param methods : loading methods : 0-NumPy(loadtxt), 1-NumPy (genfromtxt), 2-Pandas(WARNING), 3-CSV
@return data : NumPy 2D Array : data
'''
def loadcsv_no_header(filename, methods = 0):
    import numpy as np  #necessary for @return data (NumPy 2D Array)
       
    print "===Load CSV==="
       
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
@return timestamp : string Array : time stamp
@return data : NumPy 2D Array (float64)w : data
'''
def loadcsv(filename, header_rows = 1):
    import numpy as np  #necessary for @return data (NumPy 2D Array)
    import csv
    from datetime import datetime as dt
    
    print "===Load CSV==="
    
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

    # TEMPOLARY ADJUSTMENT === avoid subdivision with zero (stdev could be 0) ===
    csv_data = csv_data[:1900]
    timestamp = timestamp[:1900]

    #sort data according to time stamp
    if timestamp.size > 1:
        if timestamp[0] > timestamp[1]:
            timestamp = timestamp[::-1]
            csv_data = csv_data[::-1]
    return header, timestamp, csv_data

'''
Main procedure
'''
import numpy as np
#from datetime import datetime as dt

pca_dimensions = 3

print "===Program Initiated==="

#Read Data
#raw_data = loadcsv_no_header("series.csv", 0)
header, timestamp, raw_data = loadcsv("series_raw.csv")

#Principal Component Analysis
_, n = raw_data.shape
data_pca, evals, evecs, z_score = PCA(raw_data, pca_dimensions, 120)

#Calculate Weighted Average
composite_index = 0.0
for i in range(0, pca_dimensions):
    composite_index = composite_index + data_pca[:, i]*evals[i] #/100.0

#Save Data
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
np.savetxt("series_results.csv", csv_series_result, delimiter=',', fmt='%s')

#Z-scores
csv_header = np.hstack((np.array(['Date']), header))
csv_z_score = np.vstack((csv_header, np.vstack((csv_timestamp, z_score.T)).T))
np.savetxt("z-score.csv", csv_z_score, delimiter=',', fmt='%s')

#PCA_result
csv_header[0] = 'Proportion of Variance'    #rename
csv_pca_result = np.vstack((np.hstack((np.array(['Principal Components']), index_numbers)), np.vstack((csv_header, np.vstack((evals.T, evecs)).T)).T))
np.savetxt("pca_result.csv", csv_pca_result, delimiter=',', fmt='%s')

#Plot PCA result
plot(data_pca)

#Draw Composite Results
line_graph(composite_index, timestamp)