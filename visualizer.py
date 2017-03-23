'''
visualizer.py
@description Provide Graph-drawing Methods for Principle Component Analysis on Time-series Data
@author Kenichi Yorozu
@email rozken@gmail.com
@notice Source files on this repository is provided as-is and no guarantee
        or warranty is provided for any damage that my arise from using it.
        This code is free for your own use, the only thing I ask is small
        credit somewhere for my work. An e-mail saying you found it useful
        would also be much appreciated by myself.
'''

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
    import numpy as np
    
    print "===Drawing Heatmap==="
    
    #convert timestamp data for x-axis
    dates = mdates.date2num(timestamp)
    
    ### TEMPOLARY ADJUSTMENT ###
    header = np.array([0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32,33])
    
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
@fn heatmap
@brief show and save heatmap
@param data : 2D NumPy array : correlation Matrix
@param filename : filename for save png
@param folder_path : folder path for save png
@param IsSave : whether or not save graph as png (default : True)
@param IsShow : whether or not show graph on display (default: True)
@return none
'''
def corMat(data, filename = "correlation matrix", folder_path = "graphs/", IsSave = True, IsShow = True):
    import seaborn as sns
    import matplotlib.pyplot as plt
    import pandas as pd
    import numpy as np
    
    print "===Drawing Correlation Coefficients Matrix==="
    
    #build up data for heatmap
    df = []
    m, n = data.shape   #assume data is 2-dimensional
    for i in range(m):
        for j in range(n):
            df.append({
                    'x'    : i,
                    'y'    : j,
                    'value': data[i, j]
                })

    df = pd.DataFrame(df)
    df['x'] = df['x'].astype(int)
    df['y'] = df['y'].astype(int)
    df['value'] = df['value'].astype(float)
    
    fig = plt.figure()
    ax = fig.add_subplot(111)
        
    df_pivot = pd.pivot_table(data=df, values='value', columns='x', index='y', aggfunc=np.mean)
    sns.heatmap(df_pivot, vmin=-1.0, vmax = 1.0, square=False, ax = ax)

    ax.grid(True)
    
    if IsSave:
        #Save Plot Image
        plt.savefig(folder_path + filename + ".png", dpi=300)
    
    if IsShow:
        #Show Plot Image
        plt.show()
    
    #Free Memory
    plt.close()