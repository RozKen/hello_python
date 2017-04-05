#encoding: utf-8
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
    import seaborn as sns
    
    #notice! when you update font list, remove <user home>/.matplotlib/fontList.cache (windows)
    sns.set(font=['Meiryo UI'])
    
    print "===Plotting==="
    
    _, n = data.shape
    clr1 = '#202682'
    fig = plt.figure(figsize = (11, 6))
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
    plt.tight_layout()
    
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
    import seaborn as sns
    
    #notice! when you update font list, remove <user home>/.matplotlib/fontList.cache (windows)
    sns.set(font=['Meiryo UI'])

    print "===Drawing Line Graph==="
    
    fig = plt.figure(figsize = (11, 6))
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
    plt.tight_layout()
    
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
@param dataFrom : string : "yyyy/mm/dd" : graph start (default : timestamp.min)
@param dataTo : string : "yyyy/mm/dd" : graph end (default : timestamp.max)
@return none
'''
def heatmap(data, timestamp, header, filename = "heatmap", folder_path = "graphs/", IsSave = True, IsShow = True, dateFrom = "", dateTo = ""):
    import utility as ut    #Load Utility Function
    import seaborn as sns
    import matplotlib.pyplot as plt
    import pandas as pd
    import numpy as np
    
    #notice! when you update font list, remove <user home>/.matplotlib/fontList.cache (windows)
    sns.set(font=['Meiryo UI'])
    
    print "===Drawing Heatmap==="
    
    #set x-axis range
    argmin = 0
    argmax = timestamp[:, 0].size
    if dateFrom != "":
        argmin = ut.argDate(timestamp, dateFrom)
    if dateTo != "":
        argmax = ut.argDate(timestamp, dateTo)
    
    data = data[argmin:argmax]
    timestamp = timestamp[argmin:argmax]
    
    #build up data for heatmap
    df = []
    m, n = data.shape   #assume data is 2-dimensional
    for i in range(m):
        for j in range(n):
            df.append({
                    'Date'      : timestamp[i, 0].strftime("%Y/%m/%d"),         #dates[i],    ###TEMPORARY
                    'Header'    : j,
                    'Value'     : data[i, j]
                })

    df = pd.DataFrame(df)
    #df['Date'] = mdates.num2date(df['Date'])
    df['Date'] = df['Date'].astype(str)         #(int)    ###TEMPORARY
    df['Header'] = df['Header'].astype(str)
    df['Value'] = df['Value'].astype(float)

    fig = plt.figure(figsize = (10, 6))
    ax = fig.add_subplot(111)
    
    #list up x-axis labels    :    monthly labels
    xlabel = []
    count = 0
    for day in timestamp:
        if count == 0:
            xlabel = xlabel + [day[0].strftime("%y/%m").replace('/0','/')]     #remove 0 before months
        elif timestamp[count - 1, 0].month != day[0].month:
            if timestamp[count - 1, 0].year != day[0].year:
                xlabel = xlabel + [day[0].strftime("%y/%m").replace('/0','/')]
            else:
                xlabel = xlabel + [day[0].strftime("/%m").replace('/0','/').replace('/','')]
        else:
            xlabel = xlabel + [""]
        count = count + 1
    print xlabel
    
    #list up y-axis labels
    ylabel = []
    for i in header:
        ylabel = ylabel + [i]
    
    df_pivot = pd.pivot_table(data=df, values='Value', columns='Date', index='Header', aggfunc=np.mean)
    sns.heatmap(df_pivot, cmap='RdBu', xticklabels=xlabel, yticklabels=ylabel, vmin=-1.5, vmax = 1.5, square=False, ax = ax)
    
    for item in ax.get_yticklabels():
        item.set_rotation(0)
    for item in ax.get_xticklabels():
        item.set_rotation(90)
    
    #format labels
    ax.set_xlabel('Date')
    ax.set_ylabel(filename)
    
    plt.tight_layout()
    
    if IsSave:
        #Save Plot Image
        plt.savefig(folder_path + filename + ".png", dpi=300)
    
    if IsShow:
        #Show Plot Image
        plt.show()
    
    #Free Memory
    plt.close()

'''
@fn histogram
@brief show and save histogram
@param data : 2D NumPy array : Z-scores
@param header : name of each columns
@param filename : filename for save png
@param folder_path : folder path for save png
@param IsSave : whether or not save graph as png (default : True)
@param IsShow : whether or not show graph on display (default: True)
@return none
'''
def histogram(data, header, filename = "histogram", folder_path = "graphs/", IsSave = True, IsShow = True):
    import matplotlib.pyplot as plt
    import seaborn as sns
    import math
    
    print "===Drawing Histogram==="
    
    #notice! when you update font list, remove <user home>/.matplotlib/fontList.cache (windows)
    sns.set(font=['Meiryo UI'])
    sns.set_context(rc={"font.size":5, "axes.titlesize":8, "xtick.labelsize":8, "ytick.labelsize":8 })
    
    days, assets = data.shape
    columns = 6
    raws = int(math.ceil(float(assets) / float(columns)))
    bins = int(math.ceil(float(days) / 100.0) * 5)
    
    fig = plt.figure(figsize = (11, 6))
    count = 0
    for _ in range(assets):
        if count < assets:
            plt.subplot(raws, columns, count + 1)
            plt.title(header[count])
            sns.distplot(data[:, count], kde=False, rug=False, bins = bins)
        count = count + 1
    
    plt.tight_layout()
    
    if IsSave:
        #Save Plot Image
        plt.savefig(folder_path + filename + ".png", dpi=300)
    
    if IsShow:
        #Show Plot Image
        plt.show()
    
    #Free Memory
    plt.close()

'''
@fn corMat
@brief show and save Correlation Matrix
@param data : 2D NumPy array : correlation Matrix
@param header : name of each columns
@param filename : filename for save png
@param folder_path : folder path for save png
@param IsSave : whether or not save graph as png (default : True)
@param IsShow : whether or not show graph on display (default: True)
@return none
'''
def corMat(data, header, filename = "correlation matrix", folder_path = "graphs/", IsSave = True, IsShow = True, IsMask = False, mask_bar = 30.0):
    import seaborn as sns
    import matplotlib.pyplot as plt
    import pandas as pd
    import numpy as np
    #from mpl_toolkits.axes_grid1 import make_axes_locatable
    
    #notice! when you update font list, remove <user home>/.matplotlib/fontList.cache (windows)
    sns.set(font=['Meiryo UI'])
    sns.set_context(rc={"font.size":6.5, "axes.titlesize":8, "xtick.labelsize":8, "ytick.labelsize":8 })
    
    print "===Drawing Correlation Coefficients Matrix==="
    
    #build up data for heatmap
    df = []
    mask = np.array([])
    m, n = data.shape   #assume data is 2-dimensional
    for i in range(m):
        temp = np.array([])
        for j in range(n):
            df.append({
                    'x'    : i,
                    'y'    : j,
                    'value': data[i, j] * 100.0
                })
            if i == j:
                temp = np.hstack((temp, np.array([True])))
            else:
                temp = np.hstack((temp, np.array([abs(data[i, j] * 100.0) < mask_bar])))
        if i == 0:
            mask = np.hstack((mask, temp))
        else:
            mask = np.vstack((mask, temp))

    df = pd.DataFrame(df)
    df['x'] = df['x'].astype(int)
    df['y'] = df['y'].astype(int)
    df['value'] = df['value'].astype(float)
    
    #list up axis labels
    label = []
    for i in header:
        label = label + [i]
    
    fig = plt.figure(figsize = (9, 9))
    ax = fig.add_subplot(111)
    
    df_pivot = pd.pivot_table(data=df, values='value', columns='x', index='y', aggfunc=np.mean)
    
    if IsMask:
        sns.heatmap(df_pivot, cmap='RdBu', xticklabels=label, yticklabels=label, vmin=-100.0, vmax = 100.0, square=True, annot = True, fmt=".0f", ax = ax, cbar=False, linewidth = .5, mask = mask)
    else:
        sns.heatmap(df_pivot, cmap='RdBu', xticklabels=label, yticklabels=label, vmin=-100.0, vmax = 100.0, square=True, annot = True, fmt=".0f", ax = ax, cbar=False, linewidth = .5)

    #set layout
    ax.tick_params(labelbottom='off', labeltop='on')
    for item in ax.get_yticklabels():
        item.set_rotation(0)
    for item in ax.get_xticklabels():
        item.set_rotation(90)
    plt.xlabel("")
    plt.ylabel("")
    plt.tight_layout()
    
    if IsSave:
        #Save Plot Image
        plt.savefig(folder_path + filename + ".png", dpi=300)
    
    if IsShow:
        #Show Plot Image
        plt.show()
    
    #Free Memory
    plt.close()