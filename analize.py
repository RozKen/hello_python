﻿#encoding: utf-8
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

'''
@TODO
- eliminate Manual Inputs

***TEMPOLARY ADJUSTMENT***
#MANUAL INPUT [12] (column index of S&P500 for standardization of compound index)
'''
import statistics as st         #Load Statistics Method
import time_series_pca as pc    #Load PCA Core Method
import loader as ld             #Load CSV Handlers
import visualizer as vz         #Load Graph-drawing Methods
import utility as ut            #Load Utility Function
import numpy as np
#from datetime import datetime as dt

print "===Program Initiated==="

'''
Set Constants
'''
#Setting for PCA
pca_dimensions = 3          #valid series for composite index
normalize_days = 120    #sampling span for PCA

#Path to folders
output_folder = "data_output/"
input_folder = "data_input/"
graph_folder = "graphs/"        #default value in plotting functions

#Setting for Historical Data Logging 
IsHistorical = False        #Log Historical Data
h_start_str = "2012/1/1"    #days since raw_data acquired (2012 ~ )
h_method = 201              #how to deal with new PCA series which is not consistent with historical series
h_param1 = 12               #acquire argmin of column #13 : S&P500 : 2009/3/9 : 676.53 pt (used when h_method = 100, 101, 110, or 111)    ###TEMPOLARY ADJUSTMENT### - MANUAL INPUT [12]

#Debug Flag
IsDebug = False          #Output Graphs for Debug - Only for Historical
d_span = 1            #interval of days to Draw Graph

'''
Read CSV Data
'''
#Read Settings
settings = ld.settings(input_folder + "settings.csv")

#Read Data
header, timestamp, raw_data = ld.csv(input_folder + "series_raw.csv")
h_start = ut.argDate(timestamp, h_start_str)

'''
Analyze Original Data
'''
#Draw Original Data
#print st.summary(raw_data)

#vz.line_graph(raw_data.T, timestamp)

vz.corMat(np.corrcoef(st.corMat(raw_data * settings[0], normalize_days)), header, IsMask=True)

#Draw data distribution
#z = st.z_scores(raw_data)
#vz.histogram(z, header)

#Draw heatmap
h = st.rolling_z(raw_data * settings[1], days = normalize_days)
#np.savetxt(output_folder + "heatmap.csv", h, delimiter=',', fmt='%s')
vz.heatmap(h[ut.argDate(timestamp,h_start_str) - normalize_days:], timestamp[ut.argDate(timestamp,h_start_str):], header, dateFrom = "2014/01/01")

'''
Execute PCA
'''
if IsHistorical == False:
    #Principal Component Analysis
    data_pca, evals, evecs, z_score, composite_index = pc.PCAwithComp(raw_data, settings[0], normalize_days, pca_dimensions)
    
else:
    #variables for historical data logging
    index_historical = np.array([])
    eval_historical = np.array([])
    evec_historical = np.array([])
    
    for history in range(h_start, timestamp.size):
        
        #Principal Component Analysis
        data_pca, evals, evecs, z_score, composite_index = pc.PCAwithComp(raw_data[:history + 1], settings[0], normalize_days, pca_dimensions)
        
        '''
        Log Historical Data
        '''
        if (h_method % 1000 - h_method % 100) / 100 == 1:  #100, 101, 110, 111
            ###    set -100.0 on minimum S&P500 day / average of normalize_days around minimum S&P 500 days for historical
            ###    fit new PCA to historical series
    
            #acquire argmin of column #13 : S&P500 : 2009/3/9 : 676.53 pt
            argmin = np.argmin(raw_data[:,h_param1])
            print timestamp[argmin][0].strftime("%Y/%m/%d") + " : " + header[h_param1] + " : " + str(raw_data[argmin, h_param1])
    
            
            if history == h_start:
                if (h_method - 100 - h_method % 10)/10 == 0:  #100, 101
                    # - Set -100.0 for minimum-S&P500 day (2009/3/9)
                    composite_denom = composite_index[argmin]
                else:   #110, 111
                    # - Set -100.0 for average of normalize_days around minimum-S&P500 day
                    composite_denom = np.average(composite_index[argmin - normalize_days/2: argmin + normalize_days/2])
                composite_index = composite_index / composite_denom * (-100.0)
            
            #fit composite_index to index_historical
            if history != h_start:
                p = np.poly1d(np.polyfit(composite_index[:-1], index_historical, 1))
                if IsDebug:
                    old = composite_index[:-1] 
                composite_index = p(composite_index)
                
            ###DEBUG: output graphs on regression###
            if IsDebug:
                if history > h_start and history % d_span == 0:
                    temp = np.vstack((index_historical, old, composite_index[:-1]))
                    vz.line_graph(temp, timestamp[:history], "regression result " + str(history), graph_folder + "method " + str(h_method) + "/", IsShow = False)
    
        elif (h_method % 1000 - h_method % 100) / 100 == 2:
            ###    scale historical series so as to (max, min) or (average of normalize_days around max, min)== (100.0, -100.0)
            ###    scale new PCA so as to (max, min) == (100.0, -100.0)
            
            #Fit new PCA
            if (h_method - 100 - h_method % 10)/10 == 0:  #200, 201
                composite_max = np.max(composite_index)
                composite_min = np.min(composite_index)
            else:   #210, 211
                peak = np.argmax(composite_index)
                trough = np.argmin(composite_index)
                composite_max = np.average(composite_index[max(peak - normalize_days/2, 0) : min(peak + normalize_days/2, composite_index.size)])
                composite_min = np.average(composite_index[max(trough - normalize_days/2, 0) : min(trough + normalize_days/2, composite_index.size)])
            
            composite_peak2trough = composite_max - composite_min
            if IsDebug:
                old = composite_index
            
            composite_index = (composite_index - composite_min) / composite_peak2trough * 200.0 -100.0
                                
            ###DEBUG: output graphs on regression###
            if IsDebug:
                if history > h_start and history % d_span == 0:
                    temp = np.vstack((index_historical, old[:-1], composite_index[:-1]))
                    vz.line_graph(temp, timestamp[:history], "regression result " + str(history), graph_folder + "method " + str(h_method) + "/", IsShow = False)
    
        #Stack new PCA series to historical index
        if history == h_start:
            index_historical = np.hstack((index_historical, composite_index))
            eval_historical = np.hstack((eval_historical, evals))
            evec_historical = evecs
    
        else:
            if h_method % 10 == 0:
                #add latest value of new PCA to historical series
                index_historical = np.hstack((index_historical, composite_index[-1]))
            else:   #h_methods % 10 == 1:
                #add latest move of new PCA to historical series
                index_historical = np.hstack((index_historical, index_historical[-1] + composite_index[-1] - composite_index[-2]))
            eval_historical = np.vstack((eval_historical, evals))
            evec_historical = np.dstack((evec_historical, evecs))

'''
Output Data and Log to CSV Files
'''
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
csv_header = np.hstack((np.array(['Date']), ld.encode_array(header, 'cp932')))
csv_z_score = np.vstack((csv_header, np.vstack((csv_timestamp, z_score.T)).T))
np.savetxt(output_folder + "z-score.csv", csv_z_score, delimiter=',', fmt='%s')

#PCA_result
csv_header[0] = 'Proportion of Variance'    #rename
csv_pca_result = np.vstack((np.hstack((np.array(['Principal Components']), index_numbers)), np.vstack((csv_header, np.vstack((evals.T, evecs)).T)).T))
np.savetxt(output_folder + "pca_result.csv", csv_pca_result, delimiter=',', fmt='%s')

if IsHistorical:
    #Historical Index
    csv_historical_header = np.array(['Date', 'Global Financial Markets Monitoring System'])
    csv_historical = np.vstack((csv_timestamp, index_historical)).T
    csv_historical = np.vstack((csv_historical_header, csv_historical))
    np.savetxt(output_folder + "historical_series.csv", csv_historical, delimiter=',', fmt='%s')
    
    #Historical Eigen Values (Proportions)
    csv_eval_historical_header = np.hstack((np.array(['Date']), index_numbers))
    csv_eval_historical = np.vstack((csv_timestamp[h_start:], eval_historical.T)).T
    csv_eval_historical = np.vstack((csv_eval_historical_header, csv_eval_historical))
    np.savetxt(output_folder + "historical_evals.csv", csv_eval_historical, delimiter=',', fmt='%s')
    
    #Historical Eigen Vectors
    csv_evec_historical_header = np.hstack((np.array(['Date']), ld.encode_array(header, 'cp932')))
    for i in range(0, pca_dimensions):
        csv_evec_historical = np.vstack((csv_timestamp[h_start:], evec_historical[:, i, :])).T
        csv_evec_historical = np.vstack((csv_evec_historical_header, csv_evec_historical))
        np.savetxt(output_folder + "historical_evecs-" + str(i) + ".csv", csv_evec_historical, delimiter=',', fmt='%s')

'''
Draw and Save Graphs
'''
#Plot PCA result
#vz.plot(data_pca)

#Draw Composite Results
vz.line_graph(composite_index, timestamp)

if IsHistorical:
    #Draw Historical Cumulative Result
    vz.line_graph(index_historical, timestamp, "historical gfms")

print "=== Program Ended ==="