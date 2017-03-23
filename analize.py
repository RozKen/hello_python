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
- review methods for +/- direction of each PCA index
- review level adjustment on composite index with regression
- eliminate Manual Inputs

***TEMPOLARY ADJUSTMENT***
#4437 days until raw_data acquired (2015/7/22) *** should be timestamp.size
#MANUAL INPUT [12] (column index of S&P500 for standardization of compound index)
#assume 'settings' is 1D Column : load_settings
historical_start = 3407 #days since raw_data acquired (2012 ~ )
'''

import time_series_pca as pc
import loader as ld
import visualizer as vz
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
graph_folder = "graphs/"        #default value in plotting functions

#Read Settings
settings = ld.settings(input_folder + "settings.csv")

#Read Data
#raw_data = loadcsv_no_header("series.csv", 0)
header, timestamp, raw_data = ld.csv(input_folder + "series_raw.csv")

#Calculate and Save Historical PCA results
#how to deal with new PCA series which is not consistent with historical series
method = 201
    
#Output Images for Debug
IsDebug = True

#variables for historical data logging
index_historical = np.array([])
eval_historical = np.array([])
evec_historical = np.array([])

for history in range(historical_start, timestamp.size):
    #Principal Component Analysis
    _, n = raw_data.shape
    data_pca, evals, evecs, z_score = pc.PCA(raw_data[:history + 1], normalization_days)
    
    #Determining +/- direction of each PCA index
    signs = np.array([])
    for i in range(0, evecs[0].size):
        signs = np.hstack((signs, np.array(settings * evecs[:,i]).sum()))
        signs[i] = signs[i] / abs(signs[i])
    
    # apply signs to evecs, 
    for i in range(0, evecs[0].size):
        evecs[:, i] = signs[i] * evecs[:, i]
        data_pca[:, i] = signs[i] * data_pca[:, i]
        #Translate as Z-score    ###TEMPOLARY###
        pca_i_average = np.average(data_pca[:, i])
        pca_i_stdev = np.std(data_pca[:, i])
        data_pca[:, i] = (data_pca[:, i] - pca_i_average) / pca_i_stdev * 100.0
    
    if (method - method % 1000) / 1000 == 0:
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
        
    if (method % 1000 - method % 100) / 100 == 1:  #100, 101, 110, 111
        ###    set -100.0 on minimum S&P500 day / average of normalization_days around minimum S&P 500 days for historical
        ###    fit new PCA to historical series

        #acquire argmin of column #13 : S&P500 : 2009/3/9 : 676.53 pt
        ###TEMPOLARY ADJUSTMENT - MANUAL INPUT [12]
        argmin = np.argmin(raw_data[:,12])

        
        if history == historical_start:
            if (method - 100 - method % 10)/10 == 0:  #100, 101
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
                vz.line_graph(temp, timestamp[:history], "regression result " + str(history), graph_folder + "method " + str(method), IsShow = False)

    elif (method % 1000 - method % 100) / 100 == 2:
        ###    scale historical series so as to (max, min) or (average of normalization_days around max, min)== (100.0, -100.0)
        ###    scale new PCA so as to (max, min) == (100.0, -100.0)
        
        #Fit new PCA
        if (method - 100 - method % 10)/10 == 0:  #200, 201
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
            if history > historical_start and history % 100 == 0:
                temp = np.vstack((index_historical, old[:-1], composite_index[:-1]))
                vz.line_graph(temp, timestamp[:history], "regression result " + str(history), graph_folder + "method " + str(method) + "/", IsShow = False)

    #Stack new PCA series to historical index
    if history == historical_start:
        index_historical = np.hstack((index_historical, composite_index))
        eval_historical = np.hstack((eval_historical, evals))
        evec_historical = evecs

    else:
        if method % 10 == 0:
            #add latest value of new PCA to historical series
            index_historical = np.hstack((index_historical, composite_index[-1]))
        else:   #methods % 10 == 1:
            #add latest move of new PCA to historical series
            index_historical = np.hstack((index_historical, index_historical[-1] + composite_index[-1] - composite_index[-2]))
        eval_historical = np.vstack((eval_historical, evals))
        evec_historical = np.dstack((evec_historical, evecs))

### TEMPOLARY ADJUSTMENT
vz.heatmap(z_score[4600:], timestamp[4600:], header)

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
#vz.plot(data_pca)

#Draw Composite Results
vz.line_graph(composite_index, timestamp)

#Draw Historical Cumulative Result
vz.line_graph(index_historical, timestamp, "historical gfms")

print "=== Program Ended ==="