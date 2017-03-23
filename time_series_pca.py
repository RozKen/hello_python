'''
time_series_pca.py
@description Simple Principle Component Analysis Methods on Time-series Data
@author Kenichi Yorozu
@email rozken@gmail.com
@notice Source files on this repository is provided as-is and no guarantee
        or warranty is provided for any damage that my arise from using it.
        This code is free for your own use, the only thing I ask is small
        credit somewhere for my work. An e-mail saying you found it useful
        would also be much appreciated by myself.
'''

'''
@fn PCA
@brief Execute Principal Component Analysis
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
@fn PCAwithComp
@brief Execute PCA and Compose index
@param data : 2D NumPy array : vertical:date, horizontal:asset class, economic indicators, or etc.
@param normalize_days : integer : days to be used in normalization of PCA
@return np.dot(evecs.T, z_score.T).T : 2D NumPy array : Time series of principal components (default is 3 series)
@return evals : NumPy array : Eigen Values of covariance matrix used for analysis
@return evecs : 2D NumPy array : Eigen Vectors of covariance matrix used for analysis
@return z_score : 2D NumPy array : Normalized data of input "data"
@return composite_index : 1D NumPy array : weighted average of principal components
'''
def PCAwithComp(data, settings, normalize_days = 120, pca_dimensions = 3):
    import numpy as np
    
    #Principal Component Analysis
    data_pca, evals, evecs, z_score = PCA(data, normalize_days)
    
    #Determining +/- direction of each PCA index
    signs = np.array([])
    for i in range(0, evecs[0].size):
        signs = np.hstack((signs, np.array(settings * evecs[:,i]).sum()))
        signs[i] = signs[i] / abs(signs[i])
    
    # apply signs to evecs, 
    for i in range(0, evecs[0].size):
        evecs[:, i] = signs[i] * evecs[:, i]
        data_pca[:, i] = signs[i] * data_pca[:, i]
        #Translate as Z-score    ###TEMPOLARY######################################
        pca_i_average = np.average(data_pca[:, i])
        pca_i_stdev = np.std(data_pca[:, i])
        data_pca[:, i] = (data_pca[:, i] - pca_i_average) / pca_i_stdev * 100.0

    #Calculate Weighted Average
    composite_index = np.zeros(data_pca[:,0].size)
    for i in range(0, pca_dimensions):
        composite_index = composite_index + data_pca[:, i] * evals[i]

    return np.dot(evecs.T, z_score.T).T, evals, evecs, z_score, composite_index