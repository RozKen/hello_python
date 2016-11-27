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
- put reporting files together (eval & evec)
- add column name and time stamp
- plot principal components time series
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
    
    m, n = data.shape
    print "data: ", m, "-days x ", n, "-assets matrix"
    # Calculate Z-score
    data_normalize = np.array(data[:normalize_days])
    m_n, n_n = data_normalize.shape
    print "- for normalization: ", m_n, "-days x ", n_n, "-assets matrix"
    print "data.mean: ", data_normalize.mean(axis = 0)
    print "data.std: ", data_normalize.std(axis = 0)
    z_score =(data - data_normalize.mean(axis = 0)) / data_normalize.std(axis = 0)
    #np.savetxt("z-score.csv", z_score, delimiter=',')
    z_score_normalize = z_score[:normalize_days]
    print "Z-score: ", z_score[0]
    
    # calculate the covariance matrix (days - column: rowvar = 0, unbiased: bias = 0)
    covMat = np.cov(z_score_normalize, rowvar = 0, bias = 0)
    # calculate eigenvectors & eigenvalues of the covariance matrix
    evals, evecs = la.eigh(covMat)
    # sort eigenvalue in decreasing order
    idx = np.argsort(evals)[::-1]
    evals = evals[idx]
    # sort eigenvectors according to same index
    evecs = evecs[:,idx]
    #select the first n eigenvectors ( n is desired dimension
    # of rescaled data array, or dims_rescaled_data)
    evecs = evecs[:, :pca_dimension]
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
    from matplotlib import pyplot as plt
    
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
Main procedure
'''
import numpy as np
#Read Data
raw_data = np.loadtxt("series_raw.csv",delimiter=',')
#Principal Component Analysis
data_pca, evals, evecs, z_score = PCA(raw_data, 3, 120)
#Save Data
np.savetxt("result.csv", data_pca, delimiter=',')
np.savetxt("z-score.csv", z_score, delimiter=',')
np.savetxt("variance.csv", evals.T, delimiter=',')
np.savetxt("loadings.csv", evecs, delimiter=',')
#Plot PCA result
plot(data_pca)
