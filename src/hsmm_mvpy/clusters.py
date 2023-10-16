'''
Clustering
'''

from mne.viz import plot_topomap
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
import multiprocessing as mp
import itertools

cols = np.array(['tab:blue',
'tab:orange',
'tab:green',
'tab:red',
'tab:purple',
'tab:brown',
'tab:pink',
'tab:gray',
'tab:olive',
'tab:cyan'])

def mahalanobis(x=None, data=None):
    """Compute the Mahalanobis Distance between each row of x and the data  
    x    : vector or matrix of data with, say, p columns.
    data : ndarray of the distribution from which Mahalanobis distance of each observation of x is to be computed.
    """
    if x.shape[0] == 1:
        return np.array([0])
    else:
        x_minus_mu = x - np.mean(data, axis=0)
        VI = sp.linalg.inv(np.cov(data.T))
        left_term = np.dot(x_minus_mu, VI)
        mahal = np.dot(left_term, x_minus_mu.T)
        return np.sqrt(np.diag(mahal))

def cluster_events(init, lkhs, mags, channels, times, method='time_x_lkh_mags', max_clust=5, p_outlier=.01, info=None, nr_clust=None, calc_outliers=False, alpha=.05):
    #method = 'time_x_lkh' for clustering based on time and likelihood, or 'time_x_lkh_x_mags' for
    #         clustering based on time, likelihood and mags
    
    #zscore time and lkh
    times_scaled = (times - np.mean(times)) / np.std(times)
    lkhs_scaled = (lkhs - np.mean(lkhs)) / np.std(lkhs)

    #features
    feat = np.vstack((times_scaled,lkhs_scaled)).T
    if method == 'time_x_lkh_x_mags':
        feat = np.hstack((feat,mags))
    
    #calculate clusters up to max_clust
    kmeans_kwargs = {
        "init": "random",
        "n_init": 20,
        "max_iter": 400}

    kmeans_sols = []
    for k in range(1, max_clust+1):
        kmeans_sols.append(KMeans(n_clusters=k, **kmeans_kwargs))
        kmeans_sols[-1].fit(feat)
    
    #if nr_clust not given, determine number of clusters based on silhouette coefficient
    if not nr_clust:
        silhouette_coefficients = []

        #start at 2 clusters for silhouette coefficient
        if init.cpus > 1:
            with mp.Pool(processes=init.cpus) as pool:
                silhouette_coefficients = pool.starmap(silhouette_score, 
                    zip(itertools.repeat(feat), [x.labels_ for x in kmeans_sols[1:max_clust]]))
        else:
            for k in range(2, max_clust+1):
                silhouette_coefficients.append(silhouette_score(feat, kmeans_sols[k-1].labels_))

        #plot coefficients
        _, ax = plt.subplots(1,1,figsize=(5,5))
        plt.plot(range(2, max_clust+1), silhouette_coefficients)
        plt.xticks(range(2, max_clust+1))
        plt.xlabel("Number of Clusters")    
        plt.ylabel("Silhouette Coefficient")
        plt.title("Silhouette Coefficients")
        plt.show()
        plt.pause(.1)

        #best coefficient / select fit:
        try_n = np.argmax(silhouette_coefficients) + 2 #starts at 2
    else:
        try_n = nr_clust

    #explore solutions until user inputs 0, starting with proposed solution 
    while try_n > 0:

        #get solution
        kmeans = kmeans_sols[try_n-1]
    
        #determine outliers given fit
        if calc_outliers:
            maha_distances = np.zeros((kmeans.labels_.shape[0],2))
            for cl in range(kmeans.n_clusters):
                maha_distances[kmeans.labels_ == cl,0] = mahalanobis(feat[kmeans.labels_ == cl,:], feat[kmeans.labels_ == cl,:])
                maha_distances[kmeans.labels_ == cl,1] = 1 - sp.stats.chi2.cdf(maha_distances[kmeans.labels_ == cl,0], feat.shape[1]-1)                #add p-values

        #calc channels per cluster (needs to be done in two steps to get vmin/max)
        channels_cluster = np.zeros((kmeans.n_clusters, channels.shape[1]))
        for cl in range(kmeans.n_clusters):
            outliers = maha_distances[kmeans.labels_ == cl,1] < p_outlier if calc_outliers else np.repeat(False, np.sum(kmeans.labels_ == cl))
            channels_cluster[cl,:] = np.median(channels[kmeans.labels_ == cl,:][~outliers,:],axis=0)
            
        vmax = np.nanmax(np.abs(channels_cluster[:])) if np.nanmax(channels_cluster[:]) >= 0 else 0
        vmin = -np.nanmax(np.abs(channels_cluster[:])) if np.nanmin(channels_cluster[:]) < 0 else 0

        #plot results
        axes=[]
        yrange = (np.max(lkhs) - np.min(lkhs))
        topo_size = .6 * (np.max(times) - np.min(times)) #6% of time scale
        topo_size_y = .4 * yrange

        _, ax = plt.subplots(1,1,figsize=(20,3))
        time_step=1000/init.sfreq
        for cl in range(kmeans.n_clusters):
            outliers = maha_distances[kmeans.labels_ == cl,1] < p_outlier if calc_outliers else np.repeat(False, np.sum(kmeans.labels_ == cl))
            ax.plot(times[kmeans.labels_ == cl][outliers]*time_step, lkhs[kmeans.labels_ == cl][outliers], 'x', color=cols[cl],alpha=alpha)
            ax.plot(times[kmeans.labels_ == cl][~outliers]*time_step, lkhs[kmeans.labels_ == cl][~outliers], '.', color=cols[cl],alpha=alpha)

            #topo
            if info:
                axes.append(ax.inset_axes([(np.median(times[kmeans.labels_ == cl][~outliers]) - topo_size / 2)* time_step, np.max(lkhs[kmeans.labels_ == cl][~outliers]) + .1 * yrange, topo_size * time_step, topo_size_y], transform=ax.transData))
                plot_topomap(channels_cluster[cl,:], info, axes=axes[-1], show=False,
                                    cmap='Spectral_r', vlim=(vmin, vmax), sensors=False, contours=False)

        bottom,_ = ax.get_ylim()
        ax.set_xlim(0, init.mean_d*time_step)
        ax.set_ylim((bottom, np.max(lkhs) + yrange*.5))
        
        ax.tick_params(axis='x', labelsize=16)
        ax.tick_params(axis='y', labelsize=16)

        ax.set_xlabel('Time (ms)', fontsize=18)
        ax.set_ylabel('Likelihood', fontsize=18)
        if plt.get_backend()[0:2] == 'Qt': #fixes issue with yscaling
            plt.tight_layout()
        plt.pause(.1)

        try_n = int(input('Do you agree with this solution [enter \'0\'], or would you like to explore a different number of clusters [enter the number of clusters]?'))

    #get cluster info
    best_n_clust = kmeans.n_clusters
    cl_times = np.zeros((best_n_clust,)) #cluster times (not necessarily sorted)
    cl_mags = np.zeros((best_n_clust, mags.shape[1])) #cluster mags
    for cl in range(best_n_clust):
        outliers = maha_distances[kmeans.labels_ == cl,1] < p_outlier if calc_outliers else np.repeat(False, np.sum(kmeans.labels_ == cl))
        cl_times[cl] = np.median(times[kmeans.labels_ == cl][~outliers])
        cl_mags[cl,:] = np.median(mags[kmeans.labels_ == cl,:][~outliers,:], axis=0)

    #calc mags and params based on nclust, get max likelihood in each
    mags = cl_mags[np.argsort(cl_times), :] #mags are easy, just sort by time
    cl_times = np.sort(cl_times) #sort
    cl_durations = np.hstack((cl_times, init.mean_d)) - np.hstack((0, cl_times)) #get stage durations
    pars = np.array([np.repeat(init.shape, best_n_clust + 1), init.mean_to_scale((cl_durations - np.hstack((0, np.repeat(init.location,best_n_clust)))), init.shape)]).T #calc params, take into account locations 
    null_stages = np.where(pars[:,1] < 0)
    pars[null_stages, 1] = 1/init.shape #avoids impossible parameters
    return mags, pars