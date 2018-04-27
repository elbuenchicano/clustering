import numpy as np
import matplotlib.pyplot as plt

from    utils import *
from    tsne  import tsneExample
from    sklearn.neighbors.kde import KernelDensity
from    scipy.stats import norm
from    sklearn.externals import joblib


################################################################################
################################################################################
def dirichilet(individual):
    pass

################################################################################
################################################################################
class KDECluster:
    '''
    points is a vector of vectors [[],[]]
    '''
    def __init__(self, points, bw):
        if len(points) < 5: 
            self.kde_   = KernelDensity(kernel='gaussian', bandwidth=bw)
        else:
            self.kde_   = KernelDensity(kernel='epanechnikov', algorithm='ball_tree', bandwidth=bw, leaf_size = 50)
        
        self.points_    = points

        self.kde_.fit(points)

    #..........................................................................
    def compare(self, cluster):
        scores_self = np.exp(self.kde_.score_samples(cluster.points_))
        scores_clus = np.exp(cluster.kde_.score_samples(self.points_))

        m_self      = max(scores_self)
        m_clus      = max(scores_clus)

        return max(m_clus, m_self)
            

################################################################################
################################################################################
def cgrow(x, base):
    ''' Matlab code sample 
    m   = 1;
    x   = 0:0.01:m;
    %ld  = 0.1448; 
    ld  = 0.0869; 
    c   = 0.7; %1.5
    d   = 0.5;
    y   = exp(-d*x/ld) - (x.^2-x) / c;
    %y   = exp(-ld)*(ld.^x) ./ factorial(x);
    %y = (2 ./ (1 + exp(-2.*x)) ) - 1 + (1-0.7616);
    th = 0.3;
    y  = (1-th) * (1-y) + th    
    '''
    ld  = 0.0869 # must be fixed values extracted from log(1e-5)
    c   = 0.7 # cumulative factor
    d   = 0.5 # linear factor
    y   = np.exp(-d*x/ld) - (x**2-x) / c
    y   = (1 - base) * (1 - y) + base
    return y

################################################################################
################################################################################
def kde(samples, directory, config):
    
    max_iter    = config['max_iter']
    prob_th     = config['prob_th']
    bw          = config['bandwidth']

    #...........................................................................
    N = 20
    np.random.seed(1)
    #g1 = np.random.normal(-2, 1, int(0.3 * N) )
    #g2 = np.random.normal(5, 1, int(0.7 * N) )

    g1 = np.random.multivariate_normal([-0.5, -0.5], [[1, 0],[0, 1]], N)
    #plt.plot(g1[:, 0], g1[:, 1], '.')
    #plt.show()

    g2 = np.random.multivariate_normal([5, 6], [[1, 0.5],[0.5, 1]], N)
    #plt.plot(g2[:, 0], g2[:, 1], '.')
    #plt.show()

    g3 = np.random.multivariate_normal([-5, 6], [[1, 0.5],[0.5, 1]], N)
    #plt.plot(g2[:, 0], g2[:, 1], '.')
    #plt.show()

    X = np.concatenate( (g1, g2, g3) )[:, np.newaxis]

    samples = X


    famous_colors = ['r', 'g', 'b', 'black','brown', 'slateblue', 'salmon', 'gold', 'gray', 'orange', 'violet' , 'darkgreen', 'y']

    #  initializer whole clusters ..............................................
    clusters = []
    for sample in samples:
        clusters.append(KDECluster(sample, bw))
         
    iter    = 0
    c_flag  = False

   
    # until max iter or convergence condition is satisfied .....................
    while iter < max_iter and c_flag != True:
        
        i   = iter / max_iter
        th  = cgrow(i, prob_th)

        c_flag  = True
         
        n_clusters      = len(clusters)
        group_flag      = np.zeros(n_clusters)

        clusters_temp   = []

        for i in range(n_clusters):
            #u_progress(i, n_clusters)
            if group_flag[i] == 0:
                group_flag[i]    = 1
                for j in range(n_clusters):
                    if i != j and group_flag[j] == 0:
                        score = clusters[i].compare(clusters[j]) 
                        #print(score)
                        if score > th:
                            clusters[i].points_ = np.concatenate( (clusters[i].points_, clusters[j].points_ ))
                            group_flag[j]       = 1
                            c_flag              = False

                if c_flag == False:
                    clusters[i].kde_.fit(clusters[i].points_)

                clusters_temp.append(clusters[i])
            
        clusters = clusters_temp
        iter    += 1

    #...........................................................................
    color_pos   = 0
    marker      = ['+', 'o', '.',  '+' ,  'o', '.', '+', 'o', '.']
    marker_pos  = -1
    for cluster in clusters:
        c_pos = color_pos % 12
        if c_pos == 0:
            marker_pos += 1
        
        plt.plot(cluster.points_[:, 0], cluster.points_[:, 1], marker = marker[marker_pos] , color = famous_colors[c_pos], linewidth=0)
        color_pos   += 1

    plt.text(0, 0, "N={0} clusters".format(len(clusters)))
    plt.show()
        


    test_clusters = []

    #for sample in samples:
    #    win     = 0
    #    maxp    = 0
    #    for i in range (len(clusters)):
    #        score = clusters[i].kde_.score_samples(sample)
    #        if maxp < score:
    #            maxp = score
    #            win  = i

    #    test_clusters.append(win)
    

    #color_pos   = 0
    #marker      = ['+', 'o', '.',  '+' ,  'o', '.', '+', 'o', '.']
    #marker_pos  = -1

    #for i in range ( len( samples ) ):
    #    c_pos = test_clusters[i] % 13
    #    marker_pos = int(test_clusters[i] / 13)
               
    #    plt.plot(samples[i][:,0], samples[i][:, 1], marker = marker[marker_pos] , color = famous_colors[c_pos], linewidth=0)
    #    plt.annotate(str(test_clusters[i]), xy = (samples[i][:,0], samples[i][:, 1]))

    #plt.text(0, 0, "N={0} clusters".format(len(clusters)))
    #plt.show()
        


################################################################################
################################################################################
def clusteringBottomUp(file, data):
    samples_files   = data['samples_files']
    algorithms      = data['algorithms']
    directory       = data['directory']

    algo_dict   = { 'kde' :kde, 'dirichilet' :dirichilet }

    #...........................................................................
    samples = []
    #for samples_file in samples_files:
    #    samples.append( np.loadtxt(samples_file) )
    
    #samples = np.concatenate(samples)

    #...........................................................................
    for algo in algorithms:
        algo_confs = data[algo] if algo in data else {}
        algo_dict[algo](samples, directory, algo_confs)


################################################################################
################################################################################
def tsne(file, data):
    samples_files   = data['samples_files']

    #...........................................................................
    samples = []
    for samples_file in samples_files:
        samples.append( np.loadtxt(samples_file) )
    
    samples = np.concatenate(samples)

    tsneExample(samples)



################################################################################
################################################################################
################################################################################
################################################################################
################################################################################
################################################################################
################################################################################
################################################################################
################################################################################
################################################################################
################################################################################


    #X_plot = np.linspace(-5, 10, 1000)[:, np.newaxis]

    #true_dens = (0.3 * norm(-2, 1).pdf(X_plot[:, 0])
    #             + 0.7 * norm(5, 1).pdf(X_plot[:, 0]))

    #fig, ax = plt.subplots()
    #ax.fill(X_plot[:, 0], true_dens, fc='black', alpha=0.2,
    #        label='input distribution')


    #kde_ = KernelDensity(kernel='gaussian', bandwidth=0.5)
    #kde_.fit(X)

    #filename = 'finalized_model.sav'
    #joblib.dump(kde_, filename)
     
    ## some time later...
    #kde = joblib.load(filename)

    #log_dens = kde.score_samples(X_plot)

    ##for cluster in clusters:
    ##    ax.plot(X_plot[:, 0], np.exp(log_dens), '-')

    #ax.plot(X_plot[:, 0], np.exp(log_dens), '-')

    #famous_colors = ['r', 'g', 'b', 'black','brown', 'slateblue', 'salmon', 'gold', 'gray', 'orange', 'violet' , 'darkgreen']


    #ax.text(6, 0.38, "N={0} points".format(N))

    #ax.legend(loc='upper left')

    ##ax.plot(X[:, 0], -0.005 - 0.01 * np.random.random(X.shape[0]), '+k')
    #color_pos = 0
    #for cluster in clusters:
    #    ax.plot(cluster.points_, -0.005 - 0.01 * np.random.random(len(cluster.points_)), marker = '+' , color = famous_colors[color_pos], linewidth=0)
    #    color_pos += 1

    #ax.set_xlim(-6, 9)
    #ax.set_ylim(-0.02, 0.4)
    #plt.show()        

    #color_pos   = 0
    #marker      = ['+', 'o', '.',  '+' ,  'o', '.', '+', 'o', '.']
    #marker_pos  = -1
    #for cluster in clusters:
    #    c_pos = color_pos % 12
    #    if c_pos == 0:
    #        marker_pos += 1
        
    #    plt.plot(cluster.points_[:, 0], cluster.points_[:, 1], marker = marker[marker_pos] , color = famous_colors[c_pos], linewidth=0)
    #    color_pos   += 1

    #plt.text(0, 0, "N={0} clusters".format(len(clusters)))
    #plt.show()

  
    #max_iter    = config['max_iter']
    #prob_th     = config['prob_th']

    ##...........................................................................
    #N = 1000
    #np.random.seed(1)
    ##g1 = np.random.normal(-2, 1, int(0.3 * N) )
    ##g2 = np.random.normal(5, 1, int(0.7 * N) )

    #g1 = np.random.multivariate_normal([-0.5, -0.5], [[1, 0],[0, 1]], N)
    ##plt.plot(g1[:, 0], g1[:, 1], '.')
    ##plt.show()

    #g2 = np.random.multivariate_normal([5, 6], [[1, 0.5],[0.5, 1]], N)
    ##plt.plot(g2[:, 0], g2[:, 1], '.')
    ##plt.show()

    #X = np.concatenate( (g1, g2) )[:, np.newaxis]

    #samples = X


    #famous_colors = ['r', 'g', 'b', 'black','brown', 'slateblue', 'salmon', 'gold', 'gray', 'orange', 'violet' , 'darkgreen']

    ##  initializer whole clusters .............................................
    #clusters = []
    #for sample in samples:
    #    clusters.append(KDECluster(sample))
         
    #iter    = 0
    #c_flag  = False

   
    ## until max iter or convergence condition is satisfied
    #while iter < max_iter and c_flag != True:
    #    i   = iter / max_iter
    #    th  = cgrow(i, prob_th)

    #    c_flag  = True
         
    #    n_clusters      = len(clusters)
    #    group_flag      = np.zeros(n_clusters)

    #    clusters_temp   = []

    #    for i in range(n_clusters):
    #        if group_flag[i] == 0:
    #            group_flag[i]    = 1
    #            for j in range(n_clusters):
    #                if i != j and group_flag[j] == 0:
    #                    score = clusters[i].compare(clusters[j]) 
    #                    #print(score)
    #                    if score > th:
    #                        clusters[i].points_ = np.concatenate( (clusters[i].points_, clusters[j].points_ ))
    #                        group_flag[j]       = 1
    #                        c_flag              = False

    #            if c_flag == False:
    #                clusters[i].kde_.fit(clusters[i].points_)

    #            clusters_temp.append(clusters[i])

    #    clusters = clusters_temp
    #    iter    += 1


    #test_clusters = []

    #for sample in samples:
    #    win     = 0
    #    maxp    = 0
    #    for i in range (len(clusters)):
    #        score = clusters[i].kde_.score_samples(sample)
    #        if maxp < score:
    #            maxp = score
    #            win  = i

    #    test_clusters.append(win)
    

    #color_pos   = 0
    #marker      = ['+', 'o', '.',  '+' ,  'o', '.', '+', 'o', '.']
    #marker_pos  = -1

    #for i in range ( len( samples ) ):
    #    c_pos = test_clusters[i] % 12
    #    marker_pos = int(test_clusters[i] / 12)
               
    #    plt.plot(samples[i][:,0], samples[i][:, 1], marker = marker[marker_pos] , color = famous_colors[c_pos], linewidth=0)
        

    #plt.text(0, 0, "N={0} clusters".format(len(clusters)))
    #plt.show()

   