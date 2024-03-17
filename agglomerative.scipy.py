import datetime
import pandas as pd
from gdcm.data.load_data import FeaturesData
from pathlib import Path
import os
from scipy.spatial.distance import pdist
from scipy.cluster.hierarchy import *
from gdcm.data.preprocess import preprocess_features
import numpy as np
from sklearn.mixture import GaussianMixture
from sklearn.cluster import KMeans, SpectralClustering, AffinityPropagation, DBSCAN, AgglomerativeClustering
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score
from gdcm.common.utils import load_a_dict, save_a_dict, print_the_evaluated_results
import time


class ClusteringEstimators:

    def __init__(self, algorithm_name, n_clusters, n_init):

        self.estimator = None
        self.algorithm_name = algorithm_name.lower()
        self.n_clusters = n_clusters
        self.n_init = n_init

        self.ari = None
        self.inertia = None
        self.clusters = None
        self.centroids = list()
        self.data_scatter = None

    @staticmethod
    def compute_data_scatter(m):
        return np.sum(np.power(m, 2))

    def compute_inertia(self, m, ):
        return sum(
            [np.sum(np.power(m[np.where(self.clusters == k)] - self.centroids[k], 2)) for k in range(self.n_clusters)]
        )

    def instantiate_estimator_with_parameters(self, ):

        # Methods based on given n_clusters:
        # K-Means:
        if self.algorithm_name == "km_clu":
            self.estimator = KMeans(
                n_clusters=self.n_clusters,
                init="random", max_iter=100, n_init=self.n_init,
            )
            print(
                "Instantiate K-Means Clustering."
            )

        # Gaussian Mixture:
        elif self.algorithm_name == "gm_clu":
            self.estimator = GaussianMixture(
                n_components=self.n_clusters,
                covariance_type="full",
                init_params="random",
            )
            print(
                "Instantiate Gaussian Mixture Clustering."
            )

        # Spectral (no predict() method to tune hyperparams):
        elif self.algorithm_name == "s_clu":
            self.estimator = SpectralClustering(
                n_clusters=self.n_clusters,
                n_init=self.n_init, gamma=1.0, affinity="rbf",
            )
            print(
                "Instantiate Spectral Clustering."
            )

        # Agglomerative (no predict() method to tune hyperparams):
        elif self.algorithm_name == "a_clu":
            self.estimator = (AgglomerativeClustering(
                n_clusters=self.n_clusters,
                metric="euclidean", linkage="average",
            ))
            print(
                "Instantiate Agglomerative Clustering."
            )

        # Methods based on automatic determination of n_clusters:
        # DBSCAN (no predict() method to tune hyperparams):
        elif self.algorithm_name == "dbs_clu":
            self.estimator = DBSCAN(
                eps=5e-1, min_samples=5, p=2,
            )
            print(
                "Instantiate DBSCAN Clustering."
            )
        else:
            assert False, "Undefined clustering model."

        return self.estimator

    def fit_estimator(self, x, y):

        print(
            "Fitting and testing of " + self.algorithm_name
        )

        self.clusters = self.estimator.fit_predict(x, y)
        self.centroids = [x[np.where(self.clusters == k)].mean(axis=0) for k in range(self.n_clusters)]
        self.centroids = np.asarray(self.centroids)
        self.inertia = self.compute_inertia(m=x)
        self.data_scatter = self.compute_data_scatter(m=x)
        if y is not None:
            self.ari = adjusted_rand_score(y, self.clusters)
        print(
            f"\n ARI = {self.ari:.3f} Inertia = {self.inertia:.3f} \n"
        )
        return self.clusters

K = [5, 15]
V = [2, 5, 10, 15, 20, 200, 2000]
ALPHA = [0.3, 0.6, 0.9]
for k in K:
    for v in V:
        for alpha in ALPHA:
            data_path1 = Path("D:/PycharmProjects/NGDC_method/Datasets")
            data_name = f'n=1000_k={k}_v={v}_alpha={alpha}'

            if "n=" in data_name or "k=" in data_name or "v=" in data_name:
                synthetic_data = True
            else:
                synthetic_data = False

            results = {}
            for repeat in range(1, 11):

                repeat = str(repeat)
                results[repeat] = {}

                from gdcm.algorithms.clustering_methods_competitors import ClusteringEstimators

                if synthetic_data is True:
                    dire = "F\synthetic"
                    dn = data_name + "_" + repeat

                else:
                    dire = "F"
                    dn = data_name

                data_path = os.path.join(data_path1, dire)
                fd = FeaturesData(name=dn, path=data_path)

                x, xn, y_true = fd.get_dataset()
                x = preprocess_features(x=x, pp="mm")
                results[repeat]['y_true'] = y_true

                n_clusters = len(np.unique(y_true))

                data_dist = pdist(x, metric='minkowski', p=2)
                data_linkage = linkage(data_dist, method='average')

                clusters = fcluster(data_linkage, t=n_clusters, criterion='maxclust')

                if xn.shape[0] != 0:
                    xn = preprocess_features(x=xn, pp="mm")


                # instantiate and fit
                start = time.process_time()
                # cu = ClusteringEstimators(
                #     algorithm_name='a_clu',
                #     n_clusters=5,
                #     n_init=10
                # )

                #cu.instantiate_estimator_with_parameters()
                y_pred = fcluster(data_linkage, t=n_clusters, criterion='maxclust')
                end = time.process_time()
                # save results and logs
                results[repeat]['y_pred'] = y_pred
                results[repeat]['time'] = end-start
                results[repeat]['inertia'] = 0
                results[repeat]['data_scatter'] = 0

            save_a_dict(
                    a_dict=results, name=data_name + "_" + 'a_clu'+'_' + f"p={2}", save_path="C:/Users/Tigran/PycharmProjects/NGDC_method/Results/Pickle_a_clu_p=2"
                )
            print_the_evaluated_results(results)
            print(data_name)
            #print(results.keys())