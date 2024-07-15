import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import os
import matplotlib.pyplot as plt
from sklearn import cluster
from sklearn import preprocessing
from sklearn.datasets import make_blobs
from pathlib import Path
from gdcm.data.load_data import FeaturesData
from sklearn import metrics
from gdcm.common.utils import load_a_dict, save_a_dict, print_the_evaluated_results
from gdcm.data.preprocess import preprocess_features
import jax
def _get_subkey():
    seed = np.random.randint(low=0, high=1e6, size=1)[0]
    key = jax.random.PRNGKey(seed)
    _, subkey = jax.random.split(key)
    return subkey

class KMeansClustering:
    def __init__(self, X, num_clusters, p):
        self.p_value = p
        self.k = num_clusters  # cluster number
        self.max_iterations = 100  # max iteration. don't want to run inf time
        self.num_examples, self.num_features = X.shape  # num of examples, num of features
        #self.plot_figure = True  # plot figure

    # randomly initialize centroids
    def initialize_random_centroids(self, x):
        subkey = _get_subkey()
        centroids_idx = jax.random.randint(
            subkey, minval=0, maxval=x.shape[0], shape=(k,))
        idx = centroids_idx
        centroids = x[idx, :] + 1e-6  # white noise to prevent zero-gradient if 1st centroid = 1st data point
        centroids = centroids + 1e-6
        # centroids = np.zeros((self.K, self.num_features))  # row , column full with zero
        # for k in range(self.K):  # iterations of
        #     centroid = X[np.random.choice(range(self.num_examples))]  # random centroids
        #     centroids[k] = centroid
        return centroids  # return random centroids

    # create cluster Function
    def create_cluster(self, X, centroids):
        clusters = [[] for _ in range(self.k)]
        # print(clusters)
        for point_idx, point in enumerate(X):

            # closest_centroid = np.argmin(np.power(np.sum(np.power(np.absolute(point - centroids), self.p_value)), 1 / self.p_value))
            # print(closest_centroid)
            closest_centroid = np.argmin(np.power(np.sum(np.power(np.absolute(point - centroids), self.p_value), axis=1), 1/self.p_value))
            clusters[closest_centroid].append(point_idx)

        return clusters

        # new centroids

    def calculate_new_centroids(self, cluster, X):
        centroids = np.zeros((self.k, self.num_features))  # row , column full with zero
        for idx, cluster in enumerate(cluster):
            new_centroid = np.mean(X[cluster], axis=0)  # find the value for new centroids
            centroids[idx] = new_centroid
        return centroids

    # prediction
    def predict_cluster(self, clusters, X):
        y_pred = np.zeros(self.num_examples)  # row1 fillup with zero
        for cluster_idx, cluster in enumerate(clusters):
            for sample_idx in cluster:
                y_pred[sample_idx] = cluster_idx
        return y_pred

    # plotinng scatter plot
    # def plot_fig(self, X, y):
    #     fig = px.scatter(X[:, 0], X[:, 1], color=y)
    #     fig.show()  # visualize

    # fit data
    def fit(self, X):
        centroids = self.initialize_random_centroids(X)  # initialize random centroids
        for _ in range(self.max_iterations):
            clusters = self.create_cluster(X, centroids)  # create cluster
            previous_centroids = centroids
            centroids = self.calculate_new_centroids(clusters, X)  # calculate new centroids
            diff = centroids - previous_centroids  # calculate difference
            if not diff.any():
                break
        y_pred = self.predict_cluster(clusters, X)  # predict function
        #if self.plot_figure:  # if true
        #    self.plot_fig(X, y_pred)  # plot function
        return y_pred


if __name__ == "__main__":
    data_path = Path("D:/PycharmProjects/NGDC_method/Datasets/F/synthetic")
    for p in [1, 2, 3, 4]:
        for k in [5, 15]:
            for v in [2, 5, 10, 15, 20, 200, 2000]:
                for alpha in [0.3, 0.6, 0.9]:
                    data_name = f'n=1000_k={k}_v={v}_alpha={alpha}'
                    results = {}

                    for repeat in range(1, 11):
                        ari = []
                        potential_labels = []
                        anotherloop = {}
                        repeat = str(repeat)
                        results[repeat] = {}
                        for i in range(1, 11):

                            anotherloop[i] = {}
                            # data_path = os.path.join(data_path1, dire)
                            fd = FeaturesData(name=data_name + '_' + repeat, path=data_path)
                            X, xn, y_true = fd.get_dataset()
                            X = preprocessing.MinMaxScaler().fit_transform(X)

                            # create dataset using make_blobs from sklearn datasets
                            Kmeans = KMeansClustering(X, k, p=p)
                            y_pred = Kmeans.fit(X)

                            anotherloop[i]['y_pred'] = y_pred

                            ari.append(metrics.adjusted_rand_score(y_true, y_pred))
                            potential_labels.append(y_pred)


                        results[repeat]['y_true'] = y_true
                        results[repeat]['y_pred'] = potential_labels[ari.index(max(ari))]
                        results[repeat]['time'] = 0
                        results[repeat]['inertia'] = 0
                        results[repeat]['data_scatter'] = 0
                        results[repeat]['aris_history'] = 0
                        results[repeat]['grads_history'] = 0
                        results[repeat]['inertias_history'] = 0

                    print_the_evaluated_results(results)
                    print(data_name)

                    # save_a_dict(
                    #     a_dict=results, name=data_name,
                    #     save_path=f'D:/PycharmProjects/NGDC_method/Results/Pickle_Ind_km_clu_p={p}'
                    # )
                    #
                    # with open(f'D:/PycharmProjects/NGDC_method/txt_results_km/results_p={p}v3.txt', 'a') as f:
                    #     f.write(print_the_evaluated_results(results))
                    #     print(data_name)
                    #     f.write(data_name)
                    #     f.write('\n')
                    #     f.write('\n')
