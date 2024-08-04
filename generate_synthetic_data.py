import os, sys
import numpy as np
import pandas as pd
from pathlib import Path
from itertools import product
import matplotlib.pyplot as plt
from types import SimpleNamespace
from sklearn.datasets import make_blobs

np.set_printoptions(suppress=True, precision=3)


def clusters_cardinality(n, n_clusters, minimum=30):
    """
    :param n: Number entities
    :param n_clusters: Number of clusters
    :param minimum:  Minimum Number of entities in each cluster, default value = 30
    :return: clusters cardinality
    """

    remaining = n - n_clusters * minimum

    assert not n < np.abs(remaining), f"n = {n} < n_clusters * minmum ={n_clusters * minimum}"

    ur = [np.random.uniform() for _ in range(n_clusters)]
    ur.sort()

    tmp = []
    for k in range(n_clusters - 1):
        if k == 0:
            tmp.append(ur[k])
        else:
            tmp.append(ur[k + 1] - ur[k])

    cardinalities = [int(remaining * i + minimum) for i in tmp]
    cardinalities += [n - sum(cardinalities)]

    return cardinalities


def flat_ground_truth(ground_truth):
    """
    :param ground_truth: list, containing the clusters cardinality
                        (output of cluster cardinality from synthetic data generator)
    :return: list, the first one is the list of labels in an appropriate format
             for applying sklearn metrics.
    """
    k = 1
    interval = 1
    labels_true, labels_true_indices = [], []
    for v in ground_truth:
        for vv in range(v):
            labels_true.append(k)
        k += 1
        interval += v
    return np.asarray(labels_true)


def shuffler(n):
    idx = np.random.choice(range(0, n), replace=False, size=n)
    return idx


def generate_quantitative_features(n, v, n_clusters, alpha, cardinality, v_noise1):
    """
    :param n: Int, Number of Entries
    :param v: Int, Number of Features
    :param n_clusters: Int, Number of Clusters.
    :param alpha: float, [-1, 1], coefficient to control the clusters intermix (homogeneity of features btw clusters)
                        the smaller the 𝛼, the greater the chance that points from a cluster fall within.
    :param cardinality: list, cardinality of the clusters.
    :param v_noise1: Int, number of columns to insert noisy features as per the introduced
                    noise model1 regarding Renato, Makarenkov, Mirkin 2016 -page 348.
    :return: an entity-to-feature matrix with or without noise
    """

    # Quantitative features

    d1, d2 = -1, 1  # a range for mean
    d1_, d2_ = 0.025 * (d2 - d1), 0.05 * (d2 - d1)  # a range for covariance matrix

    x = np.zeros([n, v])  # empty entity-to-feature matrix

    interval = 0
    for k in range(n_clusters):
        mean = np.multiply(alpha, np.random.uniform(low=d1, high=d2, size=v))
        cov_ = np.random.uniform(low=d1_, high=d2_, size=(v, v))
        cov_ = np.add(cov_, cov_.T) / 2  # positive semidefinite
        cov = np.zeros([v, v])
        for i in range(v):
            for j in range(v):
                if i == j:
                    cov[i, j] = cov_[i, j]
        tmp = np.random.multivariate_normal(mean=mean, cov=cov, size=cardinality[k])
        row = 0
        for i in range(interval, cardinality[k] + interval):
            x[i, :] = tmp[row, :]  # np.random.multivariate_normal(mean=mean, cov=cov_)
            row += 1

        interval += cardinality[k]

    # Y with noise
    # Noise Model1 noisy features to insert
    max_features = np.max(x, axis=0)
    min_features = np.min(x, axis=0)
    noise1_ = np.random.uniform(low=min_features, high=max_features, size=(n, v))
    column_to_insert = np.random.choice(v, v_noise1, replace=False)
    noise1 = noise1_[:, column_to_insert]

    xn = np.concatenate((x, noise1), axis=1)

    # Noise Model2:
    # indices = list(np.random.randint(low=0, high=N, size=int(np.ceil(N * V_noise2))))
    # noises2 = np.random.uniform(low=0, high=1, size=(len(indices), V_noise2))
    # e = 0
    # for i in indices:
    #     Yn[i, :] = noises2[e, :]
    #     e += 1

    return x, xn


def generate_network_adjacency(n, cardinality, p_wth, p_btw, ):

    """
    :param cardinality: nodes list: list, node indices.
    :param p_wth:  float, [0,1], Probability for edge creation within the community.
    :param p_btw: float, [0,1], Probability for edge creation between the communities.
    :return: network adjacency matrix.
    """

    p = np.zeros([n, n])
    communities_structure = []

    intrv = 0
    for i in range(len(cardinality)):
        if i == 0:
            communities_structure.append(list(range(cardinality[i])))
        if i == len(cardinality) - 1:
            communities_structure.append(list(range(intrv, n)))
        elif 0 < i < len(cardinality) - 1:
            communities_structure.append(list(range(intrv, cardinality[i] + intrv)))
        intrv += cardinality[i]

    for k in communities_structure:
        for i in range(n):
            for j in range(i):
                if i in k and j in k:
                    if np.random.random() < p_wth:
                        p[i, j] = 1
                        p[j, i] = 1
                elif j in k and i not in k:
                    if np.random.random() < p_btw:
                        p[i, j] = 1
                        p[j, i] = 1

    return p


if __name__ == "__main__":

    n_repeats = 10
    N = {1000}  # {1000, 5000}
    K = {5, 15}  # {2, 5, 10, 20}
    V = {2, 5, 10, 15, 20, 200, 2000}  # {2, 5, 10, 50, 100, 1000}
    Alphas = set([0.3, 0.6, 0.9])  # \epsilon OR \alpha in the paper the smaller the more difficult

    settings = list(product(N, K, V, Alphas))
    for _ in settings:
        print(_)

    print("Number of data sets", len(settings)*n_repeats)

    # Generate and save synthetic data sets

    # Generate and save synthetic data sets

    # Generate and save synthetic data sets
    for setting in settings:
        n = setting[0]
        k = setting[1]
        v = setting[2]
        alpha = setting[3]
        cardinalities = clusters_cardinality(n, k)

        for r in range(1, n_repeats + 1):  # number of repeats

            # data_name: n=2000_k=10_v=10_alpha=0.5_44
            data_name = "n=" + str(n) + "_" + "k=" + str(k) + "_" + "v=" \
                        + str(v) + "_" + "alpha=" + str(alpha) + "_" + str(r)
            print(f"data_name: {data_name}")

            if not os.path.exists(f"C:/Users/Tigran/PycharmProjects/NGDC_method/Datasets/F/synthetic/"):
                os.mkdir(f"C:/Users/Tigran/PycharmProjects/NGDC_method/Datasets/F/synthetic")

            os.mkdir(f"C:/Users/Tigran/PycharmProjects/NGDC_method/Datasets/F/synthetic/" + data_name)

            x_, xn_ = generate_quantitative_features(n=n, v=v,
                                                         n_clusters=k, alpha=alpha,
                                                         cardinality=cardinalities,
                                                         v_noise1=int(np.floor(v / 2))
                                                         )
            y_ = flat_ground_truth(cardinalities)

            shuffled_idx = shuffler(n)

            x = x_[shuffled_idx, :]
            xn = xn_[shuffled_idx, :]
            y = y_[shuffled_idx]

            df_x = pd.DataFrame(data=x, )
            df_xn = pd.DataFrame(data=xn, )
            df_y = pd.DataFrame(data=y.reshape(-1, 1), )

            df_x.to_csv("C:/Users/Tigran/PycharmProjects/NGDC_method/Datasets/F/synthetic/" + data_name + "/data.csv", header=False, index=False)
            df_xn.to_csv("C:/Users/Tigran/PycharmProjects/NGDC_method/Datasets/F/synthetic/" + data_name + "/data_noise.csv", header=False, index=False)
            df_y.to_csv("C:/Users/Tigran/PycharmProjects/NGDC_method/Datasets/F/synthetic/" + data_name + "/labels.csv", header=False, index=False)
