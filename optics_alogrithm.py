from sklearn.cluster import OPTICS
import numpy as np
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

data_path_synth = Path("D:/PycharmProjects/NGDC_method/Datasets/F/synthetic")
data_path = Path("D:/PycharmProjects/NGDC_method/Datasets")

p = int(input('введите пи вэлью: '))
alpha = float(input('введите альфа: '))
for k in [5, 15]:
    for v in [2, 5, 10, 15, 20, 200, 2000]:
        data_name = f'n=1000_k={k}_v={v}_alpha={alpha}'
        results = {}
        print(f'new_dataset!: {data_name}')
        for repeat in range(1, 11):
            ari = []
            potential_labels = []
            anotherloop = {}
            repeat = str(repeat)
            results[repeat] = {}
            for i in range(1, 11):
                anotherloop[i] = {}
                # data_path = os.path.join(data_path1, dire)
                # data_name = 'brtiss'

                fd_syntch = FeaturesData(name=data_name + '_' + repeat, path=data_path_synth)

                fd = FeaturesData(name=data_name, path=data_path)

                X, xn, y_true = fd_syntch.get_dataset()
                X = preprocessing.MinMaxScaler().fit_transform(X)
                clustering = OPTICS(p=p)
                # X = preprocess_features(x=X, pp='mm')
                # create dataset using make_blobs from sklearn datasets
                y_pred = clustering.fit(X).labels_

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
            #     save_path=f'D:/PycharmProjects/NGDC_method/Results/14_07_24_Pickle_Ind_km_cluv2_p={p}'
            # )
        print(type(print_the_evaluated_results(results)))
        with open(f'D:/PycharmProjects/NGDC_method/Results/optics_res/synth/optics_{data_name}_{p}_results.txt', 'a') as f:
            f.write(print_the_evaluated_results(results))
            print(data_name)
            f.write(data_name)
            f.write('\n')
            f.write('\n')
