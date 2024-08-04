import os
import time
import argparse
import numpy as np
import jax.numpy as jnp
from pathlib import Path
from types import SimpleNamespace
from gdcm.common.utils import load_a_dict, save_a_dict, print_the_evaluated_results
from gdcm.data.preprocess import preprocess_features, preprocess_adjacency
from sklearn.metrics import adjusted_rand_score
jnp.set_printoptions(suppress=True, precision=3, linewidth=140)

import warnings
warnings.filterwarnings('ignore')

configs = {
    "results_path": Path("D:/PycharmProjects/NGDC_method/Results"),
    "figures_path": Path("D:/PycharmProjects/NGDC_method/Figures"),
    "params_path": Path("D:/PycharmProjects/NGDC_method/Params"),
    "data_path": Path("D:/PycharmProjects/NGDC_method/Datasets"),
}

configs = SimpleNamespace(**configs)

if not configs.results_path.exists():
    configs.results_path.mkdir()

if not configs.figures_path.exists():
    configs.figures_path.mkdir()

if not configs.params_path.exists():
    configs.params_path.mkdir()


def args_parser(arguments):

    _run = arguments.run
    _tau = arguments.tau
    _mu_1 = arguments.mu_1
    _mu_2 = arguments.mu_2
    _n_init = arguments.n_init
    _pp = arguments.pp.lower()
    _verbose = arguments.verbose
    _max_iter = arguments.max_iter
    _init = arguments.init.lower()
    _step_size = arguments.step_size
    _batch_size = arguments.batch_size
    _range_len = arguments.range_len
    _p_value = arguments.p_value
    _n_repeats = arguments.n_repeats
    _n_clusters = arguments.n_clusters
    _update_rule = arguments.update_rule.lower()
    _data_name = arguments.data_name.lower()
    _centroids_idx = arguments.centroids_idx

    return _run, _tau, _mu_1, _mu_2, _n_init, _pp, _verbose, _max_iter, _init,\
        _step_size, _range_len, _p_value, _n_repeats, _n_clusters, \
        _update_rule, _data_name, _centroids_idx, _batch_size


if __name__ == "__main__":

    # all the string inputs will be converted to lower case.
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--data_name", type=str, default="IRIS",
        help="Dataset's name, e.g., IRIS, or Lawyers, or dd_fix_demo."
    )

    parser.add_argument(
        "--update_rule", type=str, default="ngdc",
        help="GDC update rule, at the moment, "
             "the three following up update methods are supported"
             "   1) vgdc: applies vanilla gdc algorithm (VGDC);"
             "   2) ngdc: applies gdc with Nestrov momentum algorithm (NGDC);"
             "   3) agdc: applies gdc with Adam algorithm (Adam GDC)."

    )

    parser.add_argument(
        "--run", type=int, default=0,
        help="Run the algorithm or load the saved"
             " parameters and reproduce the results."
    )

    parser.add_argument(
        "--pp", type=str, default="mm",
        help="Data preprocessing method:"
             " MinMax/Z-Scoring/etc."
    )

    parser.add_argument(
        "--n_clusters", type=int, default=5,
        help="Number of clusters."
    )

    parser.add_argument(
        "--verbose", type=int, default=1,
        help="An integer showing the level of verbosity, "
             "the higher the more verbose."
    )

    parser.add_argument(
        "--max_iter", type=int, default=10,
        help="An integer showing the maximum number of iterations "
             "(epochs in ANN terminology)"
    )

    parser.add_argument(
        "--step_size", type=float, default=1e-2,
        help="A float showing the step size or learning rate."
    )

    parser.add_argument(
        "--range_len", type=int, default=10,
        help="NOT USED!"
             "An Integer to form a two-element list, window, where the first and second elements, in respect,"
             " are the iteration numbers of the beginning and the end of the window to compute the average and"
             "the standard deviation of the gradient values and semi-positive definiteness of hessian matrix."
             "Since in our experiments, we could not find any meaningful pattern or stopping value for this parameter,"
             "thus we excluded from the methods' description in our paper."
             "However, for consistency issues we preserve it in our implementation and later version we will remove it."
    )

    parser.add_argument(
        "--init", type=str, default="user",
        help="One of the three possible type of seed initialization:"
             "1) random, 2)K-means++, 3)user. If it is set to user, "
             "the centroids_idx argument should be provided"
    )

    parser.add_argument(
        "--centroids_idx", type=list, default=None,
        help="If init argument is set to user, this item should be provided to determine"
             " the index of seeds for centroids initialization."
    )

    parser.add_argument(
        "--p_value", type=float, default=2.,
        help="A float showing the p_value in Minkowski distance metric."
             "If it is set to None, cosine distance metric will be applied."
    )

    parser.add_argument(
        "--tau", type=float, default=1e-4,
        help="A float, per-datapoint convergence threshold: "
             "the ratio between inertia and the data scatter"
    )

    parser.add_argument(
        "--n_repeats", type=int, default=10,
        help="Number of repeats of a data set or a specific distribution"
    )

    parser.add_argument(
        "--n_init", type=int, default=10,
        help="Number of repeats with different seed initialization to select "
             "the best results on a data set."
    )

    parser.add_argument(
        "--mu_1", type=float, default=45e-2,
        help="Exponential decay rate for the first moment estimates in Adam or"
             " decay rate for Nestrov GDC. "
             "Note: default for x-only data"
    )

    parser.add_argument(
        "--mu_2", type=float, default=95e-2,
        help="Exponential decay rate for the second moment estimates "
             "(squared gradients estimates) in Adam.  "
             "Note: default for x-only data"
    )

    parser.add_argument(
        "--batch_size", type=int, default=1,
        help="Batch size."
    )

    # add an arguments with_noise data
    args = parser.parse_args()

    run, tau, mu_1, mu_2, n_init, pp, verbose, max_iter,\
        init, step_size, range_len, p_value, n_repeats, n_clusters,\
        update_rule, data_name, centroids_idx, batch_size = args_parser(arguments=args)

    print(
        "Configuration: \n"
        f"  run: {run} \n"
        f"  data name: {data_name} \n"
        f"  pre-processing: {pp} \n"
        f"  step_size: {step_size} \n"
        f"  tau: {tau} \n"
        f"  mu_1: {mu_1} \n"
        f"  mu_2: {mu_2} \n"
        f"  p-value: {p_value} \n"
        f"  update_rule: {update_rule} \n"
        f"  range_len: {range_len} \n"
        f"  init: {init} \n"
        f"  max_iter: {max_iter} \n"
        f"  batch_size: {batch_size} \n"
    )

    configs.run = run

    # Adding some details for the sake of clarity in storing and visualization
    specifier = " -data: " + data_name + \
                " -update_rule: " + str(update_rule) + \
                " -step_size: " + str(step_size) + \
                " -range_len: " + str(range_len) + \
                " -p: " + str(p_value) + \
                " -tau: " + str(tau) + \
                " -max_iter" + str(max_iter) + \
                " -init: " + init + \
                " -mu_1: " + str(mu_1) + \
                " -mu_2: " + str(mu_2) + \
                " -batch_size: " + str(batch_size)

    configs.specifier = specifier
    configs.data_name = data_name
    configs.n_repeats = n_repeats

    # to add the repeat numbers to the data_name variable for synthetic data
    if "k=" in data_name or "v=" in data_name:
        synthetic_data = True
    else:
        synthetic_data = False

    if run == 1:
        results = {}
        for repeat in range(1, configs.n_repeats+1):
            repeat = str(repeat)
            results[repeat] = {}

            print(
                "clustering features_only data: " + data_name + " repeat=" + repeat, "\n"
            )

            from gdcm.data.load_data import FeaturesData
            from gdcm.algorithms.gradient_descent_clustering_methods_features import GDCMf

            if synthetic_data is True:
                dire = "F/synthetic"
                dn = data_name + "_" + repeat

            else:
                dire = "F"
                dn = data_name

            data_path = os.path.join(configs.data_path, dire)
            fd = FeaturesData(name=dn, path=data_path)

            x, xn, y_true = fd.get_dataset()
            results[repeat]['y_true'] = y_true

            x = preprocess_features(x=x, pp=pp)
            if xn.shape[0] != 0:
                xn = preprocess_features(x=xn, pp=pp)
            n_clusters = len(np.unique(y_true))

            # instantiate and fit
            start = time.process_time()
            gdcm = GDCMf(
                p=p_value,
                tau=tau,
                mu_1=mu_1,
                mu_2=mu_2,
                init=init,
                n_init=n_init,
                verbose=verbose,
                batch_size=batch_size,
                update_rule=update_rule,
                max_iter=max_iter,
                n_clusters=n_clusters,
                step_size=step_size,
                range_len=range_len,
                centroids_idx=centroids_idx,
            )
            y_pred = gdcm.fit(x=x, distance_fn=gdcm.minkowski_fn, y=y_true)
            end = time.process_time()

            # save results and logs
            results[repeat]['y_pred'] = y_pred
            results[repeat]['time'] = end-start
            results[repeat]['inertia'] = gdcm.best_inertia
            results[repeat]['data_scatter'] = gdcm.data_scatter
            results[repeat]['aris_history'] = gdcm.aris_history
            results[repeat]['grads_history'] = gdcm.grads_history
            results[repeat]['inertias_history'] = gdcm.inertias_history
            results[repeat]['centroids'] = centroids_idx
            results[repeat]['ari'] = adjusted_rand_score(y_true, y_pred)




            configs.stop_type = gdcm.stop_type

        current_max = -10000
        max_key = 0
        for key in results.keys():
            if results[key]['ari'] > current_max:
                current_max = results[key]['ari']
                max_key = key
        y_pred = results[str(max_key)]['y_pred']
        print(y_pred.tolist())
        print(results)

        # save results dict and configs
        print('END')
        print(results, configs.specifier, configs.results_path)
        save_a_dict(a_dict=results, name=configs.specifier, save_path=configs.results_path)

        print(results, configs.specifier, 'save_path=', configs.params_path)
        save_a_dict(
            a_dict=configs, name='ngdc'+f'{p_value}'+configs.data_name, save_path=configs.params_path
        )

        print("configs \n", configs.specifier, "\n")

        print("stop type:", configs.stop_type, "\n")

        print_the_evaluated_results(results=results)
        print(results)


    elif run != 1:

        # load results dict and configs
        results = load_a_dict(
            name=configs.specifier, save_path=configs.results_path
        )

        configs = load_a_dict(
            name=configs.specifier, save_path=configs.params_path
        )

        print("configs \n", configs.specifier, "\n")

        print("stop type:", configs.stop_type, "\n")

        print_the_evaluated_results(results=results)
