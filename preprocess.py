import jax.numpy as jnp
from sklearn.preprocessing import MinMaxScaler, \
    StandardScaler, QuantileTransformer, RobustScaler


def range_standardizer(x):
    """ Returns Range standardized Datasets set.
            Input: a numpy array, representing entity-to-feature matrix.
    """
    if not isinstance(x, jnp.ndarray):
        x = jnp.asarray(x)

    x_rngs = jnp.ptp(x, axis=0)
    x_means = jnp.mean(x, axis=0)

    x_r = jnp.divide(jnp.subtract(x, x_means), x_rngs)  # range standardization

    return jnp.nan_to_num(x_r)


def range_standardizer_(x_test, x_train):
    """ Returns Range standardized Datasets set.
    Input: a numpy array, representing entity-to-feature matrix.
    """
    if not isinstance(x_train, jnp.ndarray):
        x_train = jnp.asarray(x_train)
    if not isinstance(x_test, jnp.ndarray):
        x_test = jnp.asarray(x_test)

    x_rngs = jnp.ptp(x_train, axis=0)
    x_means = jnp.mean(x_train, axis=0)

    x_r = jnp.divide(jnp.subtract(x_test, x_means), x_rngs)  # range standardization

    return jnp.nan_to_num(x_r)


def zscore_standardizer(x):
    """ Returns Z-scored standardized Datasets set.
            Input: a numpy array, representing entity-to-feature matrix.
    """
    if not isinstance(x, jnp.ndarray):
        x = jnp.asarray(x)

    x_stds = jnp.std(x, axis=0)
    x_means = jnp.mean(x, axis=0)

    x_z = jnp.divide(jnp.subtract(x, x_means), x_stds)  # z-scoring

    return jnp.nan_to_num(x_z)


def zscore_standardizer_(x_test, x_train):
    """ Returns Z-scored standardized Datasets set.
            Input: a numpy array, representing entity-to-feature matrix.
    """
    if not isinstance(x_train, jnp.ndarray):
        x_train = jnp.asarray(x_train)
    if not isinstance(x_test, jnp.ndarray):
        x_test = jnp.asarray(x_test)

    x_stds = jnp.std(x_train, axis=0)
    x_means = jnp.mean(x_train, axis=0)

    x_z = jnp.divide(jnp.subtract(x_test, x_means), x_stds)  # z-scoring

    return jnp.nan_to_num(x_z)


def quantile_standardizer(x, out_dist):

    if not isinstance(x, jnp.ndarray):
        x = jnp.asarray(x)

    QT = QuantileTransformer(output_distribution=out_dist,)
    x_q = QT.fit_transform(x)

    return x_q, QT


def quantile_standardizer_(QT, x,):

    if not isinstance(x, jnp.ndarray):
        x = jnp.asarray(x)

    x_q = QT.fit_transform(x)

    return x_q


def minmax_standardizer(x):
    if not isinstance(x, jnp.ndarray):
        x = jnp.asarray(x)
    x_mm = jnp.divide(jnp.subtract(x, x.min(axis=0)),
                     (x.max(axis=0) - x.min(axis=0)))
    return jnp.nan_to_num(x_mm)


def minmax_standardizer_(x_test, x_train):

    return jnp.divide(
        jnp.subtract(x_test, x_train.min(axis=0)),
        (x_train.max(axis=0) - x_train.min(axis=0))
    )


def robust_standardizer(x):
    if not isinstance(x, jnp.ndarray):
        x = jnp.asarray(x)
    RS = RobustScaler()
    x_rs = RS.fit_transform(x)
    return x_rs, RS


def robust_standardizer_(RS, x):
    if not isinstance(x, jnp.ndarray):
        x = jnp.asarray(x)
    x_rs = RS.fit_transform(x)
    return x_rs


def preprocess_features(x, pp):

    if not isinstance(x, jnp.ndarray):
        x = jnp.asarray(x)

    if pp == "rng":
        print("pre-processing:", pp)
        x = range_standardizer(x=x)
        print("Preprocessed data shape:", x.shape, )
    elif pp == "zsc":
        print("pre-processing:", pp)
        x = zscore_standardizer(x=x)
        print("Preprocessed data shape:", x.shape,)
    elif pp == "mm":  # MinMax
        #print("pre-processing:", pp)
        x = minmax_standardizer(x=x)
        #print("Preprocessed data shape:", x.shape,)
    elif pp == "rs":  # Robust Scaler (subtract median and divide with [q1, q3])
        print("pre-processing:", pp)
        x, rs_x = robust_standardizer(x=x)
        print("Preprocessed data shape:", x.shape,)
    elif pp == "qtn":  # quantile_transformation with Gaussian distribution as output
        x, qt_x = quantile_standardizer(x=x, out_dist="normal")
        print("Preprocessed data shape:", x.shape,)
    elif pp == "qtu":  # quantile_transformation with Uniform distribution as output
        x, qt_x = quantile_standardizer(x=x, out_dist="uniform")
        print("Preprocessed data shape:", x.shape,)
    elif pp is None:
        x = x
        print("No pre-processing")
    else:
        print("Undefined pre-processing")

    return x


def uniform_shift(p):
    """Uniform shift transform. """

    N, V = p.shape
    cnt_rnd_interact = jnp.mean(p, axis=1)  # constant random interaction

    # Uniform method
    p_u = p - cnt_rnd_interact

    return p_u


def modularity_shift(p):
    """Modularity, i.e., random interaction shift transform. """
    p_row = jnp.sum(p, axis=0)
    p_col = jnp.sum(p, axis=1)
    p_tot = jnp.sum(p)
    rnd_interact = jnp.multiply(p_row, p_col) / p_tot  # random interaction formula
    p_m = p - rnd_interact
    return p_m


def preprocess_adjacency(p, pp):

    if not isinstance(p, jnp.ndarray):
        p = jnp.asarray(p)

    if pp == "us":
        print("pre-processing:", pp)
        p = uniform_shift(p=p)
        print("Preprocessed data shape:", p.shape, )
    elif pp == "ms":
        print("pre-processing:", pp)
        p = modularity_shift(p=p)
        print("Preprocessed data shape:", p.shape, )
    else:
        print("Undefined pre-processing")

    return p




