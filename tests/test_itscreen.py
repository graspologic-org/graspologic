import numpy as np
import random
from graspy.simulations import sbm
from graspy.subgraph import ItScreen

np.random.seed(10)


def data_generator(num_graphs, N, n, prob_tensor, percent_vec):

    # Getting the number of classes
    num_types = len(percent_vec)

    # Getting vector with the number of graphs in each class
    num = [int(num_graphs * a) for a in percent_vec]

    # Creating blank arrays for all of the returns
    data = np.zeros((num_graphs, N, N))
    y_label = np.zeros((num_graphs, 1))

    # Creates vector of random indices to randomly distribute graphs in tensor
    L_ind = random.sample(range(0, num_graphs), num_graphs)

    # Loop for creating the returns
    for i in range(num_types):

        # Create tensor that will contain all of the graphs of one type
        types = np.zeros((num[i], N, N))

        # Put all the graphs of one type into types
        for j in range(len(types)):
            types[j] = sbm(n=n, p=prob_tensor[i])

        # Assigns all of the graphs in types to random indices in data
        data[L_ind[: num[i]]] = types

        # Creates corresponding labels
        y_label[L_ind[: num[i]]] = int(i)

        # Gets rid of used indices
        L_ind = L_ind[num[i]:]

    return data, y_label


# Generate data
prob_tensor = np.zeros((2, 2, 2))
prob_tensor[0] = [[0.3, 0.2], [0.2, 0.3]]
prob_tensor[1] = [[0.4, 0.2], [0.2, 0.3]]
n = [20, 180]
percent_vec = np.asarray([0.50, 0.50])
data_test, y_label_test = data_generator(100, 200, n, prob_tensor, percent_vec)


def data_test_shape():

    # Testing for correct data_test shape
    assert data_test.shape == (100, 200, 200)


def data_test_entries():

    # Testing for correct number of matrices for each class in data_test
    y_bool = y_label_test.reshape(1, 100)[0].astype(bool)
    assert len(data_test[y_bool]) == 50
    assert len(data_test[~y_bool]) == 50


def y_label_test_shape():

    # Testing for correct y_label_test shape
    assert y_label_test.shape == (100, 1)


def y_label_test_entries():

    # Testing for correct number of labels for each class in y_label_test
    assert len(y_label_test[y_label_test == 1]) == 50
    assert len(y_label_test[y_label_test == 0]) == 50


def test_itss_mgc():

    # Testing mgc
    screen = ItScreen("mgc", 0.50, 20)
    screen.fit(data_test, y_label_test)

    cor_vals_samp = screen.corrs
    S_hat_samp = screen.fit_transform(data_test, y_label_test)

    # Testing cor_vals_samp shape
    assert cor_vals_samp.shape == (200, 1)

    # Testing cor_vals_samp elements
    assert (cor_vals_samp > 0).all()

    # Testing S_hat_samp elements
    assert S_hat_samp.shape[0] == 100


def test_itss_dcorr():

    # Testing dcorr
    screen = ItScreen("dcorr", 0.50, 20)
    screen.fit(data_test, y_label_test)

    cor_vals_samp = screen.corrs
    S_hat_samp = screen.fit_transform(data_test, y_label_test)

    # Testing cor_vals_samp shape
    assert cor_vals_samp.shape == (200, 1)

    # Testing cor_vals_samp elements
    assert (cor_vals_samp > 0).all()

    # Testing S_hat_samp elements
    assert S_hat_samp.shape[0] == 100


def test_itss_rv():

    # Testing rv
    screen = ItScreen("rv", 0.50, 20)
    screen.fit(data_test, y_label_test)

    cor_vals_samp = screen.corrs
    S_hat_samp = screen.fit_transform(data_test, y_label_test)

    # Testing cor_vals_samp shape
    assert cor_vals_samp.shape == (200, 1)

    # Testing cor_vals_samp elements
    assert (cor_vals_samp > 0).all()

    # Testing S_hat_samp elements
    assert S_hat_samp.shape[0] == 100


def test_itss_cca():

    # Testing cca
    screen = ItScreen("cca", 0.50, 20)
    screen.fit(data_test, y_label_test)

    cor_vals_samp = screen.corrs
    S_hat_samp = screen.fit_transform(data_test, y_label_test)

    # Testing cor_vals_samp shape
    assert cor_vals_samp.shape == (200, 1)

    # Testing cor_vals_samp elements
    assert (cor_vals_samp > 0).all()

    # Testing S_hat_samp elements
    assert S_hat_samp.shape[0] == 100
