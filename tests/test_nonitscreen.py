import numpy as np
import pytest
import random

np.random.seed(10)

import graspy
from graspy.simulations import sbm
from graspy.subgraph import NonItScreen


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
        L_ind = L_ind[num[i] :]

        # Define vector where unedited entries will be removed as their probability is >1
        prob_mat = 2 * np.ones((1, N))

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


def p_data_test_shape():

    # Testing for correct p_data_test shape
    assert p_data_test.shape == (2, 200, 200)


def p_data_test_entries():

    # Checking that every entry in p_data_test is correct
    assert not (p_data_test[0][:20][:, :20] - 0.3).all()
    assert not (p_data_test[0][:20][:, 20:] - 0.2).all()
    assert not (p_data_test[0][20:][:, :20] - 0.2).all()
    assert not (p_data_test[0][20:][:, 20:] - 0.3).all()
    assert not (p_data_test[1][:20][:, :20] - 0.4).all()
    assert not (p_data_test[1][:20][:, 20:] - 0.2).all()
    assert not (p_data_test[1][20:][:, :20] - 0.2).all()
    assert not (p_data_test[1][20:][:, 20:] - 0.3).all()


def test_non_itss_mgc():

    # Testing mgc
    screen = NonItScreen("mgc", float("-inf"))
    screen.fit(data_test, y_label_test)

    cor_vals_samp = screen.corrs
    S_hat_samp = screen.fit_transform(data_test, y_label_test)

    # Testing if S_hat_samp has correct values and shape
    assert np.array_equal(S_hat_samp, data_test)

    # Testing cor_vals_samp shape
    assert cor_vals_samp.shape == (200, 1)


def test_non_itss_dcorr():

    # Redo for dcorr
    screen = NonItScreen("dcorr", float("-inf"))
    screen.fit(data_test, y_label_test)

    cor_vals_samp = screen.corrs
    S_hat_samp = screen.fit_transform(data_test, y_label_test)

    # Testing if S_hat_samp has correct values and shape
    assert np.array_equal(S_hat_samp, data_test)

    # Testing cor_vals_samp shape
    assert cor_vals_samp.shape == (200, 1)


def test_non_itss_rv():

    # Redo for rv
    screen = NonItScreen("rv", float("-inf"))
    screen.fit(data_test, y_label_test)

    cor_vals_samp = screen.corrs
    S_hat_samp = screen.fit_transform(data_test, y_label_test)

    # Testing if S_hat_samp has correct values and shape
    assert np.array_equal(S_hat_samp, data_test)

    # Testing cor_vals_samp shape
    assert cor_vals_samp.shape == (200, 1)


def test_non_itss_cca():

    # Redo for cca
    screen = NonItScreen("cca", float("-inf"))
    screen.fit(data_test, y_label_test)

    cor_vals_samp = screen.corrs
    S_hat_samp = screen.fit_transform(data_test, y_label_test)

    # Testing if S_hat_samp has correct values and shape
    assert np.array_equal(S_hat_samp, data_test)

    # Testing cor_vals_samp shape
    assert cor_vals_samp.shape == (200, 1)
