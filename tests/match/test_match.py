import pytest
import numpy as np
import math
import random
from graspy.match import FastApproximateQAP as FAQ
from graspy.match import SeededGraphMatching as SGM
from graspy.match import SinkhornKnopp as SK


class TestFAQ:
    @classmethod
    def setup_class(cls):
        cls.barycenter = FAQ()
        cls.rand = FAQ(n_init=100, init_method="rand")

    def test_FAQ_inputs(self):
        with pytest.raises(TypeError):
            FAQ(n_init=-1.5)
        with pytest.raises(ValueError):
            FAQ(init_method="random")
        with pytest.raises(TypeError):
            FAQ(max_iter=-1.5)
        with pytest.raises(TypeError):
            FAQ(shuffle_input="hey")
        with pytest.raises(TypeError):
            FAQ(eps=-1)
        with pytest.raises(TypeError):
            FAQ(gmp="hey")
        with pytest.raises(ValueError):
            FAQ().fit(np.random.random((3, 4)), np.random.random((3, 4)))
        with pytest.raises(ValueError):
            FAQ().fit(np.random.random((3, 3)), np.random.random((4, 4)))
        with pytest.raises(ValueError):
            FAQ().fit(np.random.random((3, 4)), np.random.random((3, 4)))
        with pytest.raises(ValueError):
            FAQ().fit(-1 * np.identity(3), -1 * np.identity(3))

    def _get_AB(self, qap_prob):
        with open("tests/match/qapdata/" + qap_prob + ".dat") as f:
            f = [int(elem) for elem in f.read().split()]

            # adjusting
        f = np.array(f[1:])
        n = int(math.sqrt(len(f) / 2))
        f = f.reshape(2 * n, n)
        A = f[:n, :]
        B = f[n:, :]
        return A, B

    def test_barycenter(self):
        A, B = self._get_AB("lipa20a")
        lipa20a = self.barycenter.fit(A, B)
        score = lipa20a.score_
        assert 3683 <= score < 3900

        A, B = self._get_AB("lipa20b")
        lipa20b = self.barycenter.fit(A, B)
        score = lipa20b.score_
        assert score == 27076

        A, B = self._get_AB("lipa30a")
        lipa30a = self.barycenter.fit(A, B)
        score = lipa30a.score_
        assert 13178 <= score < 13650

        A, B = self._get_AB("lipa30b")
        lipa30b = self.barycenter.fit(A, B)
        score = lipa30b.score_
        assert score == 151426

        A, B = self._get_AB("lipa40a")
        lipa40a = self.barycenter.fit(A, B)
        score = lipa40a.score_
        assert 31538 <= score < 32300

        A, B = self._get_AB("lipa40b")
        lipa40b = self.barycenter.fit(A, B)
        score = lipa40b.score_
        assert score == 476581

        A, B = self._get_AB("lipa50a")
        lipa50a = self.barycenter.fit(A, B)
        score = lipa50a.score_
        assert 62093 <= score < 63300

        A, B = self._get_AB("lipa50b")
        lipa50b = self.barycenter.fit(A, B)
        score = lipa50b.score_
        assert score == 1210244

    def test_rand(self):
        A, B = self._get_AB("chr12c")
        chr12c = self.rand.fit(A, B)
        score = chr12c.score_
        assert 11156 <= score < 13500

        A, B = self._get_AB("chr15a")
        chr15a = self.rand.fit(A, B)
        score = chr15a.score_
        assert 9896 <= score < 11500

class TestSGM:
    @classmethod
    def setup_class(cls):
        cls.barycenter = SGM()
        cls.rand = SGM(n_init=100, init_method="rand")

    def test_SGM_inputs(self):
        with pytest.raises(TypeError):
            SGM(n_init=-1.5)
        with pytest.raises(ValueError):
            SGM(init_method="random")
        with pytest.raises(TypeError):
            SGM(max_iter=-1.5)
        with pytest.raises(TypeError):
            SGM(shuffle_input="hey")
        with pytest.raises(TypeError):
            SGM(eps=-1)
        with pytest.raises(TypeError):
            SGM(gmp="hey")
        with pytest.raises(ValueError):
            SGM().fit(np.random.random((3, 4)), np.random.random((3, 4)),np.arange(2),np.arange(2))
        with pytest.raises(ValueError):
            SGM().fit(np.random.random((3, 3)), np.random.random((4, 4)),np.arange(2),np.arange(2))
        with pytest.raises(ValueError):
            SGM().fit(np.random.random((3, 4)), np.random.random((3, 4)),np.arange(2),np.arange(2))
        with pytest.raises(ValueError):
            SGM().fit(-1 * np.identity(3), -1 * np.identity(3),np.arange(2),np.arange(2))
        with pytest.raises(ValueError):
            SGM().fit(np.identity(3), np.identity(3), np.identity(3), np.arange(2))
        with pytest.raises(ValueError):
            SGM().fit(np.identity(3), np.identity(3), np.arange(1), np.arange(2))
        with pytest.raises(ValueError):
            SGM().fit(np.identity(3), np.identity(3), np.arange(5), np.arange(5))
        with pytest.raises(ValueError):
            SGM().fit(np.identity(3), np.identity(3), -1 * np.arange(2), -1 * np.arange(2))


    def _get_AB(self, qap_prob):
        with open("tests/match/qapdata/" + qap_prob + ".dat") as f:
            f = [int(elem) for elem in f.read().split()]

                    # adjusting
            f = np.array(f[1:])
            n = int(math.sqrt(len(f) / 2))
            f = f.reshape(2 * n, n)
            A = f[:n, :]
            B = f[n:, :]
            return A, B

    def test_barycenter_SGM(self):
        A, B = self._get_AB("chr12c")
        n = A.shape[0]
        pi = np.array([7, 5, 1, 3, 10, 4, 8, 6, 9, 11, 2, 12]) - [1] * n
        W1 = np.sort(random.sample(list(range(n)), 3))
        W2 = [pi[z] for z in W1]
        chr12c = self.barycenter.fit(A, B,W1,W2)
        score = chr12c.score_
        assert 11156 <= score < 20000

        W1 = np.sort(random.sample(list(range(n)), n-1))
        W2 = [pi[z] for z in W1]
        chr12c = self.barycenter.fit(A, B, W1, W2)
        score = chr12c.score_
        assert 11156 == score

        A, B = self._get_AB("chr15a")
        n = A.shape[0]
        pi = np.array([5,10,8,13,12,11,14,2,4,6,7,15,3,1,9]) - [1] * n
        W1 = np.sort(random.sample(list(range(n)), 3))
        W2 = [pi[z] for z in W1]
        chr15a = self.barycenter.fit(A, B, W1, W2)
        score = chr15a.score_
        assert 9896 <= score < 20000

        W1 = np.sort(random.sample(list(range(n)), n - 1))
        W2 = [pi[z] for z in W1]
        chr15a = self.barycenter.fit(A, B, W1, W2)
        score = chr15a.score_
        assert 9896 == score

    def test_rand_SGM(self):
        A, B = self._get_AB("chr12c")
        n = A.shape[0]
        pi = np.array([7, 5, 1, 3, 10, 4, 8, 6, 9, 11, 2, 12]) - [1] * n
        W1 = np.sort(random.sample(list(range(n)), 3))
        W2 = [pi[z] for z in W1]
        chr12c = self.rand.fit(A, B, W1, W2)
        score = chr12c.score_
        assert 11156 <= score < 12000

        A, B = self._get_AB("chr15a")
        n = A.shape[0]
        pi = np.array([5, 10, 8, 13, 12, 11, 14, 2, 4, 6, 7, 15, 3, 1, 9]) - [1] * n
        W1 = np.sort(random.sample(list(range(n)), 4))
        W2 = [pi[z] for z in W1]
        chr15a = self.rand.fit(A, B, W1, W2)
        score = chr15a.score_
        assert 9896 <= score < 10500




class TestSinkhornKnopp:
    @classmethod
    def test_SK_inputs(self):
        with pytest.raises(TypeError):
            SK(max_iter=True)
        with pytest.raises(ValueError):
            SK(max_iter=-1)
        with pytest.raises(TypeError):
            SK(epsilon=True)
        with pytest.raises(ValueError):
            SK(epsilon=2)

    def test_SK(self):

        # Epsilon = 1e-3
        sk = SK()
        P = np.asarray([[1, 2], [3, 4]])
        n = P.shape[0]
        Pt = sk.fit(P)

        f = np.concatenate((np.sum(Pt, axis=0), np.sum(Pt, axis=1)), axis=None)
        f1 = [round(x, 5) for x in f]
        assert (f1 == np.ones(2 * n)).all()

        # Epsilon = 1e-8
        sk = SK(epsilon=1e-8)
        P = np.asarray([[1.4, 0.2, 4], [3, 4, 0.7], [0.4, 6, 1]])
        n = P.shape[0]
        Pt = sk.fit(P)

        f = np.concatenate((np.sum(Pt, axis=0), np.sum(Pt, axis=1)), axis=None)
        f1 = [round(x, 5) for x in f]
        assert (f1 == np.ones(2 * n)).all()
