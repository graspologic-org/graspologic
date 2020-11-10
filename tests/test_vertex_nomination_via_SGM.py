import pytest
import numpy as np
from graspologic.simulations import er_np
from graspologic.nominate import VNviaSGM

np.random.seed(0)


class TestGMP:
    @classmethod
    def setup_class(cls):
        cls.vnsgm = VNviaSGM()

    def test_VNviaSGM_inputs(self):
        with pytest.raises(ValueError):
            VNviaSGM(h=-1)
        with pytest.raises(ValueError):
            VNviaSGM(h=1.5)
        with pytest.raises(ValueError):
            VNviaSGM(ell=-1)
        with pytest.raises(ValueError):
            VNviaSGM(ell=1.5)
        with pytest.raises(ValueError):
            VNviaSGM(R=-1)
        with pytest.raises(ValueError):
            VNviaSGM(R=1.5)

        with pytest.raises(ValueError):
            VNviaSGM().fit(
                0,
                np.identity((3, 4)),
                np.identity((4, 4)),
                np.arange(2),
                np.arange(2),
            )
        with pytest.raises(ValueError):
            VNviaSGM().fit(
                0,
                np.identity((4, 4)),
                np.identity((3, 4)),
                np.arange(2),
                np.arange(2),
            )
        with pytest.raises(ValueError):
            VNviaSGM().fit(
                0,
                np.identity((4, 4)),
                np.identity((4, 4)),
                np.arange(2),
                np.arange(3),
            )

    def test_vn_algorithm(self):
        n = 20
        nseeds = 4
        g1 = er_np(n=n, p=.5)
        node_shuffle = np.random.permutation(n)
        g2 = g1[np.ix_(node_shuffle, node_shuffle)]

        kklst = [(xx, yy) for xx, yy in zip(node_shuffle, np.arange(len(node_shuffle)))]
        kklst.sort(key=lambda x: x[0])
        kklst = np.array(kklst)

        vnsgm = VNviaSGM()

        for voi in range(nseeds + 1, n):
            nomlst = vnsgm.fit_predict(
                voi, g1, g2, seedsA=kklst[0:nseeds, 0], seedsB=kklst[0:nseeds, 1]
            )
            if nomlst is not None:
                assert nomlst[0][0] == kklst[np.where(kklst[:, 0] == voi)[0][0], 1]
