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

    def _get_AB(self):
        A = [
            [0, 0, 0, 1, 0, 1, 0, 1, 1, 0, 0, 1, 0, 1, 0],
            [0, 0, 1, 1, 1, 0, 0, 0, 1, 0, 0, 1, 1, 0, 0],
            [0, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 1],
            [1, 1, 0, 0, 1, 0, 1, 1, 0, 1, 0, 0, 0, 1, 0],
            [0, 1, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 1, 1, 0],
            [1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0],
            [0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0],
            [1, 0, 0, 1, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0],
            [1, 1, 1, 0, 1, 0, 0, 0, 0, 0, 1, 0, 1, 1, 0],
            [0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 1, 1, 1],
            [0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 1, 0, 0, 0],
            [1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 1, 0],
            [0, 1, 0, 0, 1, 1, 0, 0, 1, 1, 0, 1, 0, 0, 1],
            [1, 0, 0, 1, 1, 0, 0, 0, 1, 1, 0, 1, 0, 0, 1],
            [0, 0, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1, 1, 0],
        ]

        B = [
            [0, 1, 1, 1, 0, 1, 0, 1, 0, 0, 0, 0, 0, 1, 1],
            [1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0],
            [1, 0, 0, 1, 1, 0, 1, 1, 0, 1, 0, 1, 0, 0, 0],
            [1, 0, 1, 0, 0, 0, 1, 1, 0, 0, 1, 0, 0, 0, 0],
            [0, 1, 1, 0, 0, 0, 1, 0, 1, 0, 1, 0, 0, 1, 0],
            [1, 0, 0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 1],
            [0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 1, 0, 0, 1, 1],
            [1, 0, 1, 1, 0, 0, 0, 0, 0, 1, 1, 0, 0, 1, 0],
            [0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1],
            [0, 0, 1, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 1, 0],
            [0, 0, 0, 1, 1, 0, 1, 1, 1, 0, 0, 0, 1, 0, 1],
            [0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0],
            [1, 0, 0, 0, 1, 0, 1, 1, 0, 1, 0, 1, 0, 0, 0],
            [1, 0, 0, 0, 0, 1, 1, 0, 1, 0, 1, 0, 0, 0, 0],
        ]

        node_shuffle = [12, 5, 8, 4, 0, 14, 13, 1, 7, 2, 3, 10, 6, 11, 9]

        return A, B, node_shuffle

    def test_vn_algorithm(self):
        g1, g2, node_shuffle = self._get_AB()

        kklst = [(xx, yy) for xx, yy in zip(node_shuffle, np.arange(len(node_shuffle)))]
        kklst.sort(key=lambda x: x[0])
        kklst = np.array(kklst)

        vnsgm = VNviaSGM()

        voi = 5
        nseeds = 4

        nomlst = vnsgm.fit_predict(
            voi, g1, g2, seedsA=kklst[0:nseeds, 0], seedsB=kklst[0:nseeds, 1]
        )

        assert nomlst[0][0] == kklst[np.where(kklst[:, 0] == voi)[0][0], 1]
