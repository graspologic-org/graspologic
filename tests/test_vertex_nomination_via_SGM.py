import pytest
import numpy as np
from graspologic.nominate import VNviaSGM
from graspologic.simulations import er_np

np.random.seed(0)


class TestGMP:
    def test_VNviaSGM_inputs(self):
        with pytest.raises(ValueError):
            VNviaSGM(order_voi_subgraph=-1)
        with pytest.raises(ValueError):
            VNviaSGM(order_voi_subgraph=1.5)
        with pytest.raises(ValueError):
            VNviaSGM(order_seeds_subgraph=-1)
        with pytest.raises(ValueError):
            VNviaSGM(order_seeds_subgraph=1.5)
        with pytest.raises(ValueError):
            VNviaSGM(n_init=-1)
        with pytest.raises(ValueError):
            VNviaSGM(n_init=1.5)

        with pytest.raises(ValueError):
            VNviaSGM().fit(
                np.random.randn(3, 4),
                np.random.randn(4, 4),
                0,
                [np.arange(2), np.arange(2)],
            )
        with pytest.raises(ValueError):
            VNviaSGM().fit(
                np.random.randn(4, 4),
                np.random.randn(3, 4),
                0,
                [np.arange(2), np.arange(2)],
            )
        with pytest.raises(ValueError):
            VNviaSGM().fit(
                np.random.randn(4, 4),
                np.random.randn(4, 4),
                0,
                [np.arange(2), np.arange(3)],
            )
        with pytest.raises(ValueError):
            VNviaSGM().fit(
                np.random.randn(4, 4),
                np.random.randn(4, 4),
                0,
                [np.arange(3), np.arange(3)],
                max_noms=0,
            )

    def test_vn_algorithm(self):
        g1 = er_np(n=50, p=0.3)
        node_shuffle = np.random.permutation(50)

        g2 = g1[node_shuffle][:, node_shuffle]

        kklst = [(xx, yy) for xx, yy in zip(node_shuffle, np.arange(len(node_shuffle)))]
        kklst.sort(key=lambda x: x[0])
        kklst = np.array(kklst)

        voi = 6
        nseeds = 4

        vnsgm = VNviaSGM()
        nomlst = vnsgm.fit_predict(
            g1, g2, voi, [kklst[0:nseeds, 0], kklst[0:nseeds, 1]]
        )

        assert nomlst[0][0] == kklst[np.where(kklst[:, 0] == voi)[0][0], 1]
