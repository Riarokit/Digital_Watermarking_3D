"""Numerical and extraction regressions for the three correspondence modes."""
import unittest
from unittest.mock import patch

import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import maximum_bipartite_matching, min_weight_full_bipartite_matching
from scipy.optimize import linear_sum_assignment
from scipy.spatial import cKDTree

import DW1_X1_func as x1
import DW1_ELZ_func as elz
from modules.assignment import hk_matching, auction_assignment


class CorrespondenceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        rng = np.random.RandomState(123)
        cls.points = np.r_[rng.normal(size=(80, 3)), rng.normal(size=(80, 3)) + 8]
        cls.labels = np.repeat([0, 1], 80)
        cls.attacked = cls.points + rng.normal(scale=.08, size=cls.points.shape)
        cls.permutation = rng.permutation(len(cls.points))
        cls.profile_patch = patch.object(x1, "ONE_TO_ONE_PROFILE_LOGS", False)
        cls.profile_patch.start()

    @classmethod
    def tearDownClass(cls):
        cls.profile_patch.stop()

    def test_hk_cardinality_on_feasible_and_infeasible_rectangular_graphs(self):
        rng = np.random.RandomState(42)
        for n, m in ((5, 3), (8, 8), (30, 50), (50, 30)):
            for density in (.01, .05, .2, .8):
                for _ in range(5):
                    graph = csr_matrix(rng.rand(n, m) < density)
                    actual = hk_matching(graph.indptr, graph.indices, m)
                    reference = maximum_bipartite_matching(graph, perm_type="column")
                    self.assertEqual(np.sum(actual >= 0), np.sum(reference >= 0))
                    self.assertEqual(len(np.unique(actual[actual >= 0])), np.sum(actual >= 0))
                    for row in np.flatnonzero(actual >= 0):
                        self.assertTrue(graph[row, actual[row]])

    def test_auction_cost_bound_and_single_candidate(self):
        rng = np.random.RandomState(42)
        for n, m in ((8, 8), (20, 30)):
            costs = rng.rand(n, m)
            columns = np.tile(np.arange(m, dtype=np.int64), (n, 1))
            assigned, _ = auction_assignment(costs, columns, m, 1e-4)
            r, c = linear_sum_assignment(costs)
            gap = costs[np.arange(n), assigned].sum() - costs[r, c].sum()
            self.assertEqual(len(np.unique(assigned)), n)
            self.assertGreaterEqual(gap, -1e-12)
            self.assertLessEqual(gap, n * 1e-4 + 1e-12)
        assigned, _ = auction_assignment(
            np.ones((2, 1)), np.array([[0], [1]], dtype=np.int64), 2, .01
        )
        np.testing.assert_array_equal(assigned, [0, 1])

    def test_hk_keeps_first_feasible_prefix_and_original_exact_objective(self):
        # Every row initially sees the SAME few received points: low prefixes
        # cannot match all 20 rows, so this exercises candidate expansion.
        rng = np.random.RandomState(42)
        original = rng.rand(20, 3) * .01
        received = np.column_stack((np.arange(1., 21.), rng.rand(20), rng.rand(20)))
        tree = cKDTree(received)
        rows = np.arange(20)
        counts = (2, 4, 8, 16, 32)
        r, actual, chosen, profile = x1._solve_one_to_one_cluster(tree, original, 0, rows, counts, "hk")
        self.assertGreater(chosen, 16)
        self.assertTrue(any(not attempt[1] for attempt in profile["attempts"]))
        distances, candidates = tree.query(original, k=20)
        graph, pool = x1._one_to_one_candidate_graph(distances, candidates)
        local_rows, local_columns = min_weight_full_bipartite_matching(graph)
        expected = np.empty(20, dtype=int)
        expected[local_rows] = pool[local_columns]
        np.testing.assert_array_equal(actual, expected)

    def test_all_modes_unique_order_independent_and_same_serial_parallel(self):
        received = self.attacked[self.permutation]
        for mode in x1.ONE_TO_ONE_MODES:
            with self.subTest(mode=mode):
                serial = x1.match_point_cloud_one_to_one(received, self.points, self.labels, mode=mode, workers=1, verbose=False)
                parallel = x1.match_point_cloud_one_to_one(received, self.points, self.labels, mode=mode, workers=2, verbose=False)
                unshuffled = x1.match_point_cloud_one_to_one(self.attacked, self.points, self.labels, mode=mode, workers=1, verbose=False)
                np.testing.assert_array_equal(serial[0], parallel[0])
                np.testing.assert_array_equal(serial[0], unshuffled[0])
                self.assertTrue(serial[1].all())
                ids = cKDTree(received).query(serial[0])[1]
                for c in np.unique(self.labels):
                    self.assertEqual(len(np.unique(ids[self.labels == c])), 80)

    def test_small_clusters_and_one_candidate(self):
        for size in (1, 2, 3):
            original = np.column_stack((np.arange(float(size)), np.zeros((size, 2))))
            received = np.r_[original, [[10., 1., 1.]]]
            for mode in x1.ONE_TO_ONE_MODES:
                with self.subTest(size=size, mode=mode):
                    aligned, valid, _, _ = x1.match_point_cloud_one_to_one(
                        received[::-1], original, np.zeros(size), mode=mode,
                        candidate_counts=(1,), verbose=False,
                    )
                    np.testing.assert_array_equal(aligned, original)
                    self.assertTrue(valid.all())

    def test_fallback_for_count_reduction_and_infeasible_candidates(self):
        original = np.array([[0., 0., 0.], [.01, 0., 0.], [.02, 0., 0.]])
        for received in (np.array([[0., 0., 0.], [1., 0., 0.], [2., 0., 0.]]), original[:2]):
            expected = x1.match_point_cloud_to_original(received, original, distance_factor=30, verbose=False)
            for mode in x1.ONE_TO_ONE_MODES:
                aligned, valid = x1.align_attacked_vertices_to_original(
                    received, original, labels=np.zeros(3), correspondence_mode=mode,
                    one_to_one_candidates=(1,), coordinate_distance_factor=30, verbose=False,
                )
                np.testing.assert_array_equal(aligned, expected[0])
                np.testing.assert_array_equal(valid, expected[1])

    def test_distance_rejection_and_invalid_modes(self):
        original = np.array([[0., 0., 0.], [1., 0., 0.], [2., 0., 0.]])
        for mode in x1.ONE_TO_ONE_MODES:
            matched, valid, distances, threshold = x1.match_point_cloud_one_to_one(
                original + [100., 0., 0.], original, np.zeros(3), mode=mode, verbose=False,
            )
            self.assertFalse(valid.any())
            self.assertTrue(np.isnan(matched).all())
            self.assertTrue((distances > threshold).all())
        with self.assertRaises(ValueError):
            x1.align_attacked_vertices_to_original(original, original, correspondence_mode="1to1")
        with self.assertRaises(ValueError):
            x1.match_point_cloud_one_to_one(original, original, np.zeros(3), mode="unknown")
        with self.assertRaises(ValueError):
            x1.match_point_cloud_one_to_one(original * np.nan, original, np.zeros(3))

    def test_x1_and_dualband_extraction_with_all_modes(self):
        points, labels = self.points, self.labels
        bits = [0, 1, 1, 0, 1, 0, 0, 1]
        marked = x1.embed_watermark_x1(points, labels, bits, beta=1e-5, k=8)
        dual_marked = points + 1e-5 * x1.prepare_watermark_x1_dualband_displacement(points, labels, bits, k=8)
        feature = x1.dualband_geometry_feature(dual_marked)
        for mode in x1.ONE_TO_ONE_MODES:
            with self.subTest(mode=mode):
                recovered = x1.extract_watermark_x1(
                    marked[self.permutation], points, labels, len(bits), k=8,
                    correspondence_mode=mode, coordinate_distance_factor=30,
                )
                self.assertEqual(recovered, bits)
                recovered_dual = x1.extract_watermark_x1_dualband(
                    dual_marked[self.permutation], points, labels, len(bits), k=8,
                    correspondence_mode=mode, coordinate_distance_factor=30,
                    attack_reference_feature=feature,
                )
                self.assertEqual(recovered_dual, bits)

    def test_elzein_carrier_subset_extracts_with_each_mode(self):
        # Match only saved carriers against the full received vertex pool.
        points = self.points
        triangles = np.array([[i, i+1, i+2] for i in range(len(points)-2)])
        bits = [0, 1, 1, 0, 1, 0, 0, 1]
        marked, key = elz.embed_watermark_elzein_mesh(
            points, triangles, bits, a=1e-5, verbose=False,
            carrier_clusters=(tuple(range(40)), tuple(range(80, 120))),
        )
        for mode in x1.ONE_TO_ONE_MODES:
            with self.subTest(mode=mode):
                actual = elz.extract_watermark_elzein_mesh(
                    marked[self.permutation], points, triangles, key,
                    correspondence_mode=mode, coordinate_distance_factor=30,
                )
                np.testing.assert_array_equal(actual, bits)


if __name__ == "__main__":
    unittest.main()
