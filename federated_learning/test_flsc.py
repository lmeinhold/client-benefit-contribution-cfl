import numpy as np

from federated_learning.flsc import FLSC


def test_get_new_cluster_identities_from_losses_multiple():
    losses = np.array([0.1, 0.6, 0.9, 0.8, 0.2])
    n = 3
    new_identities = FLSC._get_new_cluster_identities_from_losses(losses, n)
    assert len(new_identities) == n, f"Length of identities is not equal to n (is {len(new_identities)} instead)"
    assert np.array_equal(new_identities,
                          np.array([0, 4, 1])), f"New cluster identities are not as expected: {new_identities}"


def test_get_new_cluster_identities_from_losses_list():
    losses = [0.1, 0.6, 0.9, 0.8, 0.2]
    n = 3
    new_identities = FLSC._get_new_cluster_identities_from_losses(losses, n)
    assert len(new_identities) == n, f"Length of identities is not equal to n (is {len(new_identities)} instead)"
    assert np.array_equal(new_identities,
                          np.array([0, 4, 1])), f"New cluster identities are not as expected: {new_identities}"


def test_get_new_cluster_identities_from_losses_single():
    losses = np.array([0.1, 0.6, 0.9, 0.8, 0.2])
    n = 1
    new_identities = FLSC._get_new_cluster_identities_from_losses(losses, n)
    assert len(new_identities) == n, f"Length of identities is not equal to n (is {len(new_identities)} instead)"
    assert np.array_equal(new_identities,
                          np.array([0])), f"New cluster identities are not as expected: {new_identities}"
