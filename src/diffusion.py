import numpy as np
import logging
import scipy.sparse as sp
from scipy.sparse.linalg import spsolve

import src.im_utils as im_utils
import src.adj_utils as adj_utils


def graph_prop(adj, gappy_tens, omega, thresh=0.001, iterative=True, verbose=True):
    """
    Given an adjacency matrix, a matrix of entries and a mask (omega) denoting which entries have been observed, use
    graph-based diffusion to propagate and return a completed matrix/tensor.
    Can be solved either iteratively or analytically when `iterative' is True/False respectively.

    - If `gappy_tens' is one-dimensional, we flatten into a vector (i.e. we consider data as matrix with a single band)
    - If `gappy_tens' is two-dimensional, we flatten into a vector (i.e. we consider data as matrix with a single band)
    - If `gappy_tens' is three-dimensional, we flatten the first two dimensions and use the last as the bands

    adj: adjacency matrix (unweighted and undirected), scipy.sparse matrix
    gappy_tens: array, np.ndarray (two-dimensional or three-dimensional array of entries)
    omega: array, np.ndarray (1 if observed, 0 if missing)
    thresh: float (only used if `iterative' is True)
    iterative: bool, if True then we solve using diffusion equations if False we solve for steady state using analytical
    solution (not necessarily always faster)
    """
    if verbose:
        logging.info(f"Completing array of shape {gappy_tens.shape} using graph-based diffusion")
    if omega.shape == gappy_tens.shape:
        if omega.ndim == 1:
            pass  # no check necessary, we assume the only dimension is spatial
        elif omega.ndim == 2:
            pass  # no check necessary, we assume both dimensions are spatial
        elif omega.ndim == 3:
            # in this case we assume last dim is spectral
            # we require bands to go missing together (i.e. omega the same for each band)
            # to check, sum in axis=-1 and check that there are only two unique values (or 1 if all missing/observed)
            if not len(np.unique(omega.sum(axis=-1).flatten())) in [1, 2]:
                msg = f"mask with values missing in some but not all bands not yet supported"
                logging.error(msg)
                raise ValueError(msg)
            else:
                omega = omega[..., 0]  # take one band (as we have assumed they are all the same)
        else:
            msg = f"Cannot handle `gappy_tens' of dimension {gappy_tens.ndim}"
            logging.error(msg)
            raise ValueError(msg)
    elif omega.shape != gappy_tens.shape[:-1]:
        msg = f"Shape of `omega' {omega.shape} must match first two dimensions of `gappy_tens' {gappy_tens.shape}"
        logging.error(msg)
        raise ValueError(msg)

    omega = (omega == 1).flatten()  # flattened bool array
    if gappy_tens.ndim == 1:
        observed = gappy_tens.copy().reshape(-1, 1)
    elif gappy_tens.ndim == 2:
        observed = gappy_tens.copy().reshape(-1, 1)
    elif gappy_tens.ndim == 3:
        observed = gappy_tens.copy().reshape(-1, gappy_tens.shape[2])
    # already checked for dims not in [1, 2, 3]

    adj, observed, reversing_mask = adj_utils.never_observed_check(adj, observed, with_reversing_mask=True)
    omega = omega[reversing_mask]

    # compute degree/laplacian matrix from adjacency
    degree = sp.diags(np.squeeze(np.asarray(adj.sum(axis=1))), 0, adj.shape, dtype=np.int16)
    laplacian = degree - adj

    # complete gappy tensor
    if iterative:
        completed = _iterative(
            observed,
            diffuser=(-laplacian.astype(np.float32)) / np.max(degree.data),
            omega=omega,
            thresh=thresh,
            verbose=verbose,
        )
    else:
        completed = _analytical(observed, laplacian, omega, verbose=verbose)

    # account for any filtered nodes (no edges - i.e. never observed)
    completed = im_utils.reverse_never_observed(completed, reversing_mask)

    return completed.reshape(gappy_tens.shape)  # return completed tensor in same shape as gappy tensor was provided


def _iterative(observed, diffuser, omega, thresh=0.001, verbose=True):
    completed = np.zeros_like(observed)
    completed[omega == 1] = observed[omega == 1]
    count = 0
    magn_break = None
    logging.info("Solving iteratively...")
    while True:
        delta = diffuser.dot(completed)
        # magn = np.mean(np.array(delta[~omega]) ** 2) ** 0.5
        magn = np.abs(np.array(delta[~omega]))
        if magn_break is None:
            magn_break = magn * thresh  # set the break threshold in first iteration
        else:
            magn_break = np.maximum(magn_break, magn * thresh)
        completed[~omega] += delta[~omega]
        if (count + 1) % 100 == 0:
            if verbose:
                None
                # logging.info(f"Count: {count + 1}, magn: {np.mean(magn < magn_break)}")
        if np.mean(magn < magn_break) > 1 - thresh:
            break
        # if np.isnan(magn):
        #     msg = "Magnitude of update is nan"
        #     logging.info(msg)
        #     raise RuntimeError(msg)
        count += 1
    if verbose:
        logging.info(f"Solved! ({count + 1} iterations)")
    return completed


def _analytical(observed, laplacian, omega, verbose):
    completed = np.zeros_like(observed)
    completed[omega == 1] = observed[omega == 1]
    if verbose:
        logging.info("Solving analytically...")
    laplacian += 1e-6 * sp.eye(laplacian.shape[0])  # avoid singularity
    completed[omega == 0] = (
        spsolve(
            laplacian[omega == 0][:, omega == 0],
            -laplacian[omega == 0][:, omega == 1].dot(observed[omega == 1]),
        )
    ).reshape(completed[omega == 0].shape)
    if verbose:
        logging.info("Solved!")
    return completed


def demo():
    gt = np.random.normal(0, 1, (50, 50, 3))
    mask = im_utils.rolling_stripes_mask((50, 50, 3), gap_npixels=2, repeat_npixels=10, roll_npixels=4)
    adj = sp.eye(7500, 7500) + sp.block_diag([adj_utils.udlr((50, 50))] * 3)
    return graph_prop(adj, gt.flatten(), mask.flatten())


if __name__ == "__main__":
    demo()