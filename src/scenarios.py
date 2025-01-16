import numpy as np
import matplotlib.pyplot as plt

from get_inputs import sr, toa, ncep, aster
from smw_algorithm import compute_lst
import get_cloud_masks


def partial_b10(place_name, asset_id, landsat, missing_landsat, specific):
    b10 = toa.load_asset_b10(place_name, asset_id)
    landsat_observed = get_cloud_masks.load_mask(b10.shape, missing_landsat, specific)
    b10[landsat_observed == 0] = np.nan
    return b10


def partial_b11(place_name, asset_id, landsat, missing_landsat, specific):
    b11 = toa.load_asset_b11(place_name, asset_id)
    landsat_observed = get_cloud_masks.load_mask(b11.shape, missing_landsat, specific)
    b11[landsat_observed == 0] = np.nan
    return b11


def partial_SR_band(band):
    def wrap(place_name, asset_id, landsat, missing_landsat, specific):
        sr_band = sr.load_SR_band(band)(place_name, asset_id)
        landsat_observed = get_cloud_masks.load_mask(sr_band.shape, missing_landsat, specific)
        sr_band[landsat_observed == 0] = np.nan
        return sr_band

    return wrap


def partial_asset_fvc(place_name, asset_id, landsat, missing_landsat, specific):
    srB4, srB5 = sr.load_srB4(place_name, asset_id), sr.load_srB5(place_name, asset_id)
    landsat_observed = get_cloud_masks.load_mask(srB4.shape, missing_landsat, specific)
    srB4[landsat_observed == 0] = np.nan
    srB5[landsat_observed == 0] = np.nan
    return sr.compute_fvc(nir=srB5, red=srB4)


def partial_qa(place_name, asset_id, landsat, missing_landsat, specific):
    full_qa = sr.load_asset_qa(place_name, asset_id)
    landsat_observed = get_cloud_masks.load_mask(full_qa.shape, missing_landsat, specific)
    qa = sr.apply_cloud(full_qa, is_cloud_mask=~landsat_observed)
    return qa


def demo_cloud_missing_only():
    place_name = "jakarta"
    asset_id = "LC08_122064_20200422"
    missing = 50
    specific = 0

    fig, ax = plt.subplots(1, 2, figsize=(10, 5), sharex=True, sharey=True)

    qa = partial_qa(place_name, asset_id, "L8", missing, specific)
    fvc = partial_asset_fvc(place_name, asset_id, "L8", missing, specific)
    ax[0].imshow(fvc)
    ax[0].set_title("Landsat FVC")
    em = aster.compute_dynamic_emissivity(
        fvc,  # landsat fvc has gaps from clouds
        aster.load_aster_b13(place_name),
        aster.load_aster_b14(place_name),
        aster.load_aster_fvc(place_name),
        "L8",
    )
    tir = partial_b10(place_name, asset_id, "L8", missing, specific)

    # fully observed inputs
    tpw_pos = ncep.load_asset_TPWpos(place_name, asset_id)

    lst = compute_lst.compute_LST(em, qa, tir, tpw_pos, "L8")
    ax[1].matshow(lst)
    ax[1].set_title("LST output")
    plt.show()


if __name__ == "__main__":
    demo_cloud_missing_only()