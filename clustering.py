import os
import argparse
import shapely
import json
import time
import joblib
import numpy as np

from tqdm.auto import tqdm
from functools import partial

from sklearn.neighbors import KDTree

from multiprocessing.pool import ThreadPool

from lib.biomarker.utils import (
    load_data,
    convert_time,
    get_logger,
    check_processed,
    make_dir_result,
)
from lib.biomarker.digital_tils import (
    get_cluster_region,
    get_radius,
    check_contain,
    clean_feature,
    clustering_spatial,
    get_centroid_contour,
    get_poly_contour,
    check_poly_valid,
)


def main(
    fpath,
    concave,
    use_hovernet,
    n_worker,
    outer_buffer,
    tum_distance,
    tum_samples,
    logger,
    nuclei_type_1=1,  # nuclei to be clustered/tumour
    nuclei_type_2=2,  # nuclei to counted/lymphocyte
):
    first_start = time.time()

    fname = fpath.split("/")[-1].split(".")[0]
    logger.info(
        f"distance: {tum_distance}  samples: {tum_samples}  buffer: {outer_buffer}"
    )
    logger.info(f"Processing {fname}")

    #################################################################

    start = time.time()

    logger.info("Opening data...")
    nuclei_data = load_data(fpath)

    logger.info(f"Data loaded in :{convert_time(time.time() - start)}")

    #################################################################

    logger.info("   preparing data...")
    if use_hovernet:
        # 1 is tumour
        # 2 is lymphocyte
        key_tum, key_lym, centroids_tum, centroids_lym = get_centroid_contour(
            nuclei_data=nuclei_data, tumour_label=nuclei_type_1, lym_label=nuclei_type_2
        )
    else:
        key_tum, key_lym, centroids_tum, centroids_lym = get_centroid_contour(
            nuclei_data=nuclei_data
        )
    kdtree_lym = KDTree(centroids_lym)
    kdtree_tum = KDTree(centroids_tum)

    #################################################################

    logger.info("   clustering...")
    clusters_centroids = clustering_spatial(
        centroids=centroids_tum,
        key_tum=key_tum,
        nuclei_data=nuclei_data,
        distance=tum_distance,
        min_sample=tum_samples,
    )

    #################################################################

    result_dict = {
        "num_tumour_cells": len(centroids_tum),
        "num_lymphocyte_cells": len(centroids_lym),
        "tumour_cluster": {},
    }

    #################################################################

    logger.info(f"   starting get cluster region...")
    start = time.time()
    part_cluster_region = partial(get_cluster_region, concave)
    with ThreadPool(processes=n_worker) as p:
        cluster_polygons = p.map(part_cluster_region, clusters_centroids)

    if concave:
        cluster_concaves = []
        for poly in cluster_polygons:
            if type(poly) == shapely.geometry.multipolygon.MultiPolygon:
                # multi_poly = list(poly.geoms)
                for poly2 in poly.geoms:
                    if check_poly_valid(poly2, outer_buffer):
                        cluster_concaves.append(poly2)
            else:
                if check_poly_valid(poly, outer_buffer):
                    cluster_concaves.append(poly)
    else:
        cluster_concaves = cluster_polygons

    dct_visual = {}
    for i, pol in enumerate(cluster_concaves):
        dct_visual[i] = {
            "inner_contours": np.around(np.array(pol.exterior.coords.xy), 4).T.tolist(),
            "outer_contours": None,
            "inner_TIL_contour": None,
            "outer_TIL_contour": None,
            "inner_tum_contour": None,
            "outer_tum_contour": None,
        }
    # logger.info(f"   found {len(cluster_concaves)} concave objects")
    logger.info(f"   finished get cluster region: {convert_time(time.time() - start)}s")
    #################################################################

    logger.info("   starting biomarker calculation..")
    start = time.time()
    for ix, cluster_tum_hull in enumerate(
        tqdm(cluster_concaves, bar_format="{l_bar}{bar:10}{r_bar}{bar:-10b}")
    ):
        #################################################################

        outer_cluster_tum_hull_ = cluster_tum_hull.buffer(outer_buffer)  # expanded area
        outer_cluster_tum_hull = (
            outer_cluster_tum_hull_ - cluster_tum_hull
        )  # expanded area - original area
        part_check = partial(check_contain, cluster_tum_hull, outer_cluster_tum_hull)

        #################################################################

        # if (len(outer_cluster_tum_hull.bounds) == 4) and (not outer_cluster_tum_hull.is_empty):
        c_x, c_y, radius = get_radius(outer_cluster_tum_hull_)
        outer_contours = get_poly_contour(outer_cluster_tum_hull_)
        #################################################################

        list_nearest_lymphocyte = kdtree_lym.query_radius([[c_x, c_y]], r=radius)
        nearest_lym = list_nearest_lymphocyte[0].tolist()

        nearest_lym_centroid = []
        nearest_lym_contour = []
        for x in key_lym[nearest_lym]:
            nearest_lym_centroid.append(nuclei_data[x]["centroid"])
            nearest_lym_contour.append(nuclei_data[x]["contour"])

        #################################################################

        list_nearest_tumour = kdtree_tum.query_radius([[c_x, c_y]], r=radius)
        nearest_tum = list_nearest_tumour[0].tolist()

        nearest_tum_centroid = []
        nearest_tum_contour = []
        for x in key_tum[nearest_tum]:
            nearest_tum_centroid.append(nuclei_data[x]["centroid"])
            nearest_tum_contour.append(nuclei_data[x]["contour"])

        #################################################################

        with ThreadPool(processes=n_worker) as p:
            temp_til = p.starmap(
                part_check, zip(nearest_lym_centroid, nearest_lym_contour)
            )

        with ThreadPool(processes=n_worker) as p:
            temp_tum = p.starmap(
                part_check, zip(nearest_tum_centroid, nearest_tum_contour)
            )

        #################################################################

        (
            inner_count,
            outer_count,
            stat_inner_features,
            stat_outer_features,
            inner_TIL_contour,
            outer_TIL_contour,
        ) = clean_feature(temp_til)
        (
            inner_countTum,
            outer_countTum,
            stat_inner_featuresTum,
            stat_outer_featuresTum,
            inner_tum_contour,
            outer_tum_contour,
        ) = clean_feature(temp_tum)
        result_dict["tumour_cluster"][ix] = {
            "inner_tumour_area": cluster_tum_hull.area,
            "outer_tumour_area": outer_cluster_tum_hull.area,
            "num_inner_til": inner_count,
            "num_outer_til": outer_count,
            "feature_inner_til": stat_inner_features,
            "feature_outer_til": stat_outer_features,
            "num_inner_tum": inner_countTum,
            "num_outer_tum": outer_countTum,
            "feature_inner_tum": stat_inner_featuresTum,
            "feature_outer_tum": stat_outer_featuresTum,
        }

        dct_visual[ix]["outer_contours"] = outer_contours
        dct_visual[ix]["inner_TIL_contour"] = inner_TIL_contour
        dct_visual[ix]["outer_TIL_contour"] = outer_TIL_contour
        dct_visual[ix]["inner_tum_contour"] = inner_tum_contour
        dct_visual[ix]["outer_tum_contour"] = outer_tum_contour

        #################################################################

        #################################################################
    logger.info(f"   finished calculating: {convert_time(time.time() - start)}s")
    logger.info(f"{fname} is finished in :{convert_time(time.time() - first_start)}")
    logger.info(f"\n\n")

    return result_dict, dct_visual


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "-i", "--input_dir", help="Directory to nuclei annotation from HoverNet"
    )
    parser.add_argument("-o", "--output_dir", help="Directory to save the results")
    parser.add_argument(
        "-cc",
        "--use_concave",
        action="store_true",
        help="Create concave cluster or not. If false, cluster is convex (faster)",
    )
    parser.add_argument(
        "-nd",
        "--nuclei_dist",
        default=100,
        help="Minimum distance between nuclei, clustering hyperparameter.",
    )
    parser.add_argument(
        "-nn",
        "--num_nuclei",
        default=10,
        help="Minimum number of nuclei in cluster, clustering hyperparameter.",
    )
    parser.add_argument(
        "-ob", "--outer_buffer", default=60, help="Size of the enlarged cluster area"
    )
    parser.add_argument(
        "-nw", "--num_worker", default=4, help="CPU count for multiprocessing"
    )
    parser.add_argument(
        "-hv",
        "--use_hovernet",
        action="store_true",
        help="Whether use json from HoverNet or not",
    )

    args = parser.parse_args()

    nuclei_dir = args.input_dir
    outdir = args.output_dir
    nuclei_dist = int(args.nuclei_dist)
    num_nuclei = int(args.num_nuclei)
    outer_buffer = int(args.outer_buffer)
    n_worker = int(args.num_worker)
    use_hovernet = args.use_hovernet
    use_concave = args.use_concave

    rootdir, fdir, vdir = make_dir_result(
        use_concave, outdir, outer_buffer, nuclei_dist, num_nuclei
    )
    log_file = f"{rootdir}/tum_dist[{nuclei_dist}]-tum_smpl[{num_nuclei}]_buffer[{outer_buffer}].log"
    logger = get_logger(logger_name="my_log", log_file=log_file)

    file_list, total_len, total_done = check_processed(nuclei_dir, fdir)
    file_list = sorted(file_list, key=os.path.getsize)

    for i, fpath in enumerate(file_list):
        print(f"COUNTER: {i+1+total_done}/{total_len}")
        fname = fpath.split("/")[-1].split(".")[0]
        result, dct_visual = main(
            fpath=fpath,
            concave=True,
            use_hovernet=use_hovernet,
            n_worker=n_worker,
            outer_buffer=outer_buffer,
            tum_distance=nuclei_dist,
            tum_samples=num_nuclei,
            logger=logger,
        )
        out_featureF = f"{fdir}/{fname}.json"
        with open(out_featureF, "w") as f:
            json.dump(result, f, indent=6)

        out_visualF = f"{vdir}/{fname}.dat"
        with open(out_visualF, "wb") as f:
            joblib.dump(dct_visual, f)

    print("ALL FILE HAVE BEEN PROCESSED")
