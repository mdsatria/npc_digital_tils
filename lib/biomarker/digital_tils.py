import cv2
import alphashape  # for make concave object. expensive computation
import numpy as np
from sklearn.cluster import DBSCAN
from shapely.geometry import MultiPoint, Point


def check_poly_valid(poly, buffer_size):
    buffered_poly = poly.buffer(buffer_size)
    condition_1 = len(buffered_poly.bounds) == 4
    condition_2 = not buffered_poly.is_empty
    return condition_1 and condition_2


def check_intersection(inner_tumour_hull, outer_tumour_hull, lymphocyte_contour):
    lymphocyte_hull = MultiPoint(lymphocyte_contour).convex_hull
    tasil = outer_tumour_hull.intersection(lymphocyte_hull).area
    til = inner_tumour_hull.intersection(lymphocyte_hull).area
    return [til, tasil]


def get_radius(polygon_data):
    min_x, min_y, max_x, max_y = polygon_data.bounds
    c_x, c_y = polygon_data.centroid.x, polygon_data.centroid.y
    radius = (c_x - min_x) + (max_x - c_x) + (c_y - min_y) + (max_y - c_y)
    return c_x, c_y, radius


def get_poly_contour(poly):
    """
    get all contur points from shapely polygon
    """
    coor_xy = poly.exterior.xy
    contours = [[x, y] for x, y in zip(coor_xy[0], coor_xy[1])]
    return contours


def get_cluster_region(concave=True, contours=None):
    buffer = 20
    if concave:
        shape = alphashape.alphashape(contours, 0.01)
        shape = shape.buffer(buffer)
    else:
        shape = MultiPoint(contours).convex_hull.buffer(buffer)
    return shape


def get_shapeFeatures(cts_type):
    """
    contours is numpy array[[x1, y1], ... , [xn,yn]]
    """
    contours = cts_type
    if type(contours) != np.ndarray:
        contours = np.array(contours)

    inst_box = np.array(
        [
            [np.min(contours[:, 1]), np.min(contours[:, 0])],
            [np.max(contours[:, 1]), np.max(contours[:, 0])],
        ]
    )

    bbox_h, bbox_w = inst_box[1] - inst_box[0]
    bbox_area = bbox_h * bbox_w
    contour_area = cv2.contourArea(contours)
    convex_hull = cv2.convexHull(contours)
    convex_area = cv2.contourArea(convex_hull)
    convex_area = convex_area if convex_area != 0 else 1
    solidity = float(contour_area) / convex_area
    equiv_diameter = np.sqrt(4 * contour_area / np.pi)
    if contours.shape[0] > 4:
        _, axes, orientation = cv2.fitEllipse(contours)
        major_axis_length = max(axes)
        minor_axis_length = min(axes)
    else:
        orientation = 0
        major_axis_length = 1
        minor_axis_length = 1
    perimeter = cv2.arcLength(contours, True)
    _, radius = cv2.minEnclosingCircle(contours)
    eccentricity = np.sqrt(1 - (minor_axis_length / major_axis_length) ** 2)
    result = [
        convex_area,
        contour_area,
        eccentricity,
        equiv_diameter,
        major_axis_length,
        minor_axis_length,
        perimeter,
        solidity,
        orientation,
        radius,
        bbox_area,
    ]
    result = [0 if np.isnan(x) else x for x in result]
    result = [0 if x == np.inf else x for x in result]
    return result


def check_contain(inner_tumour_chull, outer_tumour_chull, centroid, contour):
    if inner_tumour_chull.contains(Point(centroid)):
        feature = get_shapeFeatures(contour)
        return 1, feature, contour
    elif outer_tumour_chull.contains(Point(centroid)):
        feature = get_shapeFeatures(contour)
        return 2, feature, contour
    else:
        return False


def feature_stat(feature):
    total_feature = []
    total_feature.extend(np.array(feature).mean(axis=0).tolist())
    total_feature.extend(np.array(feature).sum(axis=0).tolist())
    total_feature.extend(np.array(feature).max(axis=0).tolist())
    total_feature.extend(np.array(feature).min(axis=0).tolist())
    total_feature.extend(np.array(feature).std(axis=0).tolist())
    return total_feature


def clean_feature(data):
    data = [x for x in data if x is not False]

    inner_count = len([x[0] for x in data if x[0] == 1])
    outer_count = len([x[0] for x in data if x[0] == 2])

    inner_features = [x[1] for x in data if x[0] == 1]
    outer_features = [x[1] for x in data if x[0] == 2]

    inner_nuclei = [x[2] for x in data if x[0] == 1]
    outer_nuclei = [x[2] for x in data if x[0] == 2]

    if inner_count > 0:
        stat_inner_features = feature_stat(inner_features)
    else:
        stat_inner_features = False

    if outer_count > 0:
        stat_outer_features = feature_stat(outer_features)
    else:
        stat_outer_features = False

    return (
        inner_count,
        outer_count,
        stat_inner_features,
        stat_outer_features,
        inner_nuclei,
        outer_nuclei,
    )


def get_centroid_contour(
    nuclei_data, tumour_label="Neoplastic", lym_label="Inflammatory"
):
    key_tum = np.array(
        [x for x in nuclei_data.keys() if nuclei_data[x]["type"] == tumour_label]
    )
    key_lym = np.array(
        [x for x in nuclei_data.keys() if nuclei_data[x]["type"] == lym_label]
    )

    centroids_tum = np.array([nuclei_data[x]["centroid"] for x in key_tum], dtype=int)
    centroids_lym = np.array([nuclei_data[x]["centroid"] for x in key_lym], dtype=int)

    return key_tum, key_lym, centroids_tum, centroids_lym


def clustering_spatial(
    centroids,
    key_tum,
    nuclei_data,
    distance,
    min_sample,
):
    # clusters_lym = DBSCAN(eps=lym_distance, min_samples=lym_samples, n_jobs=-1).fit(centroids_lym)
    # clusters_lym_labels = clusters_lym.labels_
    # id_clusters_lym = np.unique(clusters_lym_labels)

    # cluster tumour nuclei based on their centroids
    clusters_tum = DBSCAN(eps=distance, min_samples=min_sample, n_jobs=-1).fit(
        centroids
    )
    clusters_tum_labels = clusters_tum.labels_
    # find number of clusters
    id_clusters = np.unique(clusters_tum_labels)

    clusters_centroids = []
    info_num_cell_percluster = []
    for tumour_cluster_id in id_clusters:
        if tumour_cluster_id != -1:
            current_cls_tum_indices = key_tum[clusters_tum_labels == tumour_cluster_id]
            current_cls_tum_centroids = np.array(
                [nuclei_data[x]["centroid"] for x in current_cls_tum_indices], dtype=int
            )
            info_num_cell_percluster.append(len(current_cls_tum_centroids))
            clusters_centroids.append(current_cls_tum_centroids)

    return clusters_centroids
