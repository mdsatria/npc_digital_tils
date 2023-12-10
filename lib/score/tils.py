import os
import joblib
import numpy as np
import openslide as ops
import matplotlib.pyplot as plt
import json


class TILData:
    def __init__(self, wsi_id, slide_dir, clsinfo_dir, cluster_dir, slide_ext="svs"):
        self.wsi_id = wsi_id
        self.slide_dir = slide_dir
        self.clsinfo_dir = clsinfo_dir
        self.cluster_dir = cluster_dir
        self.slide_ext = slide_ext

        self.slide = ops.OpenSlide(f"{self.slide_dir}/{self.wsi_id}.{self.slide_ext}")

        self.features = self.load_metadata(
            f"{self.clsinfo_dir}/{self.wsi_id}.json", "json"
        )
        self.cluster_ids = list(self.features["tumour_cluster"].keys())

        if cluster_dir == None:
            self.cluster_data = None
        else:
            print("reading clusters")
            self.cluster_data = self.load_metadata(
                f"{self.cluster_dir}/{self.wsi_id}.dat", "dat"
            )

    @staticmethod
    def load_metadata(fname, data_type):
        assert data_type in ["json", "dat"], "WRONG DATA TYPE"

        if data_type == "json":
            with open(fname, "r") as f:
                data = json.load(f)

        elif data_type == "dat":
            with open(fname, "rb") as f:
                data = joblib.load(f)

        return data


class TILVisual(TILData):
    def __init__(
        self,
        wsi_id,
        slide_dir,
        clsinfo_dir,
        cluster_dir,
        nuclei_dir=None,
        slide_ext="svs",
        slide_level=1,
    ):
        super(TILVisual, self).__init__(
            wsi_id, slide_dir, clsinfo_dir, cluster_dir, slide_ext
        )
        self.slide_level = slide_level
        self.nuclei_dir = nuclei_dir

        if self.nuclei_dir is not None:
            print("reading nuclei annotation")
            self.nuclei_data = self.load_metadata(
                f"{self.nuclei_dir}/{self.wsi_id}.dat", "dat"
            )

    @staticmethod
    def is_object_inside_bbox(nuclei_centroid, bbox_coords):
        check1 = bbox_coords[0][0] <= nuclei_centroid[0]
        check2 = nuclei_centroid[0] <= bbox_coords[1][0]
        check3 = bbox_coords[0][1] <= nuclei_centroid[1]
        check4 = nuclei_centroid[1] <= bbox_coords[1][1]
        if check1 & check2 & check3 & check4:
            return True
        else:
            return False

    @staticmethod
    def get_bounding_box(points, centered_by):
        """
        calculate bounding box of outer cluster
        points = outer cluster contours
        centered_by = give padding
        return coordinate of top_left and bottom_right
        """
        x_coordinates, y_coordinates = zip(*points)
        top_left = (
            int(min(x_coordinates)) - centered_by,
            int(min(y_coordinates)) - centered_by,
        )
        bottom_right = (
            int(max(x_coordinates)) + centered_by,
            int(max(y_coordinates)) + centered_by,
        )

        return top_left, bottom_right

    @staticmethod
    def rgb_to_hex(r, g, b):
        return "#{:02x}{:02x}{:02x}".format(r, g, b)

    def find_all_nuclei_in_cluster(self, top_left_bbox, bottom_right_bbox, centered_by):
        """
        check all nuclei type in particular cluster bounding box
        """
        # control nuclei in edge of picture
        fix_edge = centered_by // 2
        top_left_bbox = [x + fix_edge for x in top_left_bbox]
        bottom_right_bbox = [x - fix_edge for x in bottom_right_bbox]
        # control nuclei in edge of picture

        contours = []
        colors = []
        for _, v in self.nuclei_data.items():
            if self.is_object_inside_bbox(
                v["centroid"], (top_left_bbox, bottom_right_bbox)
            ):
                rgb = [int(x * 255) for x in list(v["color"])]
                contours.append(v["contour"])
                colors.append(self.rgb_to_hex(rgb[0], rgb[1], rgb[2]))
                # nuclei_dct[k] = {
                #     "contour":v["contour"],
                #     "color":self.rgb_to_hex(rgb[0], rgb[1], rgb[2])
                # }
        return contours, colors

    def get_contours_in_cluster(self, cluster_id):
        """
        show all type of contours in 1 cluster
        """
        inner_contours = np.array(self.cluster_data[cluster_id]["inner_contours"])
        outer_contours = np.array(self.cluster_data[cluster_id]["outer_contours"])
        inner_TIL_contour = self.cluster_data[cluster_id]["inner_TIL_contour"]
        outer_TIL_contour = self.cluster_data[cluster_id]["outer_TIL_contour"]
        inner_tum_contour = self.cluster_data[cluster_id]["inner_tum_contour"]
        outer_tum_contour = self.cluster_data[cluster_id]["outer_tum_contour"]
        return (
            inner_contours,
            outer_contours,
            inner_TIL_contour,
            outer_TIL_contour,
            inner_tum_contour,
            outer_tum_contour,
        )

    def get_downsampled_coords(self, cluster_id, centered_by=50):
        """ """
        (
            inner_contours,
            outer_contours,
            inner_TIL_contour,
            outer_TIL_contour,
            inner_tum_contour,
            outer_tum_contour,
        ) = self.get_contours_in_cluster(cluster_id)
        # index of minimum value of

        top_left_coor, bottom_right_coor = self.get_bounding_box(
            outer_contours, centered_by=centered_by
        )
        downsample = self.slide.level_downsamples[self.slide_level]

        width = int((bottom_right_coor[0] - top_left_coor[0]) // downsample)
        height = int((bottom_right_coor[1] - top_left_coor[1]) // downsample)
        inner_contour_downsample = (inner_contours - top_left_coor) / downsample
        outer_contour_downsample = (outer_contours - top_left_coor) / downsample

        return (
            inner_contours,
            outer_contours,
            inner_TIL_contour,
            outer_TIL_contour,
            inner_tum_contour,
            outer_tum_contour,
            top_left_coor,
            bottom_right_coor,
            downsample,
            width,
            height,
            inner_contour_downsample,
            outer_contour_downsample,
        )

    def get_thumbnail(self, params, is_save, save_dir, fig_name, save_as="png"):
        dimensions = self.slide.level_dimensions
        dimensions_sum = [sum(x) for x in dimensions]
        min_index = dimensions_sum.index(min(dimensions_sum))
        thumb = self.slide.get_thumbnail(dimensions[min_index])
        fig, ax = plt.subplots(**params)
        ax.imshow(thumb)
        ax.axis("off")

        if is_save:
            plt.savefig(
                f"{save_dir}/{fig_name}.{save_as}",
                format=save_as,
                bbox_inches="tight",
                pad_inches=0,
                dpi=300,
            )
        fig.show()

    def plot(
        self,
        cluster_id,
        to_plots,
        alpha,
        save_as="png",
        centered_by=50,
        params=None,
        save=False,
        fig_name=None,
        save_path="figures",
    ):
        (
            _,
            _,
            inner_TIL_contour,
            outer_TIL_contour,
            inner_tum_contour,
            outer_tum_contour,
            top_left_coor,
            bottom_right_coor,
            downsample,
            width,
            height,
            inner_contour_downsample,
            outer_contour_downsample,
        ) = self.get_downsampled_coords(cluster_id=cluster_id, centered_by=centered_by)

        fig, ax = plt.subplots(**params)

        if to_plots["nuclei"]:
            """
            plot nuclei prediction/annotation
            """
            nuclei_contours, nuclei_colors = self.find_all_nuclei_in_cluster(
                top_left_coor, bottom_right_coor, centered_by
            )
            for ncts, ncls in zip(nuclei_contours, nuclei_colors):
                ncts = np.array(ncts)
                ncts = (ncts - top_left_coor) / downsample
                ax.fill(
                    ncts[:, 0], ncts[:, 1], c=ncls, alpha=alpha["type2"], edgecolor=None
                )  # f59e07

        if to_plots["tissue"]:
            """
            plot tissue
            """
            ax.imshow(
                self.slide.read_region(
                    top_left_coor, self.slide_level, (width, height)
                ),
                alpha=alpha["type1"],
            )

        if to_plots["inner_clusters"]:
            """
            plot inner tumour cluster
            """
            ax.plot(
                inner_contour_downsample[:, 0],
                inner_contour_downsample[:, 1],
                c="black",
                linewidth=4,
                alpha=alpha["type1"],
            )

        if to_plots["outer_clusters"]:
            """
            plot outer tumour cluster
            """
            ax.plot(
                outer_contour_downsample[:, 0],
                outer_contour_downsample[:, 1],
                c="black",
                linewidth=2,
                linestyle="--",
                alpha=alpha["type1"],
            )
        if to_plots["cluster_fill"]:
            """
            fill cluster with color
            """
            ax.fill(
                inner_contour_downsample[:, 0],
                inner_contour_downsample[:, 1],
                c="#d83b2e",
                alpha=alpha["type2"],
            )

        if to_plots["inner_tils"]:
            """
            plot inner TILs
            """
            for ci in inner_TIL_contour:
                ci = np.array(ci)
                ci = (ci - top_left_coor) / downsample
                ax.fill(
                    ci[:, 0], ci[:, 1], c="#369a50", alpha=alpha["type2"]
                )  # , edgecolor=None) #369a50

        if to_plots["outer_tils"]:
            """
            plot outer TILs
            """
            for ci in outer_TIL_contour:
                ci = np.array(ci)
                ci = (ci - top_left_coor) / downsample
                ax.fill(
                    ci[:, 0], ci[:, 1], c="#0b49bd", alpha=alpha["type2"]
                )  # , edgecolor=None) #0b49bd

        if to_plots["inner_tums"]:
            """
            plot inner tumour nuclei
            """
            for ci in inner_tum_contour:
                ci = np.array(ci)
                ci = (ci - top_left_coor) / downsample
                ax.fill(
                    ci[:, 0], ci[:, 1], c="#d83b2e", alpha=alpha["type2"]
                )  # , edgecolor=None) #d83b2e
        if to_plots["outer_tums"]:
            """
            plot outer tumour nuclei
            """
            for ci in outer_tum_contour:
                ci = np.array(ci)
                ci = (ci - top_left_coor) / downsample
                ax.fill(
                    ci[:, 0], ci[:, 1], c="#f5ff00", alpha=alpha["type2"]
                )  # , edgecolor=None) #f59e07

        ax.axis("off")

        if save:
            if not os.path.exist(save_path):
                os.makedirs(save_path)
            plt.savefig(
                f"{save_path}/{fig_name}.{save_as}",
                format=save_as,
                bbox_inches="tight",
                pad_inches=0,
                dpi=300,
            )
        fig.show()





class TILScore(TILData):
    def __init__(
        self,
        wsi_id,
        slide_dir,
        clsinfo_dir,
        cluster_dir,
        slide_ext="svs",
        percentage_cluster=False,
    ):
        super(TILScore, self).__init__(
            wsi_id, slide_dir, clsinfo_dir, cluster_dir, slide_ext
        )
        self.tumour_cluster = self.features["tumour_cluster"]
        self.n_cluster = list(self.tumour_cluster.keys())
        self.features_names_til = [
            "ntil_ntum",
            "ntil_acls",
            "ntil_atum",
            "atil_atum",
            "inner_ntil_ntum",
            "outer_ntil_ntum",
            "inner_atil_atum",
            "outer_atil_atum",
            "inner_ntil_acls",
            "outer_ntil_acls",
            "inner_ntil_atum",
            "outer_ntil_atum",
        ]

        self.features_names_morph = [
            "convex_area",
            "contour_area",
            "eccentricity",
            "equiv_diameter",
            "major_axis_length",
            "minor_axis_length",
            "perimeter",
            "solidity",
            "orientation",
            "radius",
            "bbox_area",
        ]
        # check if count all cluster or just the fraction
        self.percent_cls = percentage_cluster
        if self.percent_cls is False:
            self.cls_id_to_be_counted = self.n_cluster
        else:
            self.cls_id_to_be_counted = self.sort_cls_by_size()

    @staticmethod
    def division(a, b):
        if float(b) == 0.0:
            return 0
        else:
            return a / b

    def get_tils_per_cluster(self, cluster_id, as_dict=False):
        cluster_feature = self.tumour_cluster[str(cluster_id)]

        # to avoid error if no tumour/lymphocyte in cluster
        special_features = {
            "feature_inner_til": None,
            "feature_inner_tum": None,
            "feature_outer_til": None,
            "feature_outer_tum": None,
        }
        for k, _ in special_features.items():
            if cluster_feature[k] is False:
                special_features[k] = 0.0
            else:
                special_features[k] = cluster_feature[k][12]  # sum of area

        inner_num_til = cluster_feature["num_inner_til"]
        inner_num_tum = cluster_feature["num_inner_tum"]
        inner_cls_area = cluster_feature["inner_tumour_area"]
        inner_til_area = special_features["feature_inner_til"]
        inner_tum_area = special_features["feature_inner_tum"]

        outer_num_til = cluster_feature["num_outer_til"]
        outer_num_tum = cluster_feature["num_outer_tum"]
        outer_cls_area = cluster_feature["outer_tumour_area"]
        outer_til_area = special_features["feature_outer_til"]
        outer_tum_area = special_features["feature_outer_tum"]

        num_til = inner_num_til + outer_num_til
        num_tum = inner_num_tum + outer_num_tum
        cls_area = inner_cls_area + outer_cls_area
        til_area = inner_til_area + outer_til_area
        tum_area = inner_tum_area + outer_tum_area

        result = [
            self.division(num_til, num_tum),  # til_ratio
            self.division(num_til, cls_area),  # til_density_to_cluster
            self.division(num_til, tum_area),  # til_density_to_tum
            self.division(til_area, tum_area),  # til_ratio_area
            self.division(inner_num_til, inner_num_tum),
            self.division(outer_num_til, outer_num_tum),
            self.division(inner_til_area, inner_tum_area),
            self.division(outer_til_area, outer_tum_area),
            self.division(inner_num_til, inner_cls_area),
            self.division(outer_num_til, outer_cls_area),
            self.division(inner_num_til, inner_tum_area),
            self.division(outer_num_til, outer_tum_area),
        ]

        if not as_dict:
            return result
        else:
            result_dict = {}
            for i, k in enumerate(self.features_names_til):
                result_dict[k] = result[i]
            return result_dict

    def sort_cls_by_size(self):
        cluster_size = []
        for cls_id in self.cluster_ids:
            cluster_size.append(self.tumour_cluster[cls_id]["inner_tumour_area"])
        cluster_size = np.array(cluster_size)
        percent_len = np.ceil(self.percent_cls * len(self.cluster_ids)).astype(int)
        descending_order = np.argsort(cluster_size)[::-1][:percent_len]

        cls_id_to_be_counted = np.array(self.cluster_ids)[descending_order]

        return cls_id_to_be_counted

    def get_tils_all(self):
        result = []
        for n in self.cls_id_to_be_counted:
            result.append(self.get_tils_per_cluster(n))
        avg_res = np.array(result).mean(axis=0)
        std_res = np.array(result).std(axis=0)
        fulL_res = np.concatenate((avg_res, std_res))

        result_dct = {}
        for i, stat in enumerate(["mean", "std"]):
            for j, k in enumerate(self.features_names_til):
                result_dct[f"{stat}_{k}"] = fulL_res[i + j]
        return result_dct

    def get_tumour_m_features(self):
        feature_tum_morph = []
        for cluster_id in self.cls_id_to_be_counted:
            if self.tumour_cluster[cluster_id]["feature_inner_tum"] is not False:
                feature_tum_morph.append(
                    self.tumour_cluster[cluster_id]["feature_inner_tum"][:11]
                )
            
        feature_tum_morph = np.array(feature_tum_morph).mean(axis=0)

        dct_tum = {}
        for i, k in enumerate(self.features_names_morph):            
            dct_tum[k] = feature_tum_morph[i]

        return dct_tum
