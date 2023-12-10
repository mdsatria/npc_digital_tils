import os
import datetime
import joblib
import json
import logging
import sys
import numpy as np
from tqdm.auto import tqdm
from logging import FileHandler


def save_to_dat(data, fpath):
    with open(fpath, "wb") as f:
        joblib.dump(data, f)

def to_dat(json_data):
    dct_type = {
        1: ['Neoplastic',0],
        2: ['Inflammatory',1],
        3: ['Connective',2],
        4: ['Dead',3],
        5: ['Epithelial',4],
        }
    output = {}
    key_list = ["box", "centroid", "contour", "prob", "type", "type_id",]
    for k in list(json_data.keys()):
        if json_data[k]["type"]>0:
            output[k] = {x:None for x in key_list}
            output[k]["box"] = json_data[k]["bbox"]
            output[k]["centroid"] = json_data[k]["centroid"]
            output[k]["contour"] = json_data[k]["contour"]
            output[k]["prob"] = json_data[k]["type_prob"]
            output[k]["type"] = dct_type[json_data[k]["type"]][0]
            output[k]["type_id"] = dct_type[json_data[k]["type"]][1]

    return output


def batch_convert_to_dat(input_dir, output_dir):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    fname = [x for x in sorted(os.listdir(input_dir)) if x.endswith(".json")]

    for f in tqdm(fname):
        fpath_input = os.path.join(input_dir, f)
        fname = os.path.splitext(f)[0]
        fpath_output = os.path.join(output_dir, f"{fname}.dat")
        json_data = load_data(fpath_input)
        dat_data = to_dat(json_data)
        save_to_dat(dat_data, fpath_output)
        


    


def get_area(cts):
    x = cts[:, 0]
    y = cts[:, 1]
    area=0.5*np.sum(y[:-1]*np.diff(x) - x[:-1]*np.diff(y))
    area=np.abs(area)
    return area


def get_files(path, sort_by="size"):
    flist = [f"{path}/{x}" for x in os.listdir(path)]
    if sort_by == "size":
        flist = sorted(flist, key=os.path.getsize)
    if sort_by == "name":
        flist = sorted(flist)
    return flist


def load_data(fpath):
    data_type = fpath.split("/")[-1].split(".")[-1]

    if data_type == "dat":
        with open(fpath, "rb") as f:
            nuclei_data = joblib.load(f)

    if data_type == "json":
        with open(fpath, "r") as f:
            nuclei_data = json.load(f)
            nuclei_data = nuclei_data["nuc"]

    return nuclei_data


def convert_time(seconds):
    return str(datetime.timedelta(seconds=seconds))


def check_processed(input_dir, output_dir):
    queue_file = [f"{input_dir}/{x}" for x in sorted(os.listdir(input_dir))]
    processed_file = [
        x.split(".")[0] for x in sorted(os.listdir(output_dir)) if x.endswith(".json")
    ]

    unprocessed_file = []
    for f in queue_file:
        if f.split("/")[-1].split(".")[0] not in processed_file:
            unprocessed_file.append(f)

    return unprocessed_file, len(queue_file), len(processed_file)


def get_formatter():
    return logging.Formatter("%(asctime)s — %(levelname)s — %(message)s")


def get_console_handler():
    formatter = get_formatter()
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)
    return console_handler


def get_file_handler(log_file):
    formatter = get_formatter()
    file_handler = FileHandler(filename=log_file, mode="a")
    file_handler.setFormatter(formatter)
    return file_handler


def get_logger(logger_name, log_file):
    logger = logging.getLogger(logger_name)

    logger.propagate = False
    # THIS IS CRUCIAL FOR DUPLICATE AVOIDANCE
    if logger.hasHandlers():
        logger.handlers.clear()

    logger.setLevel(logging.INFO)
    logger.addHandler(get_console_handler())
    logger.addHandler(get_file_handler(log_file))

    return logger


def make_dir_result(concave, output_dir, outer_buffer, tum_distance, tum_samples):
    if concave:
        shapetype = "concave"
    else:
        shapetype = "convex"

    res_folder = (
        f"{shapetype}_dist[{tum_distance}]_sample[{tum_samples}]_buffer[{outer_buffer}]"
    )
    rootdir = f"{output_dir}/{res_folder}"
    fdir = f"{rootdir}/features"
    vdir = f"{rootdir}/visual"

    if os.path.exists(fdir) is False:
        os.makedirs(fdir)
    if os.path.exists(vdir) is False:
        os.makedirs(vdir)

    return rootdir, fdir, vdir
