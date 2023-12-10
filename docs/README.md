<img src="graphical_abstract.png" width=200px align="rigt">

AI-Based Risk Score from Tumour-Infiltrating Lymphocyte Predicts Locoregional-Free Survival in Nasopharyngeal Carcinoma. *MDPI Cancers*

[Journal Link](https://www.mdpi.com/journal/cancers/special_issues/UQW723W3OP)

*TL;DR*: The code provide finding intratumoural, stromal tumour infiltrating lymphocytes (TILs) by density-based clustering and generating 12 TILs score in the paper.

## Pre-requisites:
* Linux (tested on Ubuntu 22.04)
* Python = 3.8, alphashape (1.3.1), opencv-python(4.6.0), Pillow (9.3.0), scikit-learn (1.2.0), scipy (1.9.3), shapely (2.0.0)

## Installation

```
git clone https://github.com/mdsatria/npc_digital_tils.git
cd npc_digital_tils # or your clone directory
conda create --name YOUR_ENV_NAME python=3.7
conda activate YOUR_ENV_NAME
pip install -r requirements.txt
```

## How to use
1. Detect nuclei in you images/WSIswith [HoverNet](https://github.com/vqdang/hover_net)
2. Open terminal in cloned git directory 
3. `chmod +x run_clustering.sh` 
4. Change the argument based on your setting
5. `./run_clustering.sh`
6. See examples.ipynb to visualise TILs and how to generate TILs scores

## Usage and options
```
--input_dir         Directory to nuclei annotation from HoverNet
--output_dir        Directory to save the results
--use_concave       Create concave cluster or not. If false, cluster is convex (may faster)
--nuclei_dist       Minimum distance between nuclei, clustering hyperparameter.
--num_nuclei        Minimum number of nuclei in cluster, clustering hyperparameter.
--outer_buffer      Size of the enlarged cluster area
--num_worker        CPU count for multiprocessing
```

## TILs scores and visualisation
Please refer to example.ipynb
