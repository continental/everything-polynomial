'''
Copyright (C) 2024 co-pace GmbH (a subsidiary of Continental AG).
Licensed under the BSD-3-Clause License.
@author: Yao Yue
'''

import utils_waymo.preprocess_utils as preprocess_utils

preprocess_utils.create_infos_from_data(raw_data_path="data/datasets/", output_path="data/datasets/", splits = ['val_WO'], process_map = True, process_track = True)
