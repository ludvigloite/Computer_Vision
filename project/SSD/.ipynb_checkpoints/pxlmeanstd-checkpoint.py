import numpy as np
import matplotlib.pyplot as plt
from train import get_parser
from ssd.config.defaults import cfg
from ssd.data.build import make_data_loader
from vizer.draw import draw_boxes
np.random.seed(0)

config_path = “configs/res_rdd.yaml” # yaml file for RDD2020
cfg.merge_from_file(config_path)
cfg.freeze()

data_loader = make_data_loader(cfg, is_train=True)
if isinstance(data_loader, list):
    data_loader = data_loader[0]
dataset = data_loader.dataset
indices = list(range(len(dataset)))
MEAN = []
STD = []
for i in indices:
    idx = indices[i]
    image_id = dataset.image_ids[idx]
    image = dataset._read_image(image_id)
    MEAN.append(np.mean(image))
    STD.append(np.std(image))

print(np.mean(MEAN)) # 119.7336
print(np.mean(STD)) # 68.62

