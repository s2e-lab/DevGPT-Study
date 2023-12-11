import fiftyone.utils.coco as fouc

# 路径到您的COCO数据集
coco_dir = "/path/to/open-images-v6"

# 导入数据集
dataset.add_dir(coco_dir, fo.types.COCODetectionDataset)
