If your dataset is already in the COCO format, you can simply register it by:  
```python
from detectron2.data.datasets import register_coco_instances
register_coco_instances("my_dataset", {}, "json_annotation.json", "path/to/image/dir")
```
which will take care of everything (including metadata) for you.  
If your dataset is in COCO format with custom per-instance annotations, the load_coco_json function can be used.  
reference from: https://github.com/facebookresearch/detectron2/blob/master/detectron2/data/datasets/register_coco.py
