If your dataset is already in the pascal voc format, you can simply register it by:    
```python
from detectron2.data.datasets import register_pascal_voc
register_pascal_voc(name, dirname, split, year)
```
reference from: https://github.com/facebookresearch/detectron2/blob/master/detectron2/data/datasets/pascal_voc.py
