## How to use custom dataset to train model   
#### 1. Register your dataset to detectron2's standard dataset format as follows   
referenced from: https://detectron2.readthedocs.io/tutorials/datasets.html    
```python
dataset_dict = []  
for image_record in dataset_dict:
    image_record = {
        "filename": "the absolute path of an image",
        "sem_seg_file_name": "the full path to the ground truth semantic segmentation file",
        "sem_seg": "semantic segmentation ground truth in a 2D torch.Tensor",
        "height": "integer",
        "width": "integer",
        "image_id": "string or int, a unique id that identifies this image.",
        "annotations": "[
            {
             "bbox": "list of 4 numbers representing the bounding box of the instance",
             "bbox_mode": "the format of bbox",
             "category_id": "an integer representing the category label",
             "segmentation": "list[list[float]] or dict" ,
             "keypoints": "{}",
             "is_crowd": "0 or 1. Whether this instance is labeled as COCO’s “crowd region”
            },
            {}
         ]",
         "proposal_boxes": "2D numpy array with shape (K, 4) representing K precomputed proposal boxes for this image",
         "proposal_objectness_logits": "numpy array with shape (K, ), which corresponds to the objectness logits of proposals in proposal_boxes",
         "proposal_bbox_mode": "the format of the precomputed proposal bbox. It must be a member of structures.BoxMode"
    }
```

#### 2. Add metadata for datasets
Each dataset is associated with some metadata, accessible through`MetadataCatalog.get(dataset_name).some_metadata`Metadata is a key-value mapping that contains primitive information that helps interpret what’s in the dataset, e.g., names of classes, colors of classes, root of files, etc. This information will be useful for augmentation, evaluation, visualization, logging, etc. The structure of metadata depends on the what is needed from the corresponding downstream code.
+ `thing_dataset_id_to_contiguous_id` (dict[int->int]): Used by all instance detection/segmentation tasks in the COCO format. A mapping from instance class ids in the dataset to contiguous ids in range [0, #class). Will be automatically set by the function `load_coco_json`.
+ `stuff_dataset_id_to_contiguous_id` (dict[int->int]): Used when generating prediction json files for semantic/panoptic segmentation. A mapping from semantic segmentation class ids in the dataset to contiguous ids in [0, num_categories). It is useful for evaluation only.
+ `json_file`: The COCO annotation json file. Used by COCO evaluation for COCO-format datasets.
+ `panoptic_root`, `panoptic_json`: Used by panoptic evaluation.
+ `evaluator_type`: Used by the builtin main training script to select evaluator. No need to use it if you write your own main script. You can just provide the DatasetEvaluator for your dataset directly in your main script.

#### 3. Update the Config for New Datasets
Once you’ve registered the dataset, you can use the name of the dataset (e.g., “my_dataset” in example above) in DATASETS.{TRAIN,TEST}. There are other configs you might want to change to train or evaluate on new datasets:
+ `MODEL.ROI_HEADS.NUM_CLASSES` and `MODEL.RETINANET.NUM_CLASSES` are the number of thing classes for R-CNN and RetinaNet models.
+ `MODEL.ROI_KEYPOINT_HEAD.NUM_KEYPOINTS` sets the number of keypoints for Keypoint R-CNN. You’ll also need to set Keypoint OKS with `TEST.KEYPOINT_OKS_SIGMAS` for evaluation.
+ `MODEL.SEM_SEG_HEAD.NUM_CLASSES` sets the number of stuff classes for Semantic FPN & Panoptic FPN.
+ If you’re training Fast R-CNN (with precomputed proposals), `DATASETS.PROPOSAL_FILES_{TRAIN,TEST}` need to match the datasts. The format of proposal files are documented here.





