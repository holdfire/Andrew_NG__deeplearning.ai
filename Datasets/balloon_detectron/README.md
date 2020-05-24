#### 1. the balloon dataset  
from: https://github.com/matterport/Mask_RCNN/releases/download/v2.1/balloon_dataset.zip


#### 2. dataset structure  
--balloon  
----train  
------train_001.jpg  
------via_region_data.json  
----val  
------val_001.jpg  
------via_region_data.json  


#### 3. via_region_data.jon's format  
```json
{
    "train_001.jpg875646":{
        "fileref": "",
        "filename": "train_001.jpg",
        "base64_img_data": "",
        
        "regions":{
            "0": {
                "region_attributes": {},
                "shape_attributes": {
                    "all_points_x": [],
                    "all_points_y": [],
                    "name": "polygon"
                }
            },
            "1": {"......"
            }
        },
        "size": 875646,
        "file_attributes": {}
    },
    "train_002.jpgsize2": {"......"
    }
}
```
