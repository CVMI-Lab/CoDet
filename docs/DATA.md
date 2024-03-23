# Prepare datasets for CoDet


### COCO and LVIS

First, download COCO and LVIS data place them in the following way:

```
lvis/
    lvis_v1_train.json
    lvis_v1_val.json
coco/
    train2017/
    val2017/
    annotations/
        captions_train2017.json
        instances_train2017.json 
        instances_val2017.json
```
Then we follow [OVR-CNN](https://github.com/alirezazareian/ovr-cnn/blob/master/ipynb/003.ipynb) to create the open-vocabulary COCO split. The converted files should be like 

```
coco/
    zero-shot/
        instances_train2017_seen_2.json
        instances_val2017_all_2.json
```

We further follow [Detic](https://github.com/facebookresearch/Detic/tree/main) to pre-process the annotation format for easier evaluation:

```
python tools/get_coco_zeroshot_oriorder.py --data_path datasets/coco/zero-shot/instances_train2017_seen_2.json
python tools/get_coco_zeroshot_oriorder.py --data_path datasets/coco/zero-shot/instances_val2017_all_2.json
```

And process the category infomation:
```
python tools/get_lvis_cat_info.py --ann datasets/coco/zero-shot/instances_train2017_seen_2_oriorder.py
```

Next, prepare the open-vocabulary LVIS training set using 

```
python tools/remove_lvis_rare.py --ann datasets/lvis/lvis_v1_train.json
```

This will generate `datasets/lvis/lvis_v1_train_norare.json`.

Then generate `datasets/lvis/lvis_v1_train_norare_cat_info.json` by running

```
python tools/get_lvis_cat_info.py --ann datasets/lvis/lvis_v1_train_norare.json --add_freq
```

### COCO Caption
Download the parsed [caption tags](https://drive.google.com/file/d/1crYRxaL3atzAhL2Qut6Ojzzd2V3uXziV/view?usp=sharing) and put it under `datasets/coco/annotations/`.



### Conceptual Caption


Download the dataset from [this](https://ai.google.com/research/ConceptualCaptions/download) page and place them as:
```
cc3m/
    GCC-training.tsv
```

Run the following command to download the images and convert the annotations to LVIS format (Note: download images takes long).

~~~
python tools/download_cc.py --ann datasets/cc3m/GCC-training.tsv --save_image_path datasets/cc3m/training/ --out_path datasets/cc3m/train_image_info.json
~~~

This creates `datasets/cc3m/train_image_info.json`.
Then extract and filter tags to get `datasets/cc3m/train_image_info_tags.json`.
~~~
pip install nltk
pip install SceneGraphParser
python -m spacy download en_core_web_trf

python tools/concept_extract.py --anno_file datasets/cc3m/train_image_info.json
python tools/concept_filter.py --anno_file datasets/cc3m/train_image_info.json --output_file datasets/cc3m/train_image_info_tags.json
~~~
Finally, we prepare the clip text embeddings for the extracted tags
~~~
python tools/dump_clip_features.py --ann datasets/cc3m/train_image_info_tags.json --out_path metadata/cc3m_clip_a+cname.npy
~~~
<!--
Then download the parsed [caption tags](https://drive.google.com/file/d/1l9elOs00jDeXQA80qtmMXauuCz3Jf2ok/view?usp=sharing) and put it under `datasets/cc3m/`.
-->

### Objects365
Download Objects365 (v2) from the website. We only need the validation set in this project:
```
objects365/
    annotations/
        zhiyuan_objv2_val.json
    val/
        images/
            v1/
                patch0/
                ...
                patch15/
            v2/
                patch16/
                ...
                patch49/

```

The original annotation has typos in the class names, we first fix them for our following use of language embeddings.

```
python tools/fix_o365_names.py --ann datasets/objects365/annotations/zhiyuan_objv2_val.json
```
This creates `datasets/objects365/zhiyuan_objv2_val_fixname.json`.



### Metadata

```
metadata/
    cc3m_clip_a+cname.npy
    coco_clip_a+cname.npy
    cococap_clip_a+cname.npy
    lvis_v1_clip_a+cname.npy
    lvis_v1_train_cat_info.json
    o365_clip_a+cnamefix.npy
    Objects365_names_fix.csv
```

`lvis_v1_train_cat_info.json` is used by the Federated loss.
This is created by 
~~~
python tools/get_lvis_cat_info.py --ann datasets/lvis/lvis_v1_train.json
~~~

`*_clip_a+cname.npy` is the pre-computed CLIP embeddings for each dataset.
They are created by (taking LVIS as an example)
~~~
python tools/dump_clip_features.py --ann datasets/lvis/lvis_v1_val.json --out_path metadata/lvis_v1_clip_a+cname.npy
~~~

`Objects365_names_fix.csv` is our manual fix of the Objects365 names.
