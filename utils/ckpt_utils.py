import json
import os 

def chk_model(model_name):
    return os.path.isdir(model_name)

def download_ckpt(model_name):
    r""" 
    if model folder not exists then download
    the check point from tensorflow model zoo
    model names are as following url: https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/tf2_detection_zoo.md
    Model name	                                    Speed (ms)	COCO mAP	        Outputs
    CenterNet HourGlass104 512x512	                70	          41.9	            Boxes
    CenterNet HourGlass104 Keypoints 512x512	    76	          40.0/61.4	        Boxes/Keypoints
    CenterNet HourGlass104 1024x1024	            197	          44.5	            Boxes
    CenterNet HourGlass104 Keypoints 1024x1024	    211	          42.8/64.5	        Boxes/Keypoints
    CenterNet Resnet50 V1 FPN 512x512	            27	          31.2	            Boxes
    CenterNet Resnet50 V1 FPN Keypoints 512x512	    30	          29.3/50.7	        Boxes/Keypoints
    CenterNet Resnet101 V1 FPN 512x512	            34	          34.2	            Boxes
    CenterNet Resnet50 V2 512x512	                27	          29.5	            Boxes
    CenterNet Resnet50 V2 Keypoints 512x512	        30	          27.6/48.2	        Boxes/Keypoints
    EfficientDet D0 512x512	                        39	          33.6	            Boxes
    EfficientDet D1 640x640	                        54	          38.4	            Boxes
    EfficientDet D2 768x768	                        67	          41.8	            Boxes
    EfficientDet D3 896x896	                        95	          45.4	            Boxes
    EfficientDet D4 1024x1024	                    133	          48.5	            Boxes
    EfficientDet D5 1280x1280	                    222	          49.7	            Boxes
    EfficientDet D6 1280x1280	                    268	          50.5	            Boxes
    EfficientDet D7 1536x1536	                    325	          51.2	            Boxes
    SSD MobileNet v2 320x320	                    19	          20.2	            Boxes
    SSD MobileNet V1 FPN 640x640	                48	          29.1	            Boxes
    SSD MobileNet V2 FPNLite 320x320	            22	          22.2	            Boxes
    SSD MobileNet V2 FPNLite 640x640	            39	          28.2	            Boxes
    SSD ResNet50 V1 FPN 640x640 (RetinaNet50)	    46	          34.3	            Boxes
    SSD ResNet50 V1 FPN 1024x1024 (RetinaNet50)	    87	          38.3	            Boxes
    SSD ResNet101 V1 FPN 640x640 (RetinaNet101)	    57	          35.6	            Boxes
    SSD ResNet101 V1 FPN 1024x1024 (RetinaNet101)	104	          39.5	            Boxes
    SSD ResNet152 V1 FPN 640x640 (RetinaNet152)	    80	          35.4	            Boxes
    SSD ResNet152 V1 FPN 1024x1024 (RetinaNet152)	111	          39.6	            Boxes
    Faster R-CNN ResNet50 V1 640x640	            53	          29.3	            Boxes
    Faster R-CNN ResNet50 V1 1024x1024	            65	          31.0	            Boxes
    Faster R-CNN ResNet50 V1 800x1333	            65	          31.6	            Boxes
    Faster R-CNN ResNet101 V1 640x640	            55	          31.8	            Boxes
    Faster R-CNN ResNet101 V1 1024x1024	            72	          37.1	            Boxes
    Faster R-CNN ResNet101 V1 800x1333	            77	          36.6	            Boxes
    Faster R-CNN ResNet152 V1 640x640	            64	          32.4	            Boxes
    Faster R-CNN ResNet152 V1 1024x1024	            85	          37.6	            Boxes
    Faster R-CNN ResNet152 V1 800x1333	            101	          37.4	            Boxes
    Faster R-CNN Inception ResNet V2 640x640	    206	          37.7	            Boxes
    Faster R-CNN Inception ResNet V2 1024x1024	    236	          38.7	            Boxes
    Mask R-CNN Inception ResNet V2 1024x1024	    301	          39.0/34.6	        Boxes/Masks
    """
    with open ("./utils/models.json", "r") as f:
        url = json.loads(f.read())[model_name]        
    model_name_tar = url.split("/")[-1]
    model_name_dir = url.split("/")[-1].split(".")[0]
    if not chk_model(model_name_dir):
        try:           
            print('Downloading %s to %s...' % (url, model_name_tar))
            os.system("wget {}".format(str(url)))    
            os.system("tar xvf {}".format(str(model_name_tar)))    
            os.remove(model_name_tar)
            # assert os.path.exists()# check     
            return model_name_dir
        except Exception as e:
            print(e)
    return model_name_dir


def customized_ckpt_cofig(  
        pipeline_config_path,
        fine_tune_checkpoint,
        train_record_fname,
        test_record_fname,
        label_map_pbtxt_fname,
        batch_size,
        num_steps,
        num_classes
                                ):
    import re
    print('writing custom configuration file')

    with open(pipeline_config_path) as f:
        s = f.read()
    with open(pipeline_config_path, 'w') as f:
        
        # fine_tune_checkpoint
        s = re.sub('fine_tune_checkpoint: ".*?"',
                'fine_tune_checkpoint: "{}"'.format(fine_tune_checkpoint), s)

        # tfrecord files train and test.
        s = re.sub(
            '(input_path: ".*?)(PATH_TO_BE_CONFIGURED)(.*?")', 'input_path: "{}"'.format(train_record_fname), s, count=1)
        s = re.sub(
            '(input_path: ".*?)(PATH_TO_BE_CONFIGURED)(.*?")', 'input_path: "{}"'.format(test_record_fname), s, count=1)

        # label_map_path
        s = re.sub(
            'label_map_path: ".*?"', 'label_map_path: "{}"'.format(label_map_pbtxt_fname), s)

        # Set training batch_size.
        s = re.sub('batch_size: [0-9]+',
                'batch_size: {}'.format(batch_size), s)

        # Set training steps, num_steps
        s = re.sub('num_steps: [0-9]+',
                'num_steps: {}'.format(num_steps), s)
        
        s = re.sub('total_steps: [0-9]+',
                'total_steps: {}'.format(num_steps), s)
        
        # Set number of classes num_classes.
        s = re.sub('num_classes: [0-9]+',
                'num_classes: {}'.format(num_classes), s)
        
        #fine-tune checkpoint type
        s = re.sub(
            'fine_tune_checkpoint_type: "classification"', 'fine_tune_checkpoint_type: "{}"'.format('detection'), s)
            
        f.write(s)



