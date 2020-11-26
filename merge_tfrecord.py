#!/bin/python3  
import argparse
import tensorflow as tf
import os
from utils.ckpt_utils import download_ckpt, customized_ckpt_cofig

def get_all_records(path):
    """
    get all tf.records in subfolders 
    """
    train_path = path+"/train/"
    valid_path = path+"/valid/"
    train_folders = os.listdir(train_path)
    valid_folders = os.listdir(valid_path)
    filename = "/default.tfrecord"
    train_full_path_list = [train_path+f+filename for f in train_folders]
    valid_full_path_list = [valid_path+f+filename for f in valid_folders]
    return (train_full_path_list ,  valid_full_path_list)

def merge_tfrecord(list_of_records, type="train"):
    """
    merge all tfrecords in subfolders and export file 
    namely export.tfrecord
    """
    try:
        if type=="train":
            dataset = tf.data.TFRecordDataset(list_of_records)
            export_file_name = "./export_dataset/train.tfrecord"
            writer = tf.data.experimental.TFRecordWriter(export_file_name)
            writer.write(dataset)   
        else:
            dataset = tf.data.TFRecordDataset(list_of_records)
            export_file_name = "./export_dataset/valid.tfrecord"
            writer = tf.data.experimental.TFRecordWriter(export_file_name)
            writer.write(dataset)               
        return True
    except:
        return False

def __data_count(file):       
    """
    count the number of export.record to be able to split 
    training and testesting tfrecords
    """
    record_iterator = tf.compat.v1.io.tf_record_iterator(path=file)    
    count = 0
    for string_record in record_iterator:
        example = tf.train.Example()
        example.ParseFromString(string_record)
        count += 1        
    return count

def __train_test_split(dataset_size, split_rate):
    input_file = './export_dataset/export.tfrecord'
    full_dataset = tf.data.TFRecordDataset(input_file)    
    train_size = int(split_rate * dataset_size)        
    train_dataset = full_dataset.take(train_size) 
    test_dataset = full_dataset.skip(train_size)
    return  train_dataset, test_dataset

def __tfrecord_writer(dataset, type="train"):
    if type == "train":
        export_file_name = "./export_dataset/train.tfrecord"
    else:
        export_file_name = "./export_dataset/test.tfrecord"
    writer = tf.data.experimental.TFRecordWriter(export_file_name)
    writer.write(dataset)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--path', type=str, help='tfrecord path export from cvat')
    parser.add_argument('--batch_size', type=int, default=16 ,help='training batch size')
    parser.add_argument('--num_steps', type=int, default=25000 ,help='training steps')
    parser.add_argument('--num_classes', type=int, default=1 ,help='number of classes')    
    parser.add_argument('--model', type=str, default="SSD MobileNet V1 FPN 640x640" ,help='model selection')
    
    opt = parser.parse_args()    
    NOW_FULL_PATH = os.getcwd() + "/"

    if opt.path:        
        # dealing with dataset folder 
        from shutil import copyfile, rmtree        
        try:
            rmtree('./export_dataset')
        except:
            pass
        os.mkdir("./export_dataset")

        # dealing with tfrecord datasets ( with spliting into training, testing dataset)
        train_path, valid_path = get_all_records(opt.path)        
        labelmap = '/'.join(train_path[0].split("/")[:-1])
        labelmap = labelmap + "/label_map.pbtxt"
        copyfile(labelmap, "./export_dataset/label_map.pbtxt")        
        res = merge_tfrecord(list_of_records=train_path, type="train")
        res = merge_tfrecord(list_of_records=valid_path, type="valid")

        # if res == True:
        #     count = data_count("./export_dataset/export.tfrecord")           
        #     # print('total of count : ', count)
        #     train_dataset, test_dataset = train_test_split(dataset_size = count, split_rate=float(opt.ratio))
        #     tfrecord_writer(dataset=train_dataset, type="train")
        #     tfrecord_writer(test_dataset, type="test")
        # os.remove("./export_dataset/export.tfrecord")       
        print("dataset is ready")
        # check checkpoint exeists if not download it 
        model_name = download_ckpt(opt.model)
        # copy template (pipeline.config) to  export dataset folder
        copyfile(model_name + "/pipeline.config", "export_dataset/pipeline.config")      
        # get number of classes 

        customized_ckpt_cofig(
            pipeline_config_path="export_dataset/pipeline.config", 
            fine_tune_checkpoint= NOW_FULL_PATH + model_name + "/checkpoint/ckpt-0",
            train_record_fname=NOW_FULL_PATH + "export_dataset/train.tfrecord",
            test_record_fname=NOW_FULL_PATH + "export_dataset/test.tfrecord",
            label_map_pbtxt_fname=NOW_FULL_PATH + "export_dataset/label_map.pbtxt",
            batch_size=opt.batch_size,
            num_classes=opt.num_classes,
            num_steps=opt.num_steps
        )