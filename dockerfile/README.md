# [Reference from tensorflow api](https://github.com/tensorflow/models/tree/master/research/object_detection/dockerfiles/tf2)


## Building and Running
```cmd
docker build -f dockerfile/Dockerfile -t tf_object_detection .
docker run -it -v $(pwd):/home/tensorflow tf_objec_detection

```


