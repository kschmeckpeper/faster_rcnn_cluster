# faster_rcnn_cluster
Faster rcnn modified to run on the cluster


# Usage
This code runs in the following docker image.

```chaneyk/faster-rcnn-caffe```

To train the network, use the following command

```./tools/train_faster_rcnn_alt_opt.py --gpu 0 --net_name ZF --weights data/imagenet_models/ZF.v2.caffemodel --imdb voc_2007_trainval --cfg experiments/cfgs/faster_rcnn_alt_opt.yml```

Replace the imdb with the correct dataset.  If you are using a dataset that does not already exist, you may need to the dataset factory in ```lib/datasets/factory.py```.
