#/usr/bin/sh

python feature_server.py feature_extractors/ImageNet.train_1000_copper.resnet.Resnet50_512_epoch85_0.67.model --gpu 0 &
python query_server.py index/caltech256_train_514.index --s3 https://s3-us-west-2.amazonaws.com/skelleher-photos &

