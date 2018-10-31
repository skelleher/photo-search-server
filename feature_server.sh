#!/bin/sh

#python feature_server.py ImageNet.train_1000_resnet50_0.70.model.copper.resnet.ResnetCBIR $@
python feature_server.py cbir.model $@
