#!/bin/bash
#python query_server.py --host 0.0.0.0 --port 1980 $@
python query_server.py index/exo_train_514.index --s3 https://s3-us-west-2.amazonaws.com/skelleher-photos

