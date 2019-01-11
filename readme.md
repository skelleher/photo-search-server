Ridley is a small photo search engine that I wrote for fun.

You can play with it online here: http://skelleher-apps.s3-website-us-west-2.amazonaws.com/

Ridley can index photos, and then execute queries against the index. 
By default, the top-5 matching photos are returned.

The engine is comprised of three Python processes, which can one on one or multiple servers:
* image feature extractor (deep neural net)
* image search (kNN)
* web front-end


![Search results](screenshots/ridley_photo_search_engine5.jpeg?raw=true)


Feature Extraction
------------------

Photos are passed through a slightly-modified Resnet50 (trained on ImageNet) to compute a feature vector.

Since Resnets have skip layers, the feature vector includes both high-level class information,
and low-level image information such as the specific pose, shape, and color of objects. This makes Resnet
more suitable for photo matching than, say, AlexNet.

I also append the predicted class label as an extra feature. This improves search results, both subjectively 
and quantitatively.

By combining the Resnet features and class label, search results 
* usually contain the correct subject and 
* are visually similar (size, pose, and color)

Although the Resnet was trained on Imagenet, using it for the search engine generalizes well to other images.
For example, indexing and searching the caltech-256 dataset returns relevent results (visually and per class).


Search
------

With a feature vector in hand, we use kNN to search for similar photos.  
It returns the top matches based on similarity - specifically, Euclidian distance 
between their descriptors. (scikit-learn kNN does not support cosine distance, and
writing a custom pairwise_match function made search 10x slower).  


Performance
-----------

Ridley uses a simple kNN to find matching images (the entire index fits in the RAM of one machine).
This is quite slow for large numbers of images.

For example, searching a database of 1.2 million ImageNet images takes ~1.2 seconds
on an Intel Core-i5 @ 3.3 Ghz.

The search engine could be scaled up to be much faster and support much larger datasets:
* the index could be split across multiple servers and searched in parallel
* locality-sensitive hashing would allow approximate nearest neighbors (searching only a subset of the index)


Further Work
------------

TODO: a neural net trained to classify images works OK, if you chop off the 
final layer (the classifier).  This leaves the penultimate layer, which 
outputs a feature vector - the image "fingerprint," or projection into feature 
space.

However matching may perform better if a final "embedding" layer is added to 
the net, and then fine-tuned.  For example, you could fine-tune the embedding 
layer using positive and negative image pairs, with a triplet loss function.

TODO: we use the kNN implementation from scikit-learn.  It performs exhaustive 
search of the index, which does not scale well to millions (or billions) of photos.  
A better solution might be approximate nearest neighbors, with locality-
sensitive hashing.  This could use the Annoy or HNSW libraries.

TODO: try different feature vector lengths. I've tried 2048 and 512 - both
work well.  We could use PCA to further shrink the vectors to 256 or 128.
In general, kNN matching performs better (in time and accuracy) when the 
vectors are shorter.


Install
-------

Install Python 3.x - preferably Anaconda or Miniconda.
Install required packages:
  while read requirement; do conda install --yes $requirement; done < requirements.txt


Start Feature Server
--------------------

> python feature_server.py <cnn model for feature extraction>

e.g.
  > python feature_server.py models/ImageNet.resnet50_0.70.model --gpu 0


Index Images for Search
-----------------------

> python index.py <path to folder> <index name>

e.g.
  > python index.py /data/caltech256/train/ caltech256.index


Start Query Server
------------------

> python query_server.py caltech256.index

Assumes the feature_server is running on 0.0.0.0:1975.  Otherwise, specify --features_host and --features_port.


Search for Similar Images
-------------------------

> curl -X POST http://hostname:port/v1/search -H "Content-type: application/octet-stream" --data-binary @<filenmame>

e.g.
  > curl -X POST http://localhost:1980/v1/search -H "Content-type: application/octet-stream" --data-binary @puppy_dog.jpg

