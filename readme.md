Ridley is a small photo search engine.

Ridley can index photos, and then execute queries against
the index. By default, the top-5 matching photos are returned.

![Search results](screenshots/ridley_photo_search_engine5.jpeg?raw=true)

Ridley uses a deep neural net to compute photo descriptors. 
So it's not just looking at properties like color or the position of objects;
in some sense Ridley understands the semantic content of each photo.

Ridley uses a simple kNN for finding matching images; this is quite slow for 
large number of images.  For example, searching a database of 1.2 million 
ImageNet images takes ~1.2 seconds on an Intel Core-i5 @ 3.3 Ghz.


In more detail:

Photos are passed through a resnet50 (trained on 1.2 million ImageNet photos).
The neural net performs feature extraction to generate a vector of numbers - 
the image's visual fingerprint. Similar images should have similar feature 
vectors, and they should cluster together in feature space.

We use kNN to perform the actual search.  It returns the top matches based on
similarity - specifically, Euclidian distance between their feature vectors.
(scikit-learn kNN does not support cosine distance as a similarity metric)

The convolutional neural net runs faster on a GPU, but can also run on a CPU.
Time to index a photo (GPU): 10 - 15 ms
Time to index a photo (CPU): 100 - 200 ms

The time to query a photo includes feature extraction, and depends on the
database size.

TODO: a neural net trained to classify images works OK, if you chop off the 
final layer (the classifier).  This leaves the penultimate layer, which 
outputs a feature vector - the image fingerprint, or projection into feature 
space.  

However matching should perform better if a final "embedding" layer is added to 
the net, and then fine-tuned.  For example, you could fine-tune the embedding 
layer using positive and negative image pairs, with a triplet loss function.
Or, you could fine-tune the embedding layer on image captions (text) such as 
the MS-COCO dataset.

TODO: we use the kNN implementation from scikit-learn.  It performs exhaustive 
search of the index, which does not scale well to millions of photos.  
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


Start Query Server
------------------

> python query_server.py

Assumes the feature_server is running on 0.0.0.0:1975.  Otherwise, specify --features_host and --features_port.


Index Images for Search
-----------------------

> python index.py <path to folder> <index name>

e.g.
  > python index.py /data/caltech256/train/ caltech256.index


Search for Similar Images
-------------------------

> curl -X POST http://<query host>:<query port>/v1/search -H "Content-type: application/octet-stream" --data-binary @<filenmame>

e.g.
  > curl -X POST http://localhost:1980/v1/search -H "Content-type: application/octet-stream" --data-binary @puppy_dog.jpg

