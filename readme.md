Ridley is a small photo search engine that I wrote for fun.

Ridley can index photos, and then execute queries against
the index. By default, the top-5 matching photos are returned.

![Search results](screenshots/ridley_photo_search_engine5.jpeg?raw=true)

Ridley uses a deep neural net to compute photo descriptors.


Feature Extraction
------------------

Photos are passed through a Resnet50 (trained on ImageNet).

Time to compute image features:
* GPU - Titan X: 10 - 15 ms
* CPU - Core i5-4590 3.3 Ghz: 100 - 200 ms

Unlike some search engines, we use ***both the predicted class label and the output 
of the second-to-last layer*** (the "deep features" of the image).

Since this is a Resnet, the deep feature vector includes information copied forward 
from early layers of the model, such as the specific shape, color, and pose of objects.

If we use only the predicted class labels, photos of airplanes match photos of any 
other airplane, regardless of shape or pose (the image features have been lost).

If we use only the image features, photos of red round things tend to match other 
photos of red round things, regardless of subject. At least this is true when using
Euclidian distance.  I speculate that Resnet features don't cluster well (need to 
confirm with PCA or t-SNE). So kNN cannot dis-entangle the image subject from the deep 
features. A fully-connected layer is required for that.

By combining the features and class label, we get search results that a) contain the 
correct subject and b) are visually similar (size, pose, and color).


Search
------

With an image descriptor in hand, we use kNN to search for similar photos.  
It returns the top matches based on similarity - specifically, Euclidian distance 
between their descriptors. (scikit-learn kNN does not support cosine distance, and
writing a custom pairwise_match function made search 10x slower)


Performance
-----------

Ridley uses a simple kNN to find matching images; this is quite slow for large 
numbers of images.

For example, searching a database of 1.2 million ImageNet images takes ~1.2 seconds
on an Intel Core-i5 @ 3.3 Ghz.


Further Work
------------

TODO: a neural net trained to classify images works OK, if you chop off the 
final layer (the classifier).  This leaves the penultimate layer, which 
outputs a feature vector - the image fingerprint, or projection into feature 
space.

However matching should perform better if a final "embedding" layer is added to 
the net, and then fine-tuned.  For example, you could fine-tune the embedding 
layer using positive and negative image pairs, with a triplet loss function.
Or, perhaps you could fine-tune the embedding layer on image captions (text).

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

