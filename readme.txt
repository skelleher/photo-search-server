Ridley is a small photo search engine.

Ridley can index photos (tens of thousands), and then execute queries against
the index. By default, the top-5 matching photos are returned.

Ridley uses a neural net to match photos.  
So it's not just looking at properties like color or the position of objects;
in some sense Ridley understands the semantic content of each photo.


In more detail:

Photos are passed through a resnet50 (trained on 1.2 million ImageNet photos).
The neural net peforms "feature extraction" to generate a vector of numbers - 
the photo's visual fingerprint. Similar images should have similar feature 
vectors, and they should cluster together in feature space.

We use kNN to perform the actual search.  It returns the closest matches based 
on similarityy - specifically, cosine similarity of their feature vectors.

The convolutional neural net runs faster on a GPU, but can also run on a CPU.
Time to index a photo (GPU): 10 - 15 ms
Time to index a photo (CPU): 100 - 200 ms

The time to query a photo includes feature extraction 

TODO: a neural net trained to classify images works OK, if you chop off the 
final layer (the classifier).  This leaves the penultimate layer, which 
outputs a feature vector - the image fingerprint, or projection into feature 
space.  

However matching will perform better if a final "embedding" layer is added to 
the net, and then fine-tuned.  For example, you could fine-tune the embedding 
layer using positive and negative image pairs, with a triplet loss function.
Or, you could fine-tune the embedding layer on image captions (text) such as 
the MS-COCO dataset.

TODO: we use the kNN implementation from Scikit-learn.  It performs exhaustive 
search of the index, which does not scale well to millions of photos.  
A better solution might be approximate nearest neighbors, such as spatial 
hashing or the Annoy library.  In general, kNN matching performs better (in 
time and accuracy) when the vectors are shorter (1000 elements or fewer).


Install
-------

Install Python 3.x - preferably Anaconda or Miniconda.
Install required packages:
  while read requirement; do conda install --yes $requirement; done < requirements.txt


Start Feature Server
--------------------

python feature_server.py <cnn model for feature extraction>

e.g.
  python feature_server.py /data/models/ImageNet/ImageNet.train_1000_resnet50_0.70.model --gpu 0


Start Query Server
------------------

python query_server.py

Assumes the feature_server is running on 0.0.0.0:1975.  Otherwise, specify --features_host and --features_port.


Index Images for Search
-----------------------

python index.py <path to folder> <index name>

e.g.
  python index.py /Volumes/share/data/caltech256/train/ caltech256.index


Search for Similar Images
-------------------------

curl -X POST http://<query host>:<query port>/query -H "Content-type: application/octet-stream" --data-binary @<filenmame>

e.g.
  curl -X POST http://localhost:32816/query -H "Content-type: application/octet-stream" --data-binary @puppy_dog.jpg

