#!/usr/bin/env python

from flask import Flask, jsonify, request, _app_ctx_stack
from PIL import Image
import argparse
import io
import os
import sys
import time

import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn

from copper.model import Model
from copper import utils


# 
# Load feature extraction model
# Chop off last layer (classifier) if needed
# Add embedding layer of requested output size (e.g. project feature vector 2048 onto embed vector 1024)
# Perform sanity check: pass image through feature extractor and output the feature fector
#
# TODO: instead of using raw deep features from a classification model, add and fine-tune an embedding layer using e.g. triplet loss
# TODO: output both embedding and classification, and use the highest class labels to restrict the image search to a subset of index
# TODO: use approximate nearest neighbors, e.g. Annoy library: https://github.com/spotify/annoy
# TODO: use spatial hashing of embedding to restrict the image search to a subset of index
# TODO: generate an image caption and use that to aid image search
# TODO: based on highest class score, select only the embedding weights which contribute to that class to create a new embedding:
#   https://blog.insightdatascience.com/the-unreasonable-effectiveness-of-deep-learning-representations-4ce83fc663cf
#


#
# Flask web server
#

_app   = Flask(__name__)
_args  = None
_feature_extractor = None
_classifier = None


def _main():
    # force print() to be unbuffered
    sys.stdout = os.fdopen(sys.stdout.fileno(), 'w')
   
    # TODO: recover the feature descriptor mode and vector length from the model, not a command line

    parser = argparse.ArgumentParser()
    parser.add_argument("model", help="CNN model to use for image embedding")
    parser.add_argument("--host", help="hostname to listen for queries. Defaults to 0.0.0.0 (visible externally!)", nargs="?", default="0.0.0.0")
    parser.add_argument("--port", help="port number to listen for queries", nargs="?", default=1975)
    parser.add_argument("--verbose", "-v", help="print verbose query information", action="store_true")
    parser.add_argument("--gpu", help="GPU to use for feature extraction, 0-based", type = int, default = None) 

    global _args
    _args = parser.parse_args()
   
    cudnn.benchmark = True
    print("CUDA support: ", torch.cuda.is_available())
                                                        
    if torch.cuda.is_available and _args.gpu is not None:
        torch.cuda.set_device( _args.gpu )
        print( "Using GPU: ", _args.gpu )
    else:
        print( "WARNING: Using CPU" )

    # Check if model exists
    if not os.path.exists(_args.model):
        print("Error: model %s not found" % _args.model)
        return -1

    # Load the model; ignore optimizer state and command-line used to train the model (we are not fine-tuning the model)
    global _feature_extractor
    _feature_extractor, _, _ = Model.load( _args.model )

    # Disable batchnorm update and gradient history-keeping
    _feature_extractor._model.eval()
    torch.set_grad_enabled( False )

    # Remove the final layer (classifier) but save it so we can generate both a deep feature vector, and a class vector.
    # Test using both deep feature vector and class vector for image description.
    # The class vector will obviously match photos of same class.
    # The feature vector matches photos with visual similarity.
    # A weighted blend might yield subjectively better search results.
    # NOTE: we replace the final layer with an identity layer. This is MUCH easier than deleting it, which breaks forward()
    global _classifier
    _classifier = _feature_extractor._model.fc
    _feature_extractor._model.fc = torch.nn.Sequential() 

    print("classifier = ", _classifier)
    layers = list(_feature_extractor._model.children())
    print("Last layers of model:")
    for layer in layers[-2:]:
        print(" * ", layer)
    print("")

    # Move model to the GPU if available
    # Lame that Model has a _model, which is exposed in a few places. Fix it.
    if torch.cuda.is_available() and _args.gpu is not None:                               
        _feature_extractor._model = _feature_extractor._model.cuda()
        _classifier = _classifier.cuda()
    
    # Sanity check: load three images and compute their feature vectors.
    # Confirm they are not identical.
    #
    # Unfortunately I have a lot of old saved models which, because they were saved incorrectly,
    # load with only a warning about "wrong number of weights / layers".
    # These essentially output a constant vector - with some noise - regardless of the input.
    # Won't notice unless you run stats on the feature vectors.

    # Start the web server
    global _app
    _app.run(host = _args.host, port = _args.port)
    

#
# Get and release handles to the image database
#

@_app.route("/")
def root():
    return "Ridley image server running."


@_app.route("/stats")
def index_stats():
    stats = "Model = %s\nembedding length = %d\n%s" % (_feature_extractor.name, _feature_extractor.embedding_length)
    print(stats)

    return stats


@_app.route("/featurize", methods=["POST"])
def featurize():
    global _args

    if _args.verbose:
        print("headers =\n", request.headers)

    if request.headers["Content-Type"] == "application/octet-stream":
        image_bytes = request.data

        # Perform forward pass to extract the image embedding vector
        start = time.time()
        vector = _get_feature_vector( _feature_extractor, _classifier, image_bytes )
        stop = time.time()
        msecs = (stop - start) * 1000
        print("%d ms: %s bytes -> %d" % (msecs, request.headers["Content-Length"], len(vector)))

        #if _args.verbose:
        #    print(vector)

        return jsonify(vector)

    else:
        return "415 Unsupported Media Type"    


def _get_feature_vector( feature_extractor, classifier, image_bytes ):
    # Convert image_bytes to an Image for easy resize/crop
    image = Image.open( io.BytesIO(image_bytes) )

    if _args.verbose:
       print("Image = %s" % image)

    # Resize and crop the input image to match what the model expects
    # TODO: the model should tell us its input dims!
    model_width = 256
    model_height = 256
    scale_width = model_width / image.width
    scale_height = model_height / image.height
    scale = max(scale_width, scale_height)
    new_size = (int)(scale * image.width), (int)(scale * image.height)

    if _args.verbose:
        print("scale image %d x %d -> %d x %d" % (image.width, image.height, new_size[0], new_size[1]))

    image = image.resize( new_size );

    if image.mode != "RGB":
        image = image.convert("RGB")

    # extract a center crop; returns a tensor
    # TODO: the model should tell us its input dims!
    crop = utils.image_crop( image, 224, 224 )

    # create a minibatch of size 1 (PyTorch models only accept minibatches)
    batch = crop.unsqueeze(0)

    if torch.cuda.is_available() and _args.gpu is not None:
        batch = batch.cuda()

    # perform forward pass
    # we generate two vectors: the image features, and the class predictions
    # both together may yield better image description than either alone
    features = feature_extractor.forward( batch )
    raw_output  = classifier.forward( features )
    
    labels, probabilities = Model.get_predictions( raw_output )

    raw_output = raw_output.squeeze(0)
    features = features.squeeze(0)


    # TODO: do we want to normalize the feature vector for better kNN matching?
    # e.g. apply a StandardScaler to it?

    if _args.verbose:
        print("features = ", features.shape)
        print(features)
        print("raw_output = ", raw_output.shape)
        print(raw_output)
        print("labels = ", labels)
        print("probabilities = ", probabilities)

    # convert result to an array so we can send it back to client as JSON
    features = features.detach().tolist()

    # TEST: append the class label and probability to see how it affects image clustering / search
    # need to cast because Python JSON can't serialize 64-bit numbers
    #
    # PROBLEM: is this an acceptable way to encode a categorical feature for Euclidian distance?
    # We don't want the class label to totally overwhelm the other features.
    # Generally speaking, we SHOULD be using one-hot encoding but then our vector is high-dimensional, which
    # hurts kNN.
    features.insert( 0, float(labels[0] / len(raw_output)) )
    features.insert( 1, float(probabilities[0]) ) 

    return features


if __name__ == "__main__":
    _main()

