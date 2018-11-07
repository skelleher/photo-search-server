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
_model = None


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
    global _model
    _model, _, _ = Model.load( _args.model )

    # Disable batchnorm update and gradient history-keeping
    _model._model.eval()
    torch.set_grad_enabled( False )

    # HACK HACK:
    # Currently using a pretrained model, where the embedding layer has not been trained / fine-tuned. Remove it. SO GROSS.
##    _model._model = nn.Sequential(*list(_model._model.children())[:-1])
    #print(_model._model)

    layers = list(_model._model.children())
    print("Last layers of model:")
    for layer in layers[-2:]:
        print(" * ", layer)
    print("")


    # Load the model parameters: input size, normalization mean/std, embedding output vector length

    # Move model to the GPU if available
    # Lame that Model has a _model, which is exposed in a few places. Fix it.
    if torch.cuda.is_available() and _args.gpu is not None:                               
        _model._model = _model._model.cuda()                                       
    
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

def get_db():
    # Opens a new database connection if there is none yet for the
    # current application context.

    global _model 

    top = _app_ctx_stack.top
    if not hasattr(top, "model"):
        top.model = _model
        #print("Got model instance: ", top.model)
    return top.model


@_app.teardown_appcontext
def release_db(exception):
    global _model

    # Closes the database again at the end of the request.
    top = _app_ctx_stack.top
    #if hasattr(top, "model"):
        #print("Released model instance")



@_app.route("/")
def root():
    return "Ridley image server running."


@_app.route("/stats")
def index_stats():
    _model = get_db()
    stats = "Model = %s\nembedding length = %d\n%s" % (_model.name, _model.embedding_length)
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
        model = get_db()
        vector = _get_feature_vector( model, image_bytes )
        stop = time.time()
        msecs = (stop - start) * 1000
        print("%d ms: %s bytes -> %d" % (msecs, request.headers["Content-Length"], len(vector)))

        #if _args.verbose:
        #    print(vector)

        return jsonify(vector)

    else:
        return "415 Unsupported Media Type"    


def _get_feature_vector( model, image_bytes ):
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
    batch = model.forward( batch )
    embedding = batch.squeeze(0)

    # TODO: do we want to normalize the feature vector for better kNN matching?
    # e.g. apply a StandardScaler to it?

    if _args.verbose:
        print("embedding = ", embedding.shape)

    # convert result to an array so we can send it back to client as JSON
    embedding = embedding.detach().tolist()

    return embedding


if __name__ == "__main__":
    _main()

