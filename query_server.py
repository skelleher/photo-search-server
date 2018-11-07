#!/usr/bin/env python

import os
import io
import sys
import argparse
import requests
import numpy as np
from database import Index
from PIL import Image
from stat import *
from flask import Flask, jsonify, request, _app_ctx_stack


# Load a previously-generated index of images
# The index is large (it contains full paths for each file), 
# so we want to only keep the features and keys (index into index)
# in RAM
#
# On query:
#   get features for image
#   perform kNN search for similar images

# Flask web server
_app   = Flask(__name__)
_index = None
_args  = None


def _main():
    # force print() to be unbuffered
    sys.stdout = os.fdopen(sys.stdout.fileno(), 'w')
   
    parser = argparse.ArgumentParser()
    parser.add_argument("index", help="previously generated index of images")
    parser.add_argument("--host", help="hostname to listen for queries. Defaults to 0.0.0.0 (visible externally!)", nargs="?", default="0.0.0.0")
    parser.add_argument("--port", help="port number to listen for queries", nargs="?", default=1980)
    parser.add_argument("--features_host", help="hostname for the feature_server. Defaults to 0.0.0.0", nargs="?", default="0.0.0.0")
    parser.add_argument("--features_port", help="port number for the feature_server.", nargs="?", default=1975)
    parser.add_argument("--metric", help="similariity metric [euclidean, cosine] default euclidean", nargs="?", default="euclidean")
    parser.add_argument("--width", help="resize image before extracting features", default=256)
    parser.add_argument("--height", help="resize image before extracting features", default=256)
    parser.add_argument("--verbose", "-v", help="print verbose query information", action="store_true")
   
    global _args
    _args = parser.parse_args()
    
    # Check if index exists
    if not os.path.exists(_args.index):
        print("Error: index %s not found" % _args.index)
        return -1
    
    # TODO: add signature to index (and verify it)
    mode = os.stat(_args.index).st_mode
    if S_ISDIR(mode):
        print("Error: %s is not a valid index file" % _args.index)
        return -1
    
    # Load the image Index
    # TODO: memory map it; index can be huge
    global _index
    _index = Index(_args)
    _index.load_index(_args.index)
    
    # Start the web server
    global _app
    _app.run(host = _args.host, port = _args.port)
    

#
# Get and release handles to the image database
#

def get_db():
    # Opens a new database connection if there is none yet for the
    # current application context.
    global _index

    top = _app_ctx_stack.top
    if not hasattr(top, "index"):
        top.index = _index
    return top.index


@_app.teardown_appcontext
def release_db(exception):
    global _index

    # Closes the database again at the end of the request.
    #top = _app_ctx_stack.top
    #if hasattr(top, "index"):
        #print("Released Index instance")



@_app.route("/")
def root():
    return "Ridley query server running."


@_app.route("/stats")
def index_stats():
    index = get_db()
    rows = index.index.shape[0]
    stats = "Index = %s\n%d rows\n%s" % (str(index.index.columns.values), rows, str(index.knn))
    print(stats)

    return stats


@_app.route("/query", methods=["POST"])
def query():
    global _args

    if _args.verbose:
        print("headers =\n", request.headers)


    if request.headers['Content-Type'] != 'application/octet-stream':
        return "415 Unsupported Media Type"    

    image_bytes = request.data
    image = Image.open( io.BytesIO(image_bytes) )

    if _args.verbose:
        print("Image = %s" % image)

    # perform kNN search using the features

    index = get_db()
    feature_vector = _get_feature_vector(image)
    results = index.query_image(feature_vector)

    return jsonify(results)


# Convert feature vector from string to array of floats
# works with the truncated 7.3 floats we write to the index
def _string_to_float_array(str):
    str = str.replace(" ", "")
    str = str.replace("\"", "")
    str = str.replace("[", "")
    str = str.replace("]", "")
    if str.endswith(','):
        str = str[0:-1]

    tokens = str.split(",")
    ar = np.empty(len(tokens))

    for i in range(len(tokens)):
        token = tokens[i]
        f = float( token )
        ar[i] = f

    return ar


# curl -X POST http://localhost:32817/featurize -H "Content-type: application/octet-stream" --data-binary @$@  
def _get_feature_vector(image):
   # Resize
   size = (_args.width, _args.height)
   img = image.resize(size)

   byte_array = io.BytesIO()
   img.save(byte_array, format='PNG')
   data = byte_array.getvalue()

   response = requests.post(url="http://" + _args.features_host + ":" + str(_args.features_port) + "/featurize",
                       data=data,
                       headers={'Content-Type': 'application/octet-stream'})
  
   #response = response.json()
   response = response.text

   features = _string_to_float_array(response)

   return features


if __name__ == "__main__":
    _main()

