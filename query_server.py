#!/usr/bin/env python

import os
import io
import sys
import argparse
import requests
import numpy as np
from database import Database
from PIL import Image
from stat import *
from flask import Flask, jsonify, request
from flask_restful import Resource, Api, reqparse


# Load a previously-generated Database of images
# The Database is large (it contains feature vectors as ASCII strings, and full paths for each file), 
# so we want to only keep the features and keys (index into database) in RAM
#
# On query:
#   get features for image
#   perform kNN search for similar images

# Flask web server
_app = None
_api = None 
_args = None
_database = None

#
# REST resources
#

_image_list_parser = reqparse.RequestParser()
_image_list_parser.add_argument("image_url", type = str, help = "filename to use as input to photo search engine")

class ImageSearchResource(Resource):
    def post(self):
        args = _image_list_parser.parse_args()
        print("args = ", args)
        print("request = ", request)
        print("headers = ", request.headers)

        if request.headers['Content-Type'] != 'application/octet-stream':
            return "415 Unsupported Media Type"    

        image_bytes = request.data
        image = Image.open( io.BytesIO(image_bytes) )

        feature_vector = _get_feature_vector(image)
        results = _database.query_image(feature_vector)

        return jsonify(results)


class ImageListResource(Resource):
    # TODO: support POST for uploading images
    # save to temp file
    # run classifier to get class
    # if prob < threshold class = unknown
    # move file to /data/ridley/uploads/<class>
    # extract features
    # append to ridley_uploads.index
    # OPTIONAL: concat databases and restart the query server

    def get(self):
        return { "num_images" : len(_database) }
    

class ImageResource(Resource):
    def get(self, image_id = None):
        # TODO: do we need to protect database access with a lock or is flask_restful single-threaded?
        item = _database[ image_id ]

        classname, filename, features = item.split(",")
        classname = classname.strip()
        filename = filename.strip()
        features = features.strip()

        print("/images/%d = %s" % (image_id, filename))

        return { 
                image_id : 
                    { 
                        "class" : classname,
                        "filename" : filename,
                        "features" : features
                    },
                }



def _main():
    parser = argparse.ArgumentParser()
    parser.add_argument("database", help="database of images")
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
    
    # Check if database exists
    if not os.path.exists(_args.database):
        print("Error: database %s not found" % _args.database)
        return -1
    
    # TODO: add signature to database (and verify it)
    mode = os.stat(_args.database).st_mode
    if S_ISDIR(mode):
        print("Error: %s is not a valid database file" % _args.database)
        return -1
    
    # Load the image Database
    # TODO: memory map it; database can be huge
    global _database
    _database = Database(_args)
    _database.load_database(_args.database)
    
    # Start the web server
    global _app
    global _api
    _app = Flask(__name__)
    _api = Api(_app)

    _api.add_resource(ImageListResource,
            "/v1/images",
            "/v1/images/")

    _api.add_resource(ImageResource,
            "/v1/images/<int:image_id>")

    _api.add_resource(ImageSearchResource,
            "/v1/search",
            "/v1/search/")

    _app.run(host = _args.host, port = _args.port)
    


#@_app.route("/")
#def root():
#    return "Ridley query server running."


#@_app.route("/stats")
#def database_stats():
#    rows = _database.index.shape[0]
#    stats = "Database = %s\n%d rows\n%s" % (str(database.index.columns.values), rows, str(database.knn))
#    print(stats)
#
#    return stats


#@_app.route("/query", methods=["POST"])
#def query():
#    global _args
#
#    if _args.verbose:
#        print("headers =\n", request.headers)
#
#
#    if request.headers['Content-Type'] != 'application/octet-stream':
#        return "415 Unsupported Media Type"    
#
#    image_bytes = request.data
#    image = Image.open( io.BytesIO(image_bytes) )
#
#    if _args.verbose:
#        print("Image = %s" % image)
#
#    # perform kNN search using the features
#
#    feature_vector = _get_feature_vector(image)
#    results = _database.query_image(feature_vector)
#
#    return jsonify(results)


# Convert feature vector from string to array of floats
# works with the truncated 7.3 floats we write to the database
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
  
   response = response.text

   features = _string_to_float_array(response)

   return features


if __name__ == "__main__":
    _main()

