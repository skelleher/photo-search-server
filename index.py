#!/usr/bin/env python

import os
import io
import sys
import argparse
import requests
import numpy as np
from collections import namedtuple
from PIL import Image
from stat import *
from base64 import *

_args = None

_files_to_ignore = [
    "@eaDir",
    ".DS_Store",
    "._.DS_Store",
]



def _main():
    parser = argparse.ArgumentParser()
    parser.add_argument("input", help="folder to index.  Will be indexed recursively, computing features for every .PNG or .JPEG")
    parser.add_argument("output", help="filename to store index.")
    parser.add_argument("--host", help="hostname for the image feature extraction server", type = str, default="localhost")
    parser.add_argument("--port", help="port for the image feature extraction server", default=1975)
    parser.add_argument("--width", help="resize image to <width>", nargs="?", type=int, default=256)
    parser.add_argument("--height", help="resize image to <height>", nargs="?", type=int, default=256)
    parser.add_argument("--force", help="force overwrite of existing index file", action="store_true")
    parser.add_argument("-v", "--v", help="verbose logging", dest = "verbose", action="store_true")

    global _args
    _args = parser.parse_args()

    # Check if input path exists.
    if not os.path.exists(_args.input):
        print("Error: %s not found" % _args.input)
        return -1

    # Check if output file exists.
    if os.path.exists(_args.output) and not _args.force:
        print("Error: %s exists; use --force to overwrite" % _args.output)
        return -1

    print("Indexing photos: %d x %d" % (_args.width, _args.height))

    with open(_args.output, "wt") as index:
        # Don't print spaces between column names; confuses Pandas
        index.write("classname,filename,features\n")

        # Index a single file
        if os.path.isfile(_args.input):
            index_file(_args.input, index, _args)
            index.flush()
            index.close()
            return;

        # Index folder(s)
        for name in os.listdir(_args.input):
            if name[0] == '.':
                continue

            path = _args.input + os.path.sep + name

            if os.path.isfile(path):
                index_file(path, index, _args) 
            elif os.path.isdir(path):
                index_folder(path, index, _args)
    
        index.flush()
        index.close()


# Recursively calls itself for all subfolders
def index_folder(input_path, index, args):
    print("index_folder: %s" % input_path)

    for name in os.listdir(input_path):
        if _ignore_file( input_path ):
            continue

        path = input_path + os.path.sep + name

        if os.path.isfile(path):
            index_file(path, index, args) 
        if os.path.isdir(path):
            index_folder(path, index, args)

    print("\n")


def index_file(input_path, index, args):
    # extract feature vector from file (if an image)

    if _ignore_file( input_path ):
        return

    # TODO: build a path lookup table to avoid duplicating path strings a million times
    elements  = input_path.split(os.sep)
    filename = input_path # Save the entire file path, so we can load the original image on query match.
    classname = elements[-2]

    try:
        image = Image.open(input_path)

        X = _get_feature_vector(image)
   
        if args.verbose:
            print("[%16s] %32s" % (classname, filename))

        sys.stdout.write(".")
        sys.stdout.flush()
    
        # append a row to the index
        # TODO: write features in binary instead of a .csv file: huge savings
        index.write("%-32s, " % classname)
        index.write("%-128s, " % filename)

        for val in X:
            index.write("%11.6f " % float(val))

        index.write("\n")

    except Exception as ex:
        print("Error loading image %s" % input_path)
        print(type(ex))
        print(ex.args)
        print(ex)
        pass


def _ignore_file( name ):
    for ignore in _files_to_ignore:
        if ignore in name:
            return True
        if name[0] == '.':
            return True

    return False


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


def _get_feature_vector(image):
    # Resize
    size = (_args.width, _args.height)
    img = image.resize(size)

    byte_array = io.BytesIO()
    img.save(byte_array, format='PNG')
    data = byte_array.getvalue()

    response = requests.post(url="http://" + _args.host + ":" + str(_args.port) + "/v1/image_features",
                            data=data,
                            headers={'Content-Type': 'application/octet-stream'})
 
    response = response.text

    features = _string_to_float_array(response)

    return features



if __name__ == "__main__":
    _main()

