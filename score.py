#/usr/bin/env python

# For every folder in destination:
#   for every file in folder:
#       get classname, filename
#       query results
#       for result[1..5]
#           if result == class, +1 top-1, top-5

import os
import sys
import argparse
from stat import *
import requests
import json
from PIL import Image


def _main():
    # Disable buffering on stdout
    sys.stdout = os.fdopen(sys.stdout.fileno(), 'w')

    parser = argparse.ArgumentParser()
    parser.add_argument("input", help="folder or file to query.  Will be queried recursively, with accuracy scored")
    parser.add_argument("--host", help="hostname of query server", nargs="?", default="localhost")
    parser.add_argument("--port", help="port of query server", nargs="?", type=int, default=1975)
    parser.add_argument("--show", help="show query results in a popup window", action="store_true")
    parser.add_argument("--summary", help="don't show individual results; only print per-class summaries", action="store_true")
    #parser.add_argument("output", help="filename to store results.")

    args = parser.parse_args()
    path = args.input

    # Remove trailing slash from path
    path = path.rstrip(os.path.sep)

    # Check if input path exists.
    if not os.path.exists(path):
        print("Error: %s not found" % path)
        return -1

    total, top_1, top_5 = query(path, args)
    top_1_accuracy = (100.0 * float(top_1) / float(total)) if top_1 else 0.0
    top_5_accuracy = (100.0 * float(top_5) / float(total)) if top_5 else 0.0

    print("\nSummary")
    print("=======\n")
    print("Total = %d top-1 = %d top-5 = %d" % (total, top_1, top_5))
    print("Accuracy (top-1) = %6.2f" % top_1_accuracy) 
    print("Accuracy (top-5) = %6.2f" % top_5_accuracy)


def query(path, args):
    total = 0
    top_1 = 0
    top_5 = 0
    
    # Check if input path exists.
    if not os.path.exists(path):
        print("Error: %s not found" % path)
        return -1

    if os.path.isfile(path):
        (total, top_1, top_5) = query_file(path, args) 
    elif os.path.isdir(path):
        #args.show = False
        (total, top_1, top_5) = query_folder(path, args)
    
    return total, top_1, top_5


# Recursively calls itself for all subfolders
def query_folder(input_path, args):
    print("query_folder: %s" % input_path)

    total = 0
    top_1 = 0
    top_5 = 0

    for name in os.listdir(input_path):
        path = input_path + os.path.sep + name

        if name[0] == '.':
            continue

        if os.path.isfile(path):
            (q_total, q_top_1, q_top_5) = query_file(path, args) 
        if os.path.isdir(path):
            (q_total, q_top_1, q_top_5) = query_folder(path, args)

        total += q_total
        top_1 += q_top_1
        top_5 += q_top_5

    #print("\n")

    top_1_accuracy = (100.0 * float(top_1) / float(total)) if top_1 else 0.0
    top_5_accuracy = (100.0 * float(top_5) / float(total)) if top_5 else 0.0
    print("Total = %d top-1 = %d top-5 = %d" % (total, top_1, top_5))
    print("Accuracy (top-1) = %6.2f" % top_1_accuracy) 
    print("Accuracy (top-5) = %6.2f" % top_5_accuracy)



    return total, top_1, top_5


def query_file(input_path, args):
    if not args.summary:
        print("query_file: %s" % input_path)

    total = 1
    top_1 = 0
    top_5 = 0


    #get classname, filename
    #query results
    #for result[1..5]:
    #    if result == class, +1 top-1, top-5

    # ASSUMES the parent folder is the classname, e.g. dog\small_dog_1234.jpg
    elements  = input_path.split(os.sep)
    filename  = input_path 
    classname = elements[-2]

    if not args.summary:
        print("class = [%s] file = [%s]" % (classname, filename))

    # curl -X POST http://localhost:1975/query -H "Content-type: application/octet-stream" --data-binary @$@
    url = "http://%s:%d/query" % (args.host, args.port)
    headers = { "content-type" : "application/octet-stream" }

    try:
        payload = open(input_path, "rb").read()
    except Exception as ex:
        print("Error loading file %s" % input_path)
        print(type(ex))
        print(ex.args)
        print(ex)
        pass

    reply = requests.post(url, data=payload, headers=headers)
    #print(reply.status_code)
    #print(reply.text)

    try:
        matches = json.loads(reply.text)
    except Exception:
        return total, top_1, top_5


    files = [input_path]
    for i in range(len(matches)):
        _classname = matches[i]["class"]
        _filename  = matches[i]["filename"]
        files.append(_filename)

        if not args.summary:
            print("class = [%s] filename = [%s]" % (_classname, _filename))


        if _classname == classname:
            if i == 0:
                top_1 = 1

            if i < 5:
                top_5 = 1


    if args.show:
        _show_images(files)

    return total, top_1, top_5


def _show_images(files):

    height      = 128
    width       = 128
    num_files   = len(files) + 1 # leave space for the "equals" sign
    pad_w       = 10
    pad_h       = 10
    grid_width  = int(pad_w + (num_files * (width + pad_w)) + pad_w)
    grid_height = int(pad_h/2 + height + pad_h/2)
    left        = pad_w


    grid = Image.new("RGBA", (grid_width, grid_height), color=(255,255,255,0))

    # Display the query image on left
    with Image.open( files.pop(0) ) as image:
        image = image.resize((height, width), resample = Image.BILINEAR)
        grid.paste(image, box = (left, int(pad_h/2)))
        left += width + (2 * pad_w) 

    with Image.open( "equals.png" ) as image:
        image = image.resize((height, width), resample = Image.BILINEAR)
        grid.paste(image, box = (left, int(pad_h/2)))
        left += width + (2 * pad_w) 

    # Display the query results on right
    for file in files:
        with Image.open(file) as image:
            image = image.resize((height, width), resample = Image.BILINEAR)
            grid.paste(image, box = (left, int(pad_h/2)))
            left += width + pad_w 

    grid.show()



 
if __name__ == "__main__":
    _main()

