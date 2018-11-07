import os
import sys
import argparse
import pandas as pd
import numpy as np
from stat import *
from sklearn.neighbors import NearestNeighbors
from sklearn import metrics

#
# Load an index of images, for later retrieval
#

class Index(object):
    def __init__(self, args):
        self.args = args
        #self.rect = Rect(0, 0, args.width, args.height) 
        print("Index: %s" % (str(args)))


    # We save the features as truncated floats (aaa.bbb)
    # so we need a special conversion function to load them.
    @staticmethod
    def _string_to_float_array(str):
        tokens = str.split()
        ar = np.empty(len(tokens))
    
        for i in range(len(tokens)):
            f = float( tokens[i] )
            ar[i] = f

        return ar


    def load_index(self, index):
        print("Index: loading index %s" % index)
        #sys.stdout.flush()
    
        try:
            if not os.path.exists(index):
                print("Error: %s not found" % index)
                return -1
       
            # TODO: prefix index with signature
            mode = os.stat(index).st_mode
            if S_ISDIR(mode):
                print("Error: %s is not a valid index" % index)
                return -1
                    
        except Exception:
            print("Error loading index %s" % index)
            print(type(ex))
            print(ex.args)
            print(ex)
            pass
       
        # Load Index, but discard the filenames column
        # Should we use Pandas?  Is it efficient for huge files?
        # can we memory-map large files from Python?
        # Load only the classname and features
        # At query time, when we want filenames, we can fetch just the ones we want (by index)
        self.index = pd.read_csv(index, usecols = ("classname", "features"))
   
        if (self.index.columns.values != ("classname","features")).any():
            print("Index has unexpected columns: %s" % str(self.index.columns.values))
            return -1

        # Convert from big string to ndarray of floats
        self.index["features"] = self.index["features"].apply(Index._string_to_float_array)

        # Reshape the DataFrame to a [rows x cols] ndarray that knn requires
        # This is SUPER-HACKY, but it was late on a Saturday night and I needed to move forward.
        self.num_rows = len(self.index)
        self.num_features = len(self.index["features"][0])

        new_X = np.ndarray(shape=[self.num_rows, self.num_features])

        X = self.index["features"].as_matrix()
        for i in range(len(X)):
            f = X[i] # f is a list
            new_X[i] = f.reshape(1, -1)

        X = new_X
        print("Loaded %d rows, %d features" % (len(self.index), self.num_features))

        # Open the database
        self._db = open(self.args.index, "rt")


        print("Loading kNN...")
        if self.args.metric == "cosine":
            self.knn = NearestNeighbors(n_neighbors=5, algorithm="auto", metric=metrics.pairwise.cosine_distances, n_jobs=1)
        else:
            self.knn = NearestNeighbors(n_neighbors=5, algorithm="auto", metric="euclidean", n_jobs=1)

#        self.knn = NearestNeighbors(n_neighbors=5, algorithm="ball_tree", metric="euclidean", n_jobs=4)         # also consider Chebyshev
        self.knn.fit(X)
        print(self.knn)


    # Index singleton, for use with a Flask web server, since Flask is stateless between REST calls
    def get_instance(self):
        return self


    def shape(self):
        return (self.num_rows, self.num_features)


    def query_image(self, feature_vector, k=5):
        if self.args.verbose:
            print("query_image: k=%d" % k)

        X = feature_vector

        if self.args.verbose:
            print("feature vector = %s" % str(X.shape))

        X = X.reshape(1, -1)

        neighbors = self.knn.kneighbors(X, k, return_distance=True)
        distances = neighbors[0].squeeze()
        matches = neighbors[1].squeeze()
    
        # returns a list of indices into the database.
        # fetch the filenames and classes and return those
#        db = open(self.args.index, "rt")

        results = []
        for i in range(len(matches)):
            idx = int(matches[i])

            # TODO: seek to database[idx] and read the entry
            # TODO: generate a lookup table (and index into the index >.<) that tells us where to fseek() to.
            item = self._get_item_from_database( idx )

            #print("match[%d] = %d = %s" % (i, idx, item))

            classname, filename, _ = item.split(",")
            classname = classname.strip()
            filename = filename.strip()
            #print("neighbor[%d]: %s %s" % (idx, classname, filename))

            results.append({"idx": idx, "class" : classname, "filename" : filename, "distance" : distances[i]})

        return results


    def _get_item_from_database( self, idx ):
        # TODO: seek to database[idx] and read the entry
        # TODO: generate a lookup table (and index into the index >.<) that tells us where to fseek() to.
        header_offset = 28
        features_offset_in_row = 165 # seek past class ID and filename to find the feature vector
        feature_size = 12 # size of a feature vector elelemnt: float printed as %11.6f, plus a space between each feature

        hack_offset = features_offset_in_row + (feature_size * self.num_features)
        #print("num_features = %d, hack_offset = %d" % (self.num_features, hack_offset))

        file_offset = header_offset + (idx * hack_offset)
        #print("results[%d] = %d" % (idx, file_offset))
        self._db.seek(file_offset)
        item = self._db.readline()

        return item


