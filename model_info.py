#!/usr/bin/env python

import os
import sys
import argparse
import signal

from copper.model import Model

def _sigint_handler( signal, frame ):
    print( "Ctrl-C pressed, exiting" )
    sys.exit( 0 )


def _main():
    parser = argparse.ArgumentParser()
    parser.add_argument( "model_path", help="", default = None )
    parser.add_argument( "--verbose", "-v", help="", dest = "verbose", action = "store_true" )

    args = parser.parse_args()

    signal.signal( signal.SIGINT, _sigint_handler )

    cnn, optimizer, saved_args = Model.load( args.model_path )

    if args.verbose:
        print( "model =\n", cnn._model )
        print( "class_table = \n", cnn._class_table )


if __name__ == "__main__":
    _main()  


