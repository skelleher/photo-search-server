#!/usr/bin/env python

import argparse
import os
import sys

import torch
import torch.backends.cudnn as cudnn

from copper.model import Model

_args  = None
_model = None

# Load model
# Print last N layers
# Strip N layers
# Print model
# Save model
 
def _main():
    # force print() to be unbuffered
    sys.stdout = os.fdopen(sys.stdout.fileno(), 'w')
   
    parser = argparse.ArgumentParser()
    parser.add_argument("saved_model", help="Serialized PyTorch model to modify")
    parser.add_argument("--output-class-name", help="Name of new PyTorch model class to save as e.g. ResnetCBIR", dest = "output_class_name")
    parser.add_argument("--strip-layers", help="Number of layers to strip", nargs="?", type = int, default = 0) 
    parser.add_argument("--strip-class-table", help="Strip the class table, e.g. convert classifer to feature extractor (in combination with strip_layers)", dest = "strip_class_table", action = "store_true" )
    parser.add_argument("--verbose", "-v", dest = "verbose", action = "store_true" )

    global _args
    _args = parser.parse_args()
    num_layers_to_show  = _args.strip_layers + 1
    num_layers_to_strip = _args.strip_layers

    model, _, _ = Model.load( _args.saved_model )

    if _args.strip_class_table:
        print("Stripped class table")
        model._class_table = None
        model._num_classes = None

    if _args.verbose:
        _print_last_n_layers( model, num_layers_to_show)
        
    if _args.strip_layers > 0:
        print("Removed %d layers" % num_layers_to_strip)
        removed = list(model._model.children())[ : -num_layers_to_strip ]
        # PROBLEM: this doesn't keep ANY of the layer names from the model,
        # so if we save it we can't load it again (mis-matched state_dict)
        # TODO: rewrite the state_dict keys
        old_state_dict = model._model.state_dict()
        model._model = torch.nn.Sequential(*removed)
        new_state_dict = model._model.state_dict()

        # Rename the state_dict keys before saving, so they match the model.
        # Otherwise, we won't be able to reload them in the future.
        # Relies on fact that state_dict is an OrderedDict
        # We can't modify an OrderedDict while iterating, so we have to make a full copy in RAM first.
    #    for key, key2 in list(zip( old_state_dict.keys(), new_state_dict.keys() )):
    #        print("%s -> %s" % (key2, key))
    #        del new_state_dict[ key2 ]
    #        new_state_dict[ key ] = old_state_dict[ key ]
    #    model._model.load_state_dict( new_state_dict )

    if _args.verbose:
        _print_last_n_layers( model, num_layers_to_show )

    new_filename = _args.saved_model + "." + _args.output_class_name
    print("Saving to ", new_filename)
    model.save( new_filename, classname=_args.output_class_name )
#    model.save_as( new_filename, classname = _args.output_class_name )


def _print_last_n_layers( model, n ):
    print("\nLast %d layers:" % (n))
    layers = list(model._model.children())[ -n: ]

    for layer in layers:
        print(" * ", layer)


def rename_key(iterable, oldkey, newKey):
    if type(iterable) is dict:
        for key in iterable.keys():
            if key == oldkey:
                iterable[newKey] = iterable.pop(key)

    return iterable


if __name__ == "__main__":
    _main()

