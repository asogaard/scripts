#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Script for writing contents of ROOT file to text format.

@file:   textify.py
@author: Andreas Søgaard
@date:   22 March 2017
@email:  andreas.sogaard@cern.ch
"""

# Basic import(s)
import sys, os, re

# Get ROOT to stop hogging the command-line options
import ROOT
ROOT.PyConfig.IgnoreCommandLineOptions = True

# Scientific import(s)
from ROOT import TFile, TTree
try:
    from root_numpy import tree2array
    import numpy as np
    from numpy.lib.recfunctions import append_fields, drop_fields
except ImportError:
    print "WARNING: numpy and/or root_numpy were not found. If you're in lxplus, try running:"
    print "         $ source /cvmfs/sft.cern.ch/lcg/views/LCG_88/x86_64-slc6-gcc49-opt/setup.sh"
    sys.exit()
    pass

# Command-line arguments parser
import argparse

parser = argparse.ArgumentParser(description='Convert ROOT file contents to text.')

#parser.add_argument('files', metavar='files', type=str, nargs='+',
parser.add_argument('--files', dest='files', type=str, nargs='+',
                    help='Files to combine')
parser.add_argument('--output', dest='output',
                    required=True,
                    help='Name of the output text file')
parser.add_argument('--treename', dest='treename',
                    required=True,
                    help='Name of the ROOT TTree to convert')
parser.add_argument('--keep', dest='keep', type=str, nargs='*', default=list(),
                    help='Tree branches to keep (defualt: all)')
parser.add_argument('--ignore', dest='ignore', type=str, nargs='*', default=list(),
                    help='Tree branches to ignore (defualt: none)')
parser.add_argument('--weight_name', dest='weight_name', type=str,
                    default='weight',
                    help='Name of weight branch default (defaults: "weight")')
parser.add_argument('--nevents', dest='nevents', type=int,
                    default=None,
                    help='Number of events to process (default: all)')
parser.add_argument('--xsec_file', dest='xsec_file',
                    default=None,
                    help='Path to cross-sections file')


# Main function definition.
# ------------------------------------------------------------------------------
def main ():

    # Parse command-line arguments
    args = parser.parse_args()


    # Load cross-sections
    xsec = None
    if args.xsec_file is not None:
        print "Loading cross-sections"
        xsec = loadXsec(args.xsec_file)
    else:
        print "Not scaling by cross-section"
        pass
    
    # Loop input ROOT files
    tmp_filenames = list()
    for i, input_filename in enumerate(args.files):
        print "Reading file {}/{}: {}".format(i + 1, len(args.files), input_filename)
        
        # Check(s)
        if not input_filename.endswith('.root'):
            warning("'{}' is not a ROOT file.".format(input_filename))
            continue
    
        # Get tree
        input_file = TFile(input_filename, 'READ')
        tree = input_file.Get(args.treename)
        
        # Checks(s)
        if not tree:
            warning("Tree '{}' was not found in file '{}'.".format(args.treename, input_filename))
            continue
        if type(tree) is not TTree:
            warning("Tree '{}' is not a ROOT TTree.".format(args.treename))
            continue

        # Get DSID
        DSID = None
        try:
            m = re.search('(\A|.*\D+)(\d{6})\D*\.root', input_filename)
            DSID = int(m.group(2))
        except:
            pass
        
        if DSID is None:
            warning("Was unable to deduce dataset ID.")
            pass

        # Get structure numpy array from tree
        array = tree2array(tree, branches=args.keep if args.keep else None, stop=args.nevents).view(np.recarray)
        names = list(array.dtype.names)

        # Remove unwanted fields
        for field in args.ignore:
            if field not in names:
                warning("Field to be ignored '{}' was not found among fields: [{}]".format(field, ', '.join(names)))
                continue
            names.remove(field)
            array = array[names]
            pass
        header = ',\t'.join(array.dtype.names)

        # Scale weights by cross-section
        if xsec and DSID:
            if DSID not in xsec:
                warning("DSID {} was not for found among loaded cross-sections. Not scaling weights.".format(DSID))
            else:
                weights = np.ones((array.shape[0],)) * xsec[DSID]
                if args.weight_name not in names:
                    warning("Weight column named '{}' not found among [{}]. Adding column with cross-sections weights.".format(args.weight_name, header))
                    array = append_fields(array, args.weight_name, weights)
                else:
                    view = array[args.weight_name].view(np.float).copy()
                    array = drop_fields  (array, args.weight_name)
                    array = append_fields(array, args.weight_name, view * weights)
                    pass
                pass
            pass

        # Save array to temporary file
        tmp_filenames.append('.' + input_filename.split('/')[-1].replace('.root', '.csv'))
        np.savetxt(tmp_filenames[-1], array, header=header if i == 0 else '', delimiter=', \t', fmt='%-.6e')
        
        pass


    # Concatenate files
    print "Concatenating output"
    with open(args.output, 'wb') as output_file:
        for tmp_filename in tmp_filenames:
            with open(tmp_filename) as tmp_file:
                for line in tmp_file:
                    output_file.write(line)
                pass
            os.remove(tmp_filename)
            pass
        pass

    return


# Utility function(s)
# ------------------------------------------------------------------------------
def warning (string):
    print '\033[91m\033[1mWARNING\033[0m ' + string
    return

def loadXsec (path):
    """ Load cross section weights from file. """
    xsec = dict()
    with open(path, 'r') as f:
        for l in f:
            line = l.strip()
            if line == '' or line.startswith('#'):
                continue
            fields = [f.strip() for f in line.split(',')]
            try:
                if int(fields[2]) == 0:
                    continue
                xsec[int(fields[0])] = float(fields[1]) / float(fields[2]) * float(fields[3])
            except:
                # If data.
                continue
            pass
        pass
    return xsec



# Main function call.
# ------------------------------------------------------------------------------
if __name__ == '__main__':
   main()
   pass
