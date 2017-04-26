# -*- coding: utf-8 -*-
""" Collection of python utility functions.

...
"""

# Basic include(s)
import sys
import math
import itertools

from ROOT  import *
from array import *
from collections import namedtuple

# Scientific include(s); require correct python environment
try:
    import numpy as np
    from numpy.lib.recfunctions import append_fields

    from root_numpy import tree2array
except:
    print "ERROR: Scientific python packages were not set up properly."
    print " $ source snippets/pythonenv.sh"
    print "or see e.g. [http://rootpy.github.io/root_numpy/start.html]."
    raise

# Global variables
colours = [kViolet + 7, kAzure + 7, kTeal, kSpring - 2, kOrange - 3, kPink]


# Utility functions.
def wait ():
    """ Generic wait function.

    Halts the execution of the script until the user presses ``Enter``.
    """
    raw_input('...')
    return


def validateArguments (args):
    """ Function to validate commandline arguments passed to generic script.

    Checks whether the number of arguments is exactly one (i.e. only the script
    name itself). Assumes that at least one path to a ROOT file is needed.

    Args:
        args: Commandline arguments, from sys.argv

    Return:
        None

    Raise:
        IOError: Only one argument was provided.
    """
    if len(args) == 1:
        msg  = "Please specify at least one target ROOT file. Run as:\n"
        msg += " $ python %s path/to/file.root" % args[0]
        raise IOError(msg)
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
            
            if fields[2] == 'Data': 
                continue

            # @TEMP: Assuming sum-of-weights normalisation included in per-event MC weights
            xsec[int(fields[0])] = float(fields[1]) * float(fields[3])

            pass
        pass
    return xsec



def getMaximum (h):
    """ Get *actual* maximum bin in histogram or similar. """
    if type(h) in [TF1, TEfficiency]:
        return -1
    if type(h) in [TGraph, TGraphErrors]:
        return h.GetMaximum()
    N = h.GetXaxis().GetNbins()
    return max([h.GetBinContent(i + 1) for i in range(N)])



def drawText (lines = [], c = None, pos = 'NW', qualifier = 'Internal simulation', man_scale=1.):
    """ Draw text on TPad, including ATLAS line. """
    if not c: c = gPad
    c.cd()

    # Checks.
    if lines is None:
        lines = []
        pass

    t = TLatex()
    t.SetTextSize(t.GetTextSize() * man_scale)

    h = c.GetWh()
    w = c.GetWw()

    offset = 0.05
    ystep = t.GetTextSize() * 1.25
    scale = (w/float(h) if w > h else h/float(w))

    x =       c.GetLeftMargin() + offset * scale
    y = 1.0 - c.GetTopMargin()  - offset - t.GetTextSize() * 1.0 

    t.DrawLatexNDC(x, y, "#scale[1.15]{#font[72]{ATLAS}}#scale[1.05]{  %s}" % qualifier)
    y -= ystep * 1.25 

    for line in lines:
        t.DrawLatexNDC(x, y, line)
        y -= ystep;
        pass

    c.Update()

    return



def drawLegend (histograms, names, types = None, c = None,
                xmin = None,
                xmax = None,
                ymin = None,
                ymax = None,
                header = None,
                categories = None,
                categoryheader=None,
                horisontal = 'R',
                vertical   = 'T',
                width = 0.30,
                man_scale=1.):
    """ Draw legend on TPad. """
    if not c: c = gPad
    c.cd()

    N = len(histograms)
    if N != len(names):
        print "drawLegend: WARNING, number of histograms (%d) and names (%d) don't match." % (N, len(names))
        pass

    if types is None:
        types = ''
        pass

    if type(types) == str:
        types = [types] * N
        pass

    if types and N != len(types):
        print "drawLegend: WARNING, number of histograms (%d) and provided types (%d) don't match." % (N, len(types))
        return None

    fontsize = gStyle.GetLegendTextSize() * man_scale

    offset = 0.04
    height = (min(N, len(names)) + (0 if header == None else 1) + (0 if categoryheader == None else 2) + (len(categories) if categories else 0))* fontsize * 1.25

    # -- Setting x coordinates.
    if not (xmin or xmax):
        if   horisontal.upper() == 'R':
            xmax = 1. - c.GetRightMargin() - offset
        elif horisontal.upper() == 'L':
            xmin = c.GetLeftMargin() + offset
        else:
            print "drawLegend: Horisontal option '%s' not recognised." % horisontal
            return
        pass
    if xmax and (not xmin):
        xmin = xmax - width
        pass
    if xmin and (not xmax):
        xmax = xmin + width
        pass

    # -- Setting y coordinates.
    if not (ymin or ymax):
        if   vertical.upper() == 'T':
            ymax = 1. - c.GetTopMargin() - offset * 1.5
        elif vertical.upper() == 'B':
            ymin = c.GetBottomMargin() + offset
        else:
            print "drawLegend: Vertical option '%s' not recognised." % vertical
            return
        pass
    if ymax and (not ymin):
        ymin = ymax - height
        pass
    if ymin and (not ymax):
        ymax = ymin + height
        pass

    legend = TLegend(xmin, ymin, xmax, ymax)
    legend.SetTextSize(fontsize)

    if header != None:
        legend.AddEntry(None, header, '')
        pass

    for (h,n,t) in zip(histograms, names, types):
        legend.AddEntry(h, n, t)
        pass

    if categoryheader != None:
        legend.AddEntry(None, "", '')
        legend.AddEntry(None, categoryheader, '')
        pass

    if categories:
        for icat, (name, hist, opt) in enumerate(categories):
            hist.SetLineColor  (kGray+3)
            hist.SetMarkerColor(kGray+3)
            hist.SetFillColor  (kGray+2)
            legend.AddEntry(hist, name, opt)
            pass
        pass

    legend.Draw()

    # -- Make global (i.e. don't delete when going out of scope)
    from ROOT import SetOwnership
    SetOwnership( legend, 0 ) 

    c.Update()

    return legend



def getPlotMinMax (histograms, log, padding = None, ymin = None):
    """ Get optimal y-axis plotting range given list of histograms (or sim.). """
    padding = padding if padding else 1.0

    ymax = max(map(getMaximum, histograms))
    ymin = (ymin if ymin is not None else (1e-05 if log else 0.))

    if ymax < ymin: ymax = 2 * ymin
    if log:
        ymax = math.exp(math.log(ymax) + (math.log(ymax) - math.log(ymin)) * padding)
    else:
        ymax = ymax + (ymax - ymin) * padding
        pass

    return (ymin, ymax)



LegendOptions = namedtuple('LegendOptions', ['histograms', 'names', 'types', 'c', 'xmin', 'xmax', 'ymin', 'ymax', 'header', 'categories', 'categoryheader', 'horisontal', 'vertical','width'])
LegendOptions.__new__.__defaults__ = ('LP', None, None, None, None, None, None, None, None, 'R', 'T', 0.30)

TextOptions = namedtuple('TextOptions', ['lines', 'c', 'pos', 'qualifier'])
TextOptions.__new__.__defaults__ = ([], None, 'NW','Internal simulation',)

def makePlot (pathHistnamePairs,
              legendOpts = None, #LegendOptions([], []),
              textOpts   = None, #TextOptions(),
              canvas = None,
              ymin = None,
              ymax = None,
              xlim = None,
              logy = False,
              padding = None,
              xtitle = None,
              ytitle = None,
              ztitle = None,
              colours = None, 
              fillcolours = None, 
              alpha = None,
              linewidths = None,
              linestyles = None,
              markers = None,
              xlines = None,
              ylines = None,
              drawOpts = None,
              normalise = False,
              profileRMS = False):
    """ ... """

    # Variable declarations
    if not colours:
        colours = [kViolet + 7, kAzure + 7, kTeal, kSpring - 2, kOrange - 3, kPink]
        pass

    if not markers:
        markers = [20] * len(pathHistnamePairs)
        pass

    if not linewidths:
        linewidths = [2] * len(pathHistnamePairs)
        pass

    if not linestyles:
        linestyles = [1] * len(pathHistnamePairs)
        pass

    if fillcolours:
        if not type(fillcolours) == list:
            fillcolours = [fillcolours] * len(pathHistnamePairs)
            pass
        pass

    if not type(alpha) == list:
        alpha = [alpha] * len(pathHistnamePairs)
        pass

    if not drawOpts:
        drawOpts = ''
        pass

    if type(drawOpts) is str:
        drawOpts = [drawOpts] * len(pathHistnamePairs) # len(histograms)
        pass

    # Loop all pairs of paths and histograms.
    # -- Assuming list(tuple(*,*)) structure.
    histograms = list()
    if len(pathHistnamePairs) > 0 and type(pathHistnamePairs[0]) in [tuple, list]:
        for entry in pathHistnamePairs:
            if type(entry) is list:
                print "WARNING: You're passing a list of lists to makePlot."
                continue

            # Check if entry is histogram.
            if type(entry) in [TH1F, TH2F, TProfile, TEfficiency]:
                h = entry
            else:
                # Otherwise, read from file.
                path, histname = entry
                
                # -- Open file
                f = TFile.Open(path, 'READ')
                
                # -- Get histogram.
                print "histname: '%s'" % histname
                h = f.Get(histname)
                """ If TTree? """
                
                # -- Keep in memory after file is closed.
                h.SetDirectory(0)
                if   isinstance(h, TH2):
                    TH2.AddDirectory(False)
                elif isinstance(h, TH1):
                    TH1.AddDirectory(False)
                    pass
                
                pass

            # -- Append to list of histograms to be plotted.
            histograms.append(h)            
            pass    
    else:
        histograms = pathHistnamePairs
        pass


    # Common, regardless of how the histogram was obtained.
    for h in histograms:
        # -- Normalise.
        if normalise and h.Integral() > 0:
            h.Scale(1./h.Integral(0, h.GetXaxis().GetNbins() + 1))
            pass
    
        # -- Make TProfile show RMS
        if isinstance(h, TProfile) and profileRMS:
            h.SetErrorOption('s')
            pass

        pass


    # Style.
    (ymin, _ymax) = getPlotMinMax(histograms, logy, ymin = ymin, padding = padding) if histograms else (0,0)
    ymax = ymax if ymax else _ymax
    for i,h in enumerate(histograms):

        if i < len(colours):
            if alpha[i]:
                h.SetLineColorAlpha(colours[i], alpha[i])
            else:
                h.SetLineColor     (colours[i])
                pass

            h.SetMarkerColor   (colours[i])

            if fillcolours:
                if alpha[i]:
                    h.SetFillColorAlpha(fillcolours[i], alpha[i])
                else:
                    h.SetFillColor(fillcolours[i])
                    pass
                pass

            h.SetMarkerStyle(markers[i])
            h.SetLineWidth  (linewidths[i])
            h.SetLineStyle  (linestyles[i])

            #h.GetXaxis().SetNdivisions(505)
            pass

        # -- Axes.
        if type(h) in [TEfficiency]:
            h.SetTitle(';%s;%s;%s' % ( (xtitle if xtitle else ''),
                                       (ytitle if ytitle else ''),
                                       (ztitle if ztitle else ''),
                                       ))
            continue
        if xtitle: h.GetXaxis().SetTitle(xtitle)
        if ytitle: h.GetYaxis().SetTitle(ytitle)

        if normalise:
            h.GetYaxis().SetTitle(h.GetYaxis().GetTitle() + ' (normalised)')
            pass

        h.GetYaxis().SetRangeUser(ymin, ymax)

        pass

    # Canvas.

    if canvas:
        c = canvas
    else:
        is2D = (drawOpts and True in ['Z' in opt for opt in drawOpts])
        c = TCanvas('c', "", int(600 * (7./6. if is2D else 1.)), 600)
        if is2D:
            c.SetRightMargin(0.15)
            pass
        pass
    c.SetLogy(logy)
    c.cd()

    # Draw axes.
    if histograms:
        histograms[0].Draw('AXIS')
        pass
    if xlim:
        c.Update()
        histograms[0].GetXaxis().SetRangeUser(xlim[0], xlim[1])
        pass

    # Lines
    if xlines or ylines:
        c.Update()

        xmin = gPad.GetUxmin() # c.GetFrame().GetX1()
        xmax = gPad.GetUxmax() # c.GetFrame().GetX2()
        ymin = ymin if ymin else gPad.GetUymin() # c.GetFrame().GetY1()
        ymax = ymax if ymax else gPad.GetUymax() # c.GetFrame().GetY2()

        line = TLine()
        line.SetLineColor(kGray + 2)
        line.SetLineStyle(2)

        # -- x-axis
        if xlines:
            for xline in xlines:
                line.DrawLine(xline, ymin, xline, ymax)
                pass
            pass

        # -- y-axis
        if ylines:
            for yline in ylines:
                line.DrawLine(xmin, yline, xmax, yline)
                pass
            pass

        pass

    # Draw histograms.
    for i, (drawOpt, h) in enumerate(zip(drawOpts,histograms)):
        #h.Draw(drawOpt + ' SAME')
        h.Draw((drawOpt.replace('A', '') if i > 0 else drawOpt) + ' SAME') # TEMP
        pass

    # Drawing axes for TGraph fucks up
    if histograms:
        if type(histograms[0]) in [TH1, TH2, TEfficiency]:
            histograms[0].Draw('AXIS SAME')
            pass
        pass

    # Text.
    if textOpts:
        drawText( *[v for _,v in textOpts._asdict().items()])
        pass

    # Legend.
    if legendOpts:
        legendOpts = legendOpts._replace(histograms = histograms)
        drawLegend( *[v for _,v in legendOpts._asdict().items()] )
        pass

    # Update.
    c.Update()
    return c



def loadDataFast (paths, treename, branches, prefix = '', xsec = None, ignore = None, keepOnly = None, onlyData = False, onlyMC = False, DSIDvar = 'DSID', isMCvar = 'isMC', Nevents = 29, quiet = False):
    
    if not quiet: print ""
    print "loadDataFast: Reading data from %d files." % len(paths)

    if len(paths) == 0:
        if not quiet: print "loadDataFast: Exiting."
        return dict()

    ievent = 0

    # Initialise data array.
    data = None

    # Initialise DSID variable.
    DSID = None
    isMC = None

    # Loop paths.
    for ipath, path in enumerate(paths):

        # Print progress.
        if not quiet: print "\rloadDataFast:   [%-*s]" % (len(paths), '-' * (ipath + 1)),
        if not quiet: sys.stdout.flush()

        # Get file.
        f = TFile(path, 'READ')

        # Get DSID.
        if not quiet: print "output tree name:", '/'.join(treename.split('/')[:2]) + '/outputTree'
        outputTree = f.Get('/'.join(treename.split('/')[:2]) + '/outputTree')
        for event in outputTree:
            DSID = eval('event.%s' % DSIDvar)
            isMC = eval('event.%s' % isMCvar)
            break

        if not DSID:
            if not quiet: print "\rloadDataFast:   Could not retrieve DSID. File output is probably empty. Skipping %s" % path
            continue

        # Check whether to explicitly keep or ignore.
        if onlyData and isMC:
            continue
        if onlyMC and not isMC:
            continue
        if keepOnly and DSID and not keepOnly(DSID):
            if not quiet: print "\rNot keeping DSID %d." % DSID
            continue
        elif ignore and DSID and ignore(DSID):
            if not quiet: print "\rloadDataFast:   Ignoring DSID %d." % DSID
            continue

        # Get tree.
        t = f.Get(treename)
        
        if not t:
            if not quiet: print "\rloadDataFast:   Tree '%s' was not found in file '%s'. Skipping." % (treename, path)
            continue

        # Load new data array.
        arr = tree2array(t,
                         branches = [prefix + br for br in branches] + ['weight'], 
                         )

        # Add cross sections weights.
        if xsec and DSID and isMC:

            # Ignore of we didn't provide cross section information.
            if DSID not in xsec:
                print "\rloadDataFast:   Skipping DSID %d (no sample info)." % DSID
                continue

            # Scale weight by cross section.
            arr['weight'] *= xsec[DSID]

            pass

        # Add isMC array.
        arr = append_fields(arr, 'isMC', np.ones(arr[prefix + branches[0]].shape) * isMC)

        # Add DSID array.
        try:
            arr = append_fields(arr, 'DSID', np.ones(arr['isMC'].shape) * DSID)
        except:
            print "WARNING: Couldn't concatenate DSID %s:" % DSID
            print " ", arr
            print " ",arr['isMC']
            print " ",arr['isMC'].shape
            print " ",DSID
            pass

        # Append to existing data arra
        if data is None:
            data = arr
        else:
            try:
                data = np.concatenate((data, arr))
            except:
                # If array contains only a single row...
                # @TODO: Fix
                pass
            pass

        pass

    # Dict-ify.
    values = dict()
    if data is not None:
        for branch in data.dtype.names:
            values[branch] = data[branch]
            pass
        pass

    # Change branch names to remove prefix.
    for key in values:
        values[key.replace(prefix, '')] = values.pop(key)
        pass

    return values




def loadDataFast_ (tree, variables, prefix = ''):

    # Load new data array.
    arr = tree2array(tree,
                     branches = [prefix + var for var in variables], # if var not in vec_names],
                     include_weight = True,
                     )

    values = dict()
    for var in arr.dtype.names:
        values[var] = arr[var].tolist()
        pass

    return values

def loadData_ (tree, variables, prefix = ''):

    # -- Loading characters
    loadingCharacters  = ['\\', '|', '/', '-']
    iLoadingCharacter  = 0
    nLoadingCharacters = len(loadingCharacters)

    # -- Get number of entries in tree
    Nevents = tree.GetEntries()

    # -- Set up objects for reading tree.
    values = dict()
    for var in variables + ['weight'] :
        values[var] = list()        
        #tree.SetBranchAddress( var if var == 'weight' else prefix + var, branch[var])
        tree.SetBranchStatus( var if var == 'weight' else prefix + var, 1)

        pass

    # -- Read tree
    ievent = 0
    while tree.GetEntry(ievent):
        # -- Break early, if needed.
        if Nevents is not None and ievent == Nevents:
            break
        
        for var in variables + ['weight']:
            #values[var].append( branch[var][0] )
            values[var].append( eval('tree.%s' % var) )
            pass
        ievent += 1
        
        if ievent % 10000 == 0:
            print "\rloadData: [%s]" % loadingCharacters[iLoadingCharacter],
            sys.stdout.flush()
            iLoadingCharacter = (iLoadingCharacter + 1) % nLoadingCharacters
            pass
        
        pass

    return values

def loadData (paths, treename, branches, prefix = '', xsec = None, ignore = None, keepOnly = None, onlyData = False, onlyMC = False, DSIDvar = 'DSID', isMCvar = 'isMC', Nevents = 29, fast = True):
    
    print ""
    print "loadData: Reading data from %d files %s 'fast' flag." % (len(paths), "using" if fast else "without")

    if len(paths) == 0:
        print "loadData: Exiting."
        return dict()

    ievent = 0

    # Initialise data array.
    data = None

    # Initialise DSID variable.
    DSID = None
    isMC = None

    # Loop paths.
    values = dict()
    for ipath, path in enumerate(paths):

        # Print progress.
        print "\rloadData:   [%-*s]" % (len(paths), '-' * (ipath + 1)),
        sys.stdout.flush()

        # Get file.
        f = TFile(path, 'READ')

        # Get DSID.
        outputTree = f.Get('/'.join(treename.split('/')[0:2]) + '/outputTree')
        for event in outputTree:
            DSID = eval('event.%s' % DSIDvar)
            isMC = eval('event.%s' % isMCvar)
            break

        if not DSID:
            print "\rloadData:   Could not retrieve DSID file output is probably empty. Skipping."
            continue

        # Check whether to explicitly keep or ignore.
        if onlyData and isMC:
            continue
        if onlyMC and not isMC:
            continue
        if keepOnly and DSID and not keepOnly(DSID):
            print "\rNot keeping DSID %d." % DSID
            continue
        elif ignore and DSID and ignore(DSID):
            print "\rloadData:   Ignoring DSID %d." % DSID
            continue

        # Get tree.
        t = f.Get(treename)

        if not t:
            print "\rloadData:   Tree '%s' was not found in file '%s'. Skipping." % (treename, path)
            continue

        # Load data using under-the-hood function
        if fast:
            current_values = loadDataFast_(t, branches, prefix)
        else:
            current_values = loadData_(t, branches, prefix)
            pass


        # Add cross sections weights.
        if xsec and DSID: # and isMC:

            # Ignore of we didn't provide cross section information.
            if DSID not in xsec:
                print "\rloadData:   Skipping DSID %d (no sample info)." % DSID
                continue

            # Scale weight by cross section.
            #current_values['weight'] *= xsec[DSID]
            current_values['weight'] = (np.array(current_values['weight']) *  xsec[DSID]).tolist()

            pass

        # Add isMC array.
        current_values['isMC'] = (np.ones_like(current_values[prefix + branches[0]]) * isMC).tolist()

        # Add DSID array.
        current_values['DSID'] = (np.ones_like(current_values[prefix + branches[0]]) * DSID).tolist()

        # Append to existing data arra
        if values:
            #values = { key : np.concatenate((values[key], current_values[key])) for key in values }
            for key in values:
                values[key] += current_values[key]
                pass
        else:
            values = current_values
        pass

    # Change branch names to remove prefix.
    for key in values:
        values[key.replace(prefix, '')] = values.pop(key)
        pass

    return values


'''
def loadData (paths, treename, variables, prefix = '', xsec = None, ignore = None, DSIDvar = 'DSID', Nevents = None):
    """ Read in data arrays from TTree. """
    values = dict()

    print ""
    print "loadData: Reading data from %d files." % len(paths)

    loadingCharacters  = ['\\', '|', '/', '-']
    iLoadingCharacter  = 0
    nLoadingCharacters = len(loadingCharacters)

    if len(paths) == 0: return 

    ievent = 0
    for ipath, path in enumerate(paths):

        DSID = None

        # Getting tree.
        f = TFile.Open(path, 'READ')

        if xsec:
            for event in f.Get('outputTree'):
                DSID = eval('event.%s' % DSIDvar)
                break

            # -- Only consider samples for which we have chosen to provide cross section information.
            if DSID not in xsec:
                continue

            if 'xsec' not in values:
                values['xsec'] = array('d', [])
                pass

            pass 

        # -- Ignore signal samples.
        if ignore and ignore(DSID):
            print "Ignoring DSID %d." % DSID
            continue

        t = f.Get(treename)

        if not t:
            "loadData: Could not get tree '%s' from file '%s'." % (treename, path)
            continue
        N = t.GetEntries()

        # -- Set up objects for reading tree.
        branch = dict()
        for var in variables + ['weight'] :
            branch[var] = array('d', [0])
            
            if var not in values:
                values[var] = array('d', [])
                pass
            
            t.SetBranchAddress( var if var == 'weight' else prefix + var, branch[var] )
            pass

        # -- Read tree
        i = 0
        while t.GetEntry(i):
            # -- Break early, if needed.
            if Nevents is not None and ievent == Nevents:
                break
            ievent += 1

            for var in variables + ['weight']:
                values[var].append( branch[var][0] )
                pass
            i += 1

            if i % 10000 == 0:
                print "\rloadData:  [%s]" % loadingCharacters[iLoadingCharacter],
                sys.stdout.flush()
                iLoadingCharacter = (iLoadingCharacter + 1) % nLoadingCharacters
                pass

            pass

        if xsec:
            xsec_weight = xsec[DSID]
            values['xsec'] += array('d', [xsec_weight for _ in xrange(i)]) # N
            pass

        if Nevents is not None and ievent == Nevents:
            break
        
        pass

    # Scale 'weight' to also include cross section
    values['weight'] = array('d', [ w * x for (w,x) in zip(values['weight'], values['xsec'])])
    values.pop('xsec')

    print ""
    print "loadData: Done."
    print ""

    return values
'''

def displayName (var, latex = False):
    output = var

    # tau21
    if   var == "tau21":                output = "#tau_{21}"
    elif var == "tau21_ut":             output = "#tau_{21,un-trimmed}"
    elif var == "tau21_ungroomed":      output = "#tau_{21,un-trimmed}"
    elif var == "tau21_mod_rhoPrime":   output = "#tilde{#tau}_{21} "
    elif var == "tau21_mod_rhoDDT":     output = "#tau_{21}^{DDT}"
    elif var == "tau21DDT":             output = "#tau_{21}^{DDT}"
    elif var == "tau21_SDDT":           output = "#tilde{#tau}_{21}^{(S)DDT}"
    # D2
    elif var == "D2":                   output = "D_{2}"
    elif var == "D2mod":                output = "#tilde{D}_{2}"
    elif var == "D2_SDDT":              output = "#tilde{D}_{2}^{(S)DDT}"
    # Kinematic variables
    elif var.lower() == "pt":           output = "p_{T} "
    elif var.lower() == "m":            output = "M"
    elif var.lower() == "eta":          output = "#eta"
    elif var.lower() == "theta":        output = "#theta"
    elif var.lower() == "phi":          output = "#phi"
    elif var.lower() == "qoverp":       output = "q/p"
    elif var.lower() == "mu":           output = "#mu"
    elif var.lower() == "d0":           output = "d_{0}"
    elif var.lower() == "z0":           output = "z_{0}"
    # rho
    elif var == "rho":                  output = "#rho"
    elif var == "rho_ut":               output = "#rho_{untrimmed}"
    elif var == "rhoPrime":             output = "#rho'"
    elif var == "rhoPrime_ut":          output = "#rho'_{untrimmed}"
    elif var == "rhoDDT":               output = "#rho^{DDT}"
    elif var == "rhoDDT_ut":            output = "#rho^{DDT}_{untrimmed}"
    # log(...)
    elif var.lower().startswith('log'): output = "#log(%s)" % displayName(var[3:])


    return r'$%s$' % output.replace('#', '\\') if latex else output

def displayUnit (var):
    if   var.lower() == 'pt':     return 'GeV'
    elif var.lower() == 'm':      return 'GeV'
    elif var.lower() == 'qoverp': return 'GeV^{-1}'
    elif var.lower() == 'd0':     return 'mm'
    elif var.lower() == 'z0':     return 'mm'
    elif var.lower() == "logm":   return "log(%s)" % displayUnit("m")
    elif var.lower() == "logpt":  return "log(%s)" % displayUnit("pt")
    return ''

def displayNameUnit (var, latex = False):
    name = displayName(var, latex)
    unit = displayUnit(var)
    return name + (r" [%s]" % unit if unit else unit)


def dict_product(dicts):
    """ Returns cartesian product of dict entries, as a dict with one-entry value arrays, using itertools.product """
    return (dict(itertools.izip(dicts, x)) for x in itertools.product(*dicts.itervalues()))
