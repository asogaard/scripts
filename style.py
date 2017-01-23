from ROOT  import *
from array import *

# Global style variables.
font = 42

fontSizeS = 0.035 # 0.035
fontSizeM = 0.038  # 0.040
fontSizeL = 0.040

kMyBlue  = 1001;
myBlue   = TColor(kMyBlue,   0./255.,  30./255.,  59./255.)
kMyRed   = 1002;
myRed    = TColor(kMyRed,  205./255.,   0./255.,  55./255.)
kMyGreen = kGreen + 2
kMyLightGreen = kGreen - 10


# Custom style definition.
AStyle = TStyle('AStyle', "AStyle")

# -- Canvas colours
AStyle.SetFrameBorderMode(0)
AStyle.SetFrameFillColor(0)
AStyle.SetCanvasBorderMode(0)
AStyle.SetCanvasColor(0)
AStyle.SetPadBorderMode(0)
AStyle.SetPadColor(0)
AStyle.SetStatColor(0)

# -- Canvas size and margins

AStyle.SetPadRightMargin (0.05)
AStyle.SetPadBottomMargin(0.15)
AStyle.SetPadLeftMargin  (0.15)
AStyle.SetPadTopMargin (0.05)
AStyle.SetTitleOffset(1.2, 'x')
AStyle.SetTitleOffset(1.6, 'y') # 1.5
AStyle.SetTitleOffset(1.6, 'z')

# -- Fonts
AStyle.SetTextFont(font)

AStyle.SetTextSize(fontSizeS)

for coord in ['x', 'y', 'z']:
    AStyle.SetLabelFont(font,      coord)
    AStyle.SetTitleFont(font,      coord)
    AStyle.SetLabelSize(fontSizeM, coord)
    AStyle.SetTitleSize(fontSizeM, coord)
    pass

AStyle.SetLegendFont(font)
AStyle.SetLegendTextSize(fontSizeS)

# -- Histograms
AStyle.SetMarkerStyle(20)
AStyle.SetMarkerSize(1.2)
AStyle.SetHistLineWidth(2)
AStyle.SetLineStyleString(2,"[12 12]") # postscript dashes

AStyle.SetErrorX(0.001) # No x-axis errors
AStyle.SetEndErrorSize(0.) # No errorbar caps

# -- Canvas
AStyle.SetOptTitle(0)
AStyle.SetOptStat(0)
AStyle.SetOptFit(0)

AStyle.SetPadTickX(1)
AStyle.SetPadTickY(1)
AStyle.SetLegendBorderSize(0)

# Colour palette.
def set_palette(name='palette', ncontours=999):
    """Set a color palette from a given RGB list
    stops, red, green and blue should all be lists of the same length
    see set_decent_colors for an example"""

    stops = [0.00, 1.00]
    red   = [0.98,  0./255.]
    green = [0.98, 30./255.]
    blue  = [0.98, 59./255.]

    s = array('d', stops)
    r = array('d', red)
    g = array('d', green)
    b = array('d', blue)

    npoints = len(s)
    TColor.CreateGradientColorTable(npoints, s, r, g, b, ncontours)
    gStyle.SetNumberContours(ncontours)
    return

set_palette()

# Set (and force) style.
# --------------------------------------------------------------------

gROOT.SetStyle("AStyle")
gROOT.ForceStyle()
