#!/usr/bin/env bash
#
# Script for setting environment variables on lxplus, which enables the use of several usefule python packages.
#
# See also [http://rootpy.github.io/root_numpy/start.html]

# Setup LCG SWAN environment.
source /cvmfs/sft.cern.ch/lcg/releases/LCG_85swan2/gcc/4.9.3/x86_64-slc6/setup.sh;

BASEPATH="/cvmfs/sft.cern.ch/lcg/releases/LCG_85swan2"
ARCH_COMP="x86_64-slc6-gcc49-opt"

# -- Add paths for python.
BASEPATH_PYTHON="${BASEPATH}/Python/2.7.10/${ARCH_COMP}"
export PATH="${BASEPATH_PYTHON}/bin:${PATH}";
export LD_LIBRARY_PATH="${BASEPATH_PYTHON}/lib:${LD_LIBRARY_PATH}";
export PYTHONPATH="${BASEPATH_PYTHON}/lib/python2.7/site-packages:${PYTHONPATH}";
export PKG_CONFIG_PATH="${BASEPATH_PYTHON}/lib/pkgconfig:${PKG_CONFIG_PATH}";

export PYTHON_HOME="${BASEPATH_PYTHON}";
cd "${BASEPATH_PYTHON}"
export PYTHONHOME="${PYTHON_HOME}"
cd - 1>/dev/null # from ${BASEPATHPYTHON} (/cvmfs/sft.cern.ch/lcg/releases/LCG_85swan2/Python/2.7.10/x86_64-slc6-gcc49-opt)

# -- Add paths for numpy.
BASEPATH_NUMPY="${BASEPATH}/numpy/1.11.0/${ARCH_COMP}"
export PATH="${BASEPATH_NUMPY}/bin:${PATH}";
export LD_LIBRARY_PATH="${BASEPATH_NUMPY}/lib:${LD_LIBRARY_PATH}";
export PYTHONPATH="${BASEPATH_NUMPY}/lib/python2.7/site-packages:${PYTHONPATH}";

# -- Add paths for root_numpy.
BASEPATH_ROOTNUMPY="${BASEPATH}/root_numpy/4.5.1/${ARCH_COMP}"
export LD_LIBRARY_PATH="${BASEPATH_ROOTNUMPY}/lib:${LD_LIBRARY_PATH}";
export PYTHONPATH="${BASEPATH_ROOTNUMPY}/lib/python2.7/site-packages:${PYTHONPATH}";

# -- Add paths for scipy.
BASEPATH_SCIPY="${BASEPATH}/scipy/0.15.1/${ARCH_COMP}"
export LD_LIBRARY_PATH="${BASEPATH_SCIPY}/lib:${LD_LIBRARY_PATH}";
export PYTHONPATH="${BASEPATH_SCIPY}/lib/python2.7/site-packages:${PYTHONPATH}";

# -- Add paths for pyparsing (prerequisite for matplotlib)
BASEPATH_PYPARSING="${BASEPATH}/pyparsing/2.0.3/${ARCH_COMP}"
export LD_LIBRARY_PATH="${BASEPATH_PYPARSING}/lib:${LD_LIBRARY_PATH}";
export PYTHONPATH="${BASEPATH_PYPARSING}/lib/python2.7/site-packages:${PYTHONPATH}";

# -- Add paths for setuptools (prerequisite for matplotlib)
BASEPATH_SETUPTOOLS="${BASEPATH}/setuptools/20.1.1/${ARCH_COMP}"
export PATH="${BASEPATH_SETUPTOOLS}/bin:$PATH";
export LD_LIBRARY_PATH="${BASEPATH_SETUPTOOLS}/lib:${LD_LIBRARY_PATH}";
export PYTHONPATH="${BASEPATH_SETUPTOOLS}/lib/python2.7/site-packages:${PYTHONPATH}";

# -- Add pathds for dateutil (prerequisite for matplotlib)
BASEPATH_DATEUTIL="${BASEPATH}/python_dateutil/2.4.0/${ARCH_COMP}"
export LD_LIBRARY_PATH="${BASEPATH_DATEUTIL}/lib:${LD_LIBRARY_PATH}";
export PYTHONPATH="${BASEPATH_DATEUTIL}/lib/python2.7/site-packages:${PYTHONPATH}";

# -- Add paths for matplotlib.
BASEPATH_MATPLOTLIB="${BASEPATH}/matplotlib/1.5.1/${ARCH_COMP}"
export PATH="${BASEPATH_MATPLOTLIB}/bin:${PATH}";
export LD_LIBRARY_PATH="${BASEPATH_MATPLOTLIB}/lib:${LD_LIBRARY_PATH}";
export PYTHONPATH="${BASEPATH_MATPLOTLIB}/lib/python2.7/site-packages:${PYTHONPATH}";

# -- Add paths for sklearn.
BASEPATH_SKLEARN="${BASEPATH}/scikitlearn/0.17.1/${ARCH_COMP}"
export LD_LIBRARY_PATH="${BASEPATH_SKLEARN}/lib:${LD_LIBRARY_PATH}";
export PYTHONPATH="${BASEPATH_SKLEARN}/lib/python2.7/site-packages:${PYTHONPATH}";

# Add ~/.local/bin to PATH
export PATH=$HOME/.local/bin:$PATH

# Fix "ValueError: unknown locale: UTF-8" when importing matplotlib
export LC_ALL="en_US.UTF-8"
export LANG="en_US.UTF-8"