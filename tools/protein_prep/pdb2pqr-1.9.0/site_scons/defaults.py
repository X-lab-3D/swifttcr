import distutils.sysconfig
import os
import sys
import platform
from datetime import datetime

defaultPrefix = os.path.expanduser('~/pdb2pqr/')
defaultURL = 'http://' + platform.node() + '/pdb2pqr/'
defaultMaxAtoms = 10000
codePath = os.getcwd()

pythonBin = sys.executable

buildTime = datetime.today()
productVersion = '1.9.0'# + buildTime.strftime('%Y%m%d%H%M')