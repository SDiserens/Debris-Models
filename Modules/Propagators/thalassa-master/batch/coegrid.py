#!/usr/bin/env python3
"""
GENERATE GRID OF INITIAL CONDITIONS
Generate a grid file of initial conditions starting from user-assigned intervals
and spacings in the initial epoch (expressed as MJD TT) and in the orbital
elements in the EMEJ2000 frame.

Author:
  Davide Amato
  The University of Arizona
  davideamato@email.arizona.edu

Revisions:
  180518: Script creation.
  180523: Parse arguments; create grid table from input file in JSON format.
  180526: Create directory structure, add fields to JSON input, create input.txt
          and object.txt in each subdirectory.
  180529: chunkSize is now global.
  180610: add settings for variable solar flux and SRP eclipses to JSON input.

"""

import sys
import os
import shutil
import json
import numpy as np
import datetime
import argparse

# Globals
chunkSize = 10000

def genGrid(nTot,gDict):
  """
  Creates a grid.dat file containing the grid of orbital elements in tabular
  format, starting from the grid specifications contained in gDict.

  Author:
    Davide Amato
    The University of Arizona
    davideamato@email.arizona.edu
  
  Revisions:
    180523: function created.
    180610: add settings for variable solar flux and SRP eclipses.
  """
  
  # Generate nTot-by-8 array, and dump to disk.
  grid = np.empty([nTot,8])
  
  # Initialize Simulation ID (SID) to keep track of the number of propagations.
  SID = 1

  # The grid array is filled in the order: MA, AOP, RAAN, INC, ECC, SMA, MJD.
 
  # Get deltas
  for key in gDict:
    if gDict[key]['points'] > 1:
      gDict[key]['delta'] = (gDict[key]['end'] - gDict[key]['start']) / (gDict[key]['points'] - 1)
    else:
      gDict[key]['delta'] = 0.
    
  # Here's the Big Nested Loop.
  for i0 in range(0, gDict['MJD']['points']):
    MJD = gDict['MJD']['start'] + i0 * gDict['MJD']['delta']

    for i1 in range(0, gDict['SMA']['points']):
      SMA = gDict['SMA']['start'] + i1 * gDict['SMA']['delta']

      for i2 in range(0, gDict['ECC']['points']):
        ECC = gDict['ECC']['start'] + i2 * gDict['ECC']['delta']

        for i3 in range(0, gDict['INC']['points']):
          INC = gDict['INC']['start'] + i3 * gDict['INC']['delta']

          for i4 in range(0, gDict['RAAN']['points']):
            RAAN = gDict['RAAN']['start'] + i4 * gDict['RAAN']['delta']

            for i5 in range(0, gDict['AOP']['points']):
              AOP = gDict['AOP']['start'] + i5 * gDict['AOP']['delta']

              for i6 in range(0, gDict['MA']['points']):
                MA = gDict['MA']['start'] + i6 * gDict['MA']['delta']
                
                grid[SID - 1,:] = [SID,MJD,SMA,ECC,INC,RAAN,AOP,MA]
                SID  = SID + 1

  return grid

def createInput(dirPath,gSettings):
  """
  Creates an input.txt file in a format readable by the THALASSA Fortran
  executable, using the settings in the grid definition dictionary.
  It reads the input file template from ../in/input.txt, so keep this up to date
  in case that changes!

  Author:
    Davide Amato
    The University of Arizona
    davideamato@email.arizona.edu
  
  Revisions:
    180526: function created.
    190116: Change line numbers in to accommodate the flag "imcoll".
  """
  
  with open(os.path.join('../in','input.txt')) as f:
    inpFile = f.readlines()
  

  # Model settings
  model = gSettings["Model"]
  inpFile[13] = "insgrav:   {:1d}\n".format(int(model["NS gravity"]["Flag"]))
  inpFile[14] = "isun:      {:1d}\n".format(int(model["Lunisolar"]["Sun"]))
  inpFile[15] = "imoon:     {:1d}\n".format(int(model["Lunisolar"]["Moon"]))

  if model["Drag"]["Flag"] == False:
    inpFile[16] = "idrag:     0\n"
  else:
    dm = model["Drag"]["Model"].lower()
    if dm == "wertz":
      idrag = 1
    elif dm == "us76":
      idrag = 2
    elif dm == "j77":
      idrag = 3
    elif dm == "msis00":
      idrag = 4
    else:
      raise ValueError('Value "' + model["Drag"]["Model"] + '" invalid.')
    inpFile[16] = "idrag:     {:1d}\n".format(idrag)
  if model["Drag"]["Solar flux"].lower() == "constant":
    inpFile[17] = "iF107:     0\n"
  elif model["Drag"]["Solar flux"].lower() == "variable":
    inpFile[17] = "iF107:     1\n"
  else:
    raise ValueError('Value "' + model["Drag"]["Solar flux"] + '" invalid.')

  if model["SRP"]["Flag"] == False:
    inpFile[18] = "iSRP:      {:1d}\n".format(int(model["SRP"]["Flag"]))
  else:
    inpFile[18] = "iSRP:      {:1d}\n".format(int(model["SRP"]["Flag"]))
    if model["SRP"]["Eclipses"]:
       inpFile[18] = "iSRP:      2\n"
  
  if model["Lunisolar"]["Ephemerides"] == "DE431":
    inpFile[19] = "iephem:    1\n"
  elif model["Lunisolar"]["Ephemerides"] == "Meeus":
    inpFile[19] = "iephem:    2\n"
  else:
    raise ValueError('Value "' + model["Lunisolar"]["Ephemerides"] + '" invalid.')
  
  inpFile[20] = "gdeg:    {:3d}\n".format(model["NS gravity"]["Degree"])
  if model["NS gravity"]["Order"] <= model["NS gravity"]["Degree"]:
    inpFile[21] = "gord:    {:3d}\n".format(model["NS gravity"]["Order"])
  else:
    raise ValueError("Order {0:d} of the gravity field is greater than degree {1:d}".format(model["NS gravity"]["Order"],model["NS gravity"]["Degree"]))
  


  # Integration settings
  integ = gSettings["Integration"]
  inpFile[29] = "tol:      {:22.15E}\n".format(integ["Tolerance"])
  inpFile[30] = "tspan:    {:22.15E}\n".format(integ["Duration"] * 365.25)
  inpFile[31] = "tstep:    {:22.15E}\n".format(integ["Step"])
  inpFile[39] = "eqs:      {:2d}\n".format(integ["Equations"])



  # Output settings
  inpFile[44] = "verb:     0\n"
  inpFile[45] = "out:   " + os.path.abspath(os.path.join(dirPath, ' '))


  with open(os.path.join(dirPath,'input.txt'),'w') as f:
    f.writelines(inpFile)
  



def createObject(dirPath,gSettings,ICs):
  """
  Creates an object.txt file in a format readable by the THALASSA Fortran
  executable, using the settings in the grid definition dictionary.
  It reads the input file template from ../in/object.txt, so keep this up to date
  in case that changes!

  Author:
    Davide Amato
    The University of Arizona
    davideamato@email.arizona.edu
  
  Revisions:
    180526: function created.
  """
  
  with open(os.path.join('../in','object.txt')) as f:
    objFile = f.readlines()
  
  objFile[3] = "{:+22.15E}; MJD  [TT]\n".format(ICs[0])
  objFile[4] = "{:+22.15E}; SMA  [km]\n".format(ICs[1])
  objFile[5] = "{:+22.15E}; ECC  [-]\n".format(ICs[2])
  objFile[6] = "{:+22.15E}; INC  [deg]\n".format(ICs[3])
  objFile[7] = "{:+22.15E}; RAAN [deg]\n".format(ICs[4])
  objFile[8] = "{:+22.15E}; AOP  [deg]\n".format(ICs[5])
  objFile[9] = "{:+22.15E}; M    [deg]\n".format(ICs[6])

  SCraft = gSettings["Spacecraft"]
  objFile[11] = "{:+22.15E}; Mass        [kg]\n".format(SCraft["Mass"])
  objFile[12] = "{:+22.15E}; Area (drag) [m^2]\n".format(SCraft["Drag area"])
  objFile[13] = "{:+22.15E}; Area (SRP)  [m^2]\n".format(SCraft["SRP area"])
  objFile[14] = "{:+22.15E}; CD          [-]\n".format(SCraft["CD"])
  objFile[15] = "{:+22.15E}; CR          [-]\n".format(SCraft["CR"])

  with open(os.path.join(dirPath,'object.txt'),'w') as f:
    f.writelines(objFile)
  



def main():
  
  parser = argparse.ArgumentParser(description='Generate a grid of orbital '
  'elements for propagation with THALASSA.')
  
  parser.add_argument('outDir',nargs='?',\
  help='path to the output directory for the batch propagations')
  if len(sys.argv)==1:
    parser.print_help(sys.stderr)
    sys.exit(1)
  args = parser.parse_args()
  




  gridDefFile = 'griddef.json'
  print('THALASSA GRID CREATION SCRIPT')
  print('Reading grid definition from ' + os.path.abspath(gridDefFile) + '...', end=" ")

  # Read grid definition from the griddef file in JSON format. SMA is in km,
  # and angles are in degrees.
  with open(gridDefFile,'r') as f:
    gridDefDict = json.load(f)
  
  print('Done.\n')

  nTot = 1
  for icVal in gridDefDict["Grid"]:
    nTot = nTot * gridDefDict["Grid"][icVal]['points']

  proceedMsg = """You are creating a grid for {0} propagations.
**WARNING**: This will also delete everything in the output directory, if it exists.
Do you want to continue? (Y/N)\n""".format(nTot)

  proceed = input(proceedMsg)
  if proceed.lower() != 'y':
    sys.exit(1)

  
  print('Preparing a grid for {0} propagations...'.format(nTot), end=" ", \
  flush=True)
  
  grid = genGrid(nTot,gridDefDict["Grid"])




  # Create grid file and copy griddef.json in the output directory
  if os.path.exists(os.path.abspath(args.outDir)):
    shutil.rmtree(args.outDir)
  os.makedirs(os.path.abspath(args.outDir))
  
  now = datetime.datetime.now()
  gridHeader = '# THALASSA GRID FILE\n# Generated on ' + \
  now.isoformat() +  '.\n# Columns: SID, MJD (TT), SMA (km), ECC, INC (deg), ' \
  'RAAN (deg), AOP (deg), MA (deg)\n'

  with open(os.path.join(args.outDir,'grid.dat'),'w') as f:
    f.write(gridHeader)
    np.savetxt(f,grid[:,:],fmt='%010u,' + 7*'%22.15E,')

  print('Done.')
  print('Grid table written to ' + os.path.join(args.outDir,'grid.dat'))

  shutil.copyfile('./griddef.json',os.path.join(args.outDir,'griddef.json'))
  print('Grid definition file copied to ' + 
  os.path.abspath(os.path.join(args.outDir,'griddef.json')))

  


  print('Creating output directories...', end=" ", flush=True)

  # Divide the grid into chunks of "chunkSize" simulations each
  nChunks = nTot // chunkSize
  for iDir in range(1,nChunks + 2):
    chunkTxt = 'C{:03d}'.format(iDir)
    subDir = os.path.join(args.outDir,chunkTxt)
    os.makedirs(subDir)
    
    startSID = (iDir - 1) * chunkSize
    endSID   = min((iDir * chunkSize),nTot)
    igrid    = 0

    for SID in grid[startSID:endSID,0]:
      SIDtxt = 'S{:010d}'.format(int(SID))
      subSubDir = os.path.join(subDir,SIDtxt)
      os.makedirs(subSubDir)
      
      createInput(subSubDir,gridDefDict)
      createObject(subSubDir,gridDefDict,grid[startSID + igrid,1:])
      
      igrid += 1

  print("Done.")
  

if __name__ == '__main__':
  main()
