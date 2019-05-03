#!/usr/bin/env python3
"""
BATCH TOLERANCE SCRIPT FOR THALASSA
Launches batch propagations in which the tolerance parameter in input.txt
is progressively varied in the interval [tolmin,tolmax], with logarithmic
spacing. The output of each propagation is saved to a separate directory.
The script also compiles a summary file for the batch propagations.

Author:
  Davide Amato
  The University of Arizona
  davideamato@email.arizona.edu

Revisions:
  180809: Improved comments. Moved this script to the batch/ directory. Change 
  input file formatting for compatibility with THALASSA v1.0.
  190116: Change line numbers in modifyInput to accommodate the flag "imcoll" in
  input.txt
  
"""

import sys
import os
import subprocess
import shutil
import numpy as np
import datetime
import time
import multiprocessing as mp
import psutil
from batch_proc import processSummary

def generateTolVec(tolMax,tolMin,ntol):
  # Generates the tolerance vector, logarithmically spaced from 'tolmax' to
  # tolmin.
  tolVec = np.logspace(float(tolMax),float(tolMin),num=ntol)
  return tolVec

def modifyInput(inpPath,lineN,tol,outPath,eqs):
  # Modifies the line 'lineN' (counted from zero) in the input file to assign
  # the specified tolerance 'tol'.
  # TODO: Use regex to find the lines to modify.
  
  # Read contents of input file
  inpFile = open(inpPath,'rU')
  lines   = inpFile.readlines()
  inpFile.close
  
  # Change tolerance, equations, and output path
  lines[lineN[0]] = lines[lineN[0]][:11] + '%.15E\n' % (tol)
  lines[lineN[1]] = lines[lineN[1]][:11] + str(eqs) + '\n'
  lines[lineN[2]] = lines[lineN[2]][:5]  + outPath + '/\n'
  
  # Write contents to input file
  inpFile = open(inpPath,'w')
  inpFile.writelines(lines)
  inpFile.close

def thalassaRep(rep_time,inputPath,ICPath):
 # Launch the same thalassa propagation several times.
  for _ in range(0,rep_time):
    os.chdir('../')
    subprocess.call(['./thalassa.x',os.path.abspath(inputPath),
    os.path.abspath(ICPath)])
    os.chdir('./batch')

def tolRun(tolVec,tol,eqs,rep_time,masterPath,ICPath):
  # Execute a run in tolerance.
  # =======

  print('\nStarting propagation',str(np.where(tolVec == tol)[0][0]+1),'out of',str(len(tolVec)),'...')
  subDir = '%.5g' % np.log10(tol)

  # Generate an input file in the current output folder by copying and
  # modifying the one that is already in the MASTER_FOLDER
  outPath = os.path.join(masterPath,'tol' + subDir)
  if os.path.exists(outPath):
      print('Output path exists, its contents will be PURGED.')
      shutil.rmtree(outPath)
  os.makedirs(outPath)
  inputPath = os.path.join(outPath,'input.txt')
  shutil.copy(os.path.join(masterPath,'input.txt'),inputPath)
  modifyInput(inputPath,[29,39,45],tol,outPath,eqs)
  
  # Launch the propagations over the number of available CPUs
  thalassaRep(rep_time,inputPath,ICPath)

  return outPath


def main():
  args = sys.argv[1:]
  
  if not args:
    print ('Usage: ./batch_tol.py MASTER_DIRECTORY [--tmax log10(tolmax)]'
           '[--tmin log10(tolmin)] [--ntol ntol] [--eqs eqs]')
    print ('The script reads initial conditions and settings from the '
           '"object.txt" and "input.txt" files, respectively. These *must be '
           'already present* in the MASTER_DIRECTORY.')
    sys.exit(1)
  
  masterPath = os.path.abspath(args[0]); del args[0]
  l10tMax = -4.
  l10tMin = -15.
  ntol    = 12
  eqs     = 1

  # Output to user
  date_start = datetime.datetime.now()
  print('Thalassa - batch propagation in tolerance and equations.')
  print('Batch is starting on', date_start)
  
  # Command line parsing
  try:
    if args[0] == '--tmax':
      l10tMax = args[1]
      del args[0:2]
    
    if args[0] == '--tmin':
      l10tMin = args[1]
      del args[0:2]
    
    if args[0] == '--ntol':
      ntol = args[1]
      del args[0:2]
    
    if args[0] == '--eqs':
      eqs = args[1]
      del args[0:2]
  except IndexError:
    pass
  
  tolVec  = generateTolVec(l10tMax,l10tMin,ntol)
  
  # Initializations
  rep_time = 3
  ICPath   = os.path.join(masterPath,'object.txt')
  
  # Launch propagations
  # nproc = psutil.cpu_count(logical=False)
  # # nproc = 2
  outDirs = []
  runArgs = [(tolVec,tol,eqs,rep_time,masterPath,ICPath) for tol in tolVec]
  # with mp.Pool(processes=(nproc-2)) as pool:
  with mp.Pool(processes=1) as pool:
    outDirs = pool.starmap(tolRun,runArgs)
  
  date_end = datetime.datetime.now()
  print('Batch ended on', date_end)
  print('Total duration:', date_end - date_start)
  
  # Process statistics
  sys.stdout.write('Processing statistics... ')
  processSummary(masterPath,outDirs)
  sys.stdout.write('Done.\n')


if __name__ == '__main__':
  main()
