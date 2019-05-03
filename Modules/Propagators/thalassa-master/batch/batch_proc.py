#!/usr/bin/env python

import os
import sys
import numpy as np

def readStats(statDir):
  # Read integration statistics from Thalassa's output file.
  
  statPath = os.path.join(statDir,'stats.dat')
  try:
    stat = np.loadtxt(statPath)
  except ValueError:
    print('Warning: invalid statistics file at ' + statPath)
    stat = np.ones(10)*np.nan
  
  return stat

def processSummary(masterPath,dirs):
  # Process Thalassa batch statistics and write to summary file.
  
  stats = []
  for direc in dirs:
    appo   = readStats(direc)
    
    CPUT   = appo[:,3]
    CPUAvg = np.average(CPUT)
    
    # Save run data to the array to be output
    stats.append(appo[0,:])
    stats[-1][3] = CPUAvg
  
  with open(os.path.join(masterPath,'summary.csv'),'w') as summFile:
    summFile.write('# THALASSA - BATCH PROPAGATION SUMMARY\n')
    summFile.write('# Legend:\n# Calls,Steps,Tolerance,Avg_CPU[s],MJD_f,SMA[km],'
    'ECC,INC[deg],RAAN[deg],AOP[deg],M[deg]\n# ' + 80*'=' + '\n')
    try:
      for line in stats:
        saveLine = np.asarray(line)
        np.savetxt(summFile,saveLine.reshape(1,11),fmt=2*'%12i,' + 2*'%13.6g,' + 7*'%22.15E,')
    
    except TypeError:
      summFile.write(11*'NaN, ' + '\n')

def main():
  args = sys.argv[1:]

  if not args:
    print('Usage: ./batch_proc.py MASTER_DIRECTORY\n'
          'Process data from Thalassa batch propagations saved in MASTER_DIRECTORY.')
    sys.exit(1)
  
  masterPath = os.path.abspath(args[0]); del args[0]

  # Walk over directory list
  relDirs = next(os.walk(masterPath))[1]
  dirs = [os.path.join(masterPath,direc) for direc in sorted(relDirs)]
  
  processSummary(masterPath,dirs)

  
  



if __name__ == '__main__':
  main()