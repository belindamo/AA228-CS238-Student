import sys
import numpy as np

def init_Q(infile):
  if infile == "data/small.csv":
    return np.zeros((100, 4))
  elif infile == "data/medium.csv":
    return np.zeros((50000, 7))
  elif infile == "data/large.csv":
    return np.zeros((312020, 9))
  else:
    raise Exception("Incorrect data file input name")

def init_gamma(infile):
  if infile == "data/small.csv" or infile == "data/large.csv":
    return 0.95
  elif infile == "data/medium.csv":
    return 1
  else: 
    raise Exception("Incorrect data file input name")

def compute(infile, outfile):
    file = open(infile, 'r')
    data = file.readlines()
    data.pop(0) # remove headers line
    file.close()

    Q = init_Q(infile)
    H, W = Q.shape
    alpha = 0.2
    gamma = init_gamma(infile)

    # train model via Q-learning
    for i in range(1):
      for line in data:
        row = np.fromstring(line, dtype='int', sep=',')
        s = row[0] - 1 # for s, a, sp - convert from 1-index to 0-index
        a = row[1] - 1
        r = row[2]
        sp = row[3] - 1
        max_ap = np.argmax(Q[sp,:])
        Q[s,a] = Q[s,a] + ( alpha * ( r + (gamma * max_ap) - Q[s,a] ) )

    # output data
    out = open(outfile, "a")
    out.seek(0)
    for s in range(H):
      state = Q[s]
      optimal_action = np.argmax(state) + 1
      line = "{}\n".format(optimal_action)
      out.write(line)
    out.close()

def main():
  if len(sys.argv) != 3:
    raise Exception("usage: python project1.py <infile>.csv <outfile>.gph")

  inputfilename = sys.argv[1]
  outputfilename = sys.argv[2]
  compute(inputfilename, outputfilename)

if __name__ == '__main__':
  main()
