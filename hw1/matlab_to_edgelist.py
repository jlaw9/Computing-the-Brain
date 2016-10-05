#! /usr/bin/env python

# These datasets were downloaded from here: http://courses.cs.vt.edu/cs6824/2016-fall/assignments/assignment-1.html
# for assignment 1 of the cs6824 Computing the Brain course.
# Use scipy to parse the matlab files, then convert the resulting numpy matrices into an edge list file format

__author__ = "Jeff Law"

# installed using 'pip install --user scipy'
from scipy.io import loadmat
import networkx as nx
import sys
from optparse import OptionParser

# default dir containing the files
DATADIR = "."
# names of the files
MLFILES = {
    "primate_cortex": "%s/1991-cerebral-cortex-felleman-primate-cerebral-cortex-fv30.mat",
    "macaque_cortex": "%s/1993-Proc-Royal-Society-organization-neural-systems-macaque71.mat",
    "cat_cortex": "%s/1995-journal-neuroscience-connectivity-cerebral-cortex-cat.mat",
    "macaque_functional": "%s/2007-pnas-honey-network-structure-functional-connectivity-macaque47.mat",
    # just use the coactivation matrix and assign the matrix position as the node name
    "coactivation": "%s/2013-PNAS-Crossley-Cognitive-relevance-coactivation-matrix.mat",
}
# format of the matlab files
MLFORMAT = {
    "primate_cortex": ('CIJ', 'Names'),
    "macaque_cortex": ('CIJ', 'Names'),
    # has CIJctx and CIJall
    #"cat_cortex": ('CIJctx', 'Names'),
    "cat_cortex": ('CIJall', ''),
    "macaque_functional": ('CIJ', 'Names'),
    # doesn't have names, has coordinates
    "coactivation": ('Coactivation_matrix', ''),
}


def matlab_to_nx(ml_matrix, ml_matrix_names='', name=''):
    """ function to parse a scipy.io.loadmat loaded matlab file and store it as a networkx graph object
    *ml_matrix*: matlab (numpy) matrix
    *ml_matrix_names*: names of the rows/columns of the matrix
    *name*: name of the graph (optional)
    *returns*: a networkx graph object or -1 if (# rows != # of columns or # rows != # names)
"""
    print "# of rows and columns in ml_matrix: %d %d" % (len(ml_matrix), len(ml_matrix[0]))
    if len(ml_matrix) != len(ml_matrix[0]):
        print "Error: # of rows in ml_matrix != # columns"
        return -1
    if len(ml_matrix_names) != 0 and len(ml_matrix) != len(ml_matrix_names):
        print "Error: # of rows in ml_matrix != # of names: %d %d" % (len(ml_matrix), len(ml_matrix_names))
        return -1
    G = nx.DiGraph(name=name)
    for i in xrange(len(ml_matrix)):
        for j in xrange(len(ml_matrix[i])):
            if ml_matrix[i][j] > 0:
                if len(ml_matrix_names) != 0:
                    u = str(ml_matrix_names[i].strip())
                    v = str(ml_matrix_names[j].strip())
                else:
                    u = i
                    v = j
                G.add_edge(u, v, weight=ml_matrix[i][j])

    print nx.info(G)

    return G


def main():
    # define arguments
    parser = OptionParser()
    parser.add_option('','--datadir',type='string',\
                      help='Data directory containing the "*.mat" files. Default is current dir.')
    # parse the command line arguments
    (opts, args) = parser.parse_args()

    DATADIR = opts.datadir
    if not opts.datadir:
        print "Using current directory as datadir. Use the --datadir option to specify another location"
        DATADIR = '.'

    for mlfile in MLFILES:
        MLFILES[mlfile] = MLFILES[mlfile] % (DATADIR)

    for mlfile in MLFILES:
        print "\n" + '-'*20
        print "loading %s (%s)" % (mlfile, MLFILES[mlfile])
        loaded_mlfile = loadmat(MLFILES[mlfile])
        ml_matrix = loaded_mlfile[MLFORMAT[mlfile][0]]
        names = '' if MLFORMAT[mlfile][1] == '' else loaded_mlfile[MLFORMAT[mlfile][1]]
        G = matlab_to_nx(ml_matrix, names, name=mlfile)
        if G == -1:
            # if the parser failed, skip this entry
            continue
            #print "Quitting"
            #sys.exit()

        out_file = MLFILES[mlfile].replace('.mat','') + '.txt'
        print "writing %s to: %s" % (mlfile, out_file)

        with open(out_file, 'w') as out:
            out.write('#tail\thead\tedge_weight\n')
            out.write('\n'.join(['%s\t%s\t%s' % (u,v,str(G.edge[u][v]['weight'])) for u,v in sorted(G.edges())]))
    print "done"


if __name__ == "__main__":
    main()
