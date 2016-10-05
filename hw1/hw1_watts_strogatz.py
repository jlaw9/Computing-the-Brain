#! /usr/bin/env python

# this script is for assignment 1 in the Computing the Brain Course

import matplotlib.pyplot as plt
import networkx as nx
import sys
import os
import numpy as np
import random
from tqdm import tqdm
from optparse import OptionParser
import utilsPoirel as utils


def generate_watts_strogatz_graph(n, k, p):
    """  A single call to this function should create an undirected, regular, ring graph with n nodes and nk/2 edges, and then rewire every edge with probability p, as described in the caption of Figure 1 of Collective dynamics of 'small-world' networks. The parameters n and k are integers and k can be even. 
    *n*: the number of the nodes in the network
    *k*: the average degree of a node
    *p*: a probability between 0 and 1. 
    *returns*: a networkx undirected Graph 
    """

    G = nx.Graph()
    # add the nodes
    for i in xrange(n):
        G.add_node(i)

    # add the edges into a ring lattice
    num_edges = (n*k)/2
    for i in xrange(num_edges):
        curr_node = i % n
        # use division and modulators to find out what the next node to connect an edge to should be
        next_node = (((i+1) % n) + (i/n)) % n

        #print i, curr_node, next_node
        G.add_edge(curr_node, next_node)

    # check to make sure edges can be rewired
    num_complete_graph_edges = (n*(n-1)) / 2
    if num_edges == num_complete_graph_edges:
        print "graph is complete. Not rewiring edges"
        return G

    total_rewired = 0
    # now rewire the edges
    for i in xrange(num_edges):
        tail = i % n
        head = (((i+1) % n) + (i/n)) % n
        # rewire each edge with probability p 
        rewire = random.random() <= p
        if rewire:
            total_rewired += 1
            rand_head = head
            # rewire the edge by connecting to a random next_node 
            # force the random edge to not be a duplicate or already existing edge
            while rand_head in G.neighbors(tail):
                # randint is inclusive so don't include the n node
                rand_head = random.randint(0,n-1)
                # edges are undirected so sort the node ids
            # remove the original edge and add the rewired edge
            G.remove_edge(tail, head)
            new_edge = tuple(sorted((tail, rand_head)))
            G.add_edge(new_edge[0], new_edge[1])
            #print "replaced: ", (tail, head), "with:", new_edge

    print "rewired %d / %d (%d%%) edges" % (total_rewired, len(G.edges()), (total_rewired / float(num_edges))*100)

    return G


# ------------ unused functions ----------------------
#def avg_shortest_path_length(G):
#    """ Compute the shortest path between all pairs of nodes and take the average
#    Calls the networkx function average_shortest_path_length() which calls nx.single_source_shortest_path_length(G, node) to compute the shortest path from each node to every other node
#    Takes about 2 minutes on a graph with 5000 nodes 
#    """
#    #l = -1
#    #max_num_nodes = 1000
#    #if len(G.nodes()) <= max_num_nodes:
#    # TODO this includes self loops (length of 0) lowering the path lengths for all of the nodes
#    l = nx.average_shortest_path_length(G)
#    #else:
#    #    print "average_shortest_path_length calculation not setup for graphs with nodes > %d yet" % (max_num_nodes)
#    #shortest_path_lengths = shortest_path_lengths(G)
#    return l
#
#
#def shortest_path_lengths(G):
#    """  calls nx.shortest_path_length(G, source=node) 
#    to compute the shortest path from each node to every other node.
#    Removes 'self' paths of length 0
#    """
#    shortest_path_lengths = {}
#    for node in G.nodes():
#        #shortest_path_lengths[node] = nx.single_source_dijkstra_path_length(G, node)
#        # single source shortest path length is for unweighted graphs
#        shortest_path_lengths[node] = nx.single_source_shortest_path_length(G, node)
#        # remove the self path of 0
#        del shortest_path_lengths[node][node]
#        #avg_shortest_path_lengths[n] = avg_spl
#
#    return shortest_path_lengths
#
#
#def clustering_coeff(G):
#    """ Calls the networkx function average_clustering() which estimates the average clustering coefficient of G
#    """
#    if isinstance(G, nx.DiGraph):
#        print "Warning: converting directed graph to undirected to compute clustering coefficient"
#    c = nx.average_clustering(G.to_undirected())
#    return c


def plot_watts_strogatz_graph(G, fig_file):
    nx.draw_circular(G, with_labels=False, color='k')
    #nx.draw_networkx(G, pos)
    plt.draw()
    #plt.show()
    plt.savefig(fig_file)
    plt.close()


def plot_plc(plc_vals, fig_file='', title=''):
    lp0 = plc_vals[0][0]['l']
    cp0 = plc_vals[0][0]['c']
    # take the p of 0 out of the dictionary so it's not in the results
    del plc_vals[0]
    l_ratios = {}
    c_ratios = {}
    for p in plc_vals:
        l_ratios[p] = []
        c_ratios[p] = []
        for attempt in plc_vals[p]:
            l_ratios[p].append((plc_vals[p][attempt]['l'] / float(lp0)))
            c_ratios[p].append((plc_vals[p][attempt]['c'] / float(cp0)))
            #print p, attempt, plc_vals[p][attempt]['l'], l_ratios[p][attempt], plc_vals[p][attempt]['c'], c_ratios[p][attempt]

    # compute the average and standard deviation of each using numpy
    l_avg = []
    l_std = []
    c_avg = []
    c_std = []
    for p in plc_vals:
        l_avg.append(np.mean(l_ratios[p]))
        l_std.append(np.std(l_ratios[p]))
        c_avg.append(np.mean(c_ratios[p]))
        c_std.append(np.std(c_ratios[p]))
        #print p, np.mean(l_ratios[p]), np.std(l_ratios[p]), np.mean(c_ratios[p]), np.std(c_ratios[p])
        #l_avg[p] = sum([l_ratio for l_ratio in l_ratios[p]]) / float(len(l_ratios))
        #c_avg[p] = sum([c_ratio for c_ratio in c_ratios[p]]) / float(len(c_ratios))

    #fig, ax = plt.subplot(1)

    # plot the avg (point) and std deviation (using numpy) of each of the given probability (p) values
    fig, ax = plt.subplots()
    p = [p for p in plc_vals]
    # plot them on the same figure
    plt.errorbar(p, l_avg, yerr=l_std, label='l(p)/l(0)', fmt='o', color='black', clip_on=False)
    plt.errorbar(p, c_avg, yerr=c_std, label='c(p)/c(0)', fmt='o', color='white', ecolor='black', clip_on=False)
    #plt.errorbar(p, l_avg, yerr=l_std, fmt='x', barsabove=True)
    #plt.errorbar(p, c_avg, yerr=c_std, fmt='*', barsabove=True)

    #plt.set_yaxis("")
    ax.set_xscale('log')
    ax.set_ylim(ymax=1.05, ymin=0.0)
    ax.set_xlabel("p (rewiring probability)")
    ax.set_ylabel("ratio of l(p)/l(0) or c(p)/c(0)")

    plt.title(title)
    plt.legend(loc='best')

    print "writing fig_file: %s" % (fig_file)
    plt.savefig(fig_file)
    #plt.show()
    plt.close()


def main():

    opts = parse_args()
    #n_sets = [100]
    n_sets = [100, 1000, 5000]
    k_sets = [20]
    #p_sets = [0, 0.1, 0.5, 0.8, 1]
    # do the 0 set to have the baseline
    p_sets = [0] + [round(0.05*k, 8) for k in range(2,20,5)] + [1]
    # choose an x of 0.6 as it has a range to 0.3*10^(-5)
    p_sets += [round(0.6**k, 8) for k in range(2,23,1)]
    print "using %d p_sets: %s" % (len(p_sets), str(p_sets))
    attempts = 10
    if opts.fig1:
        n_sets = [20]
        k_sets = [4]
        #p_sets = [0, 0.1, 0.5, 0.8, 1]
        # do the 0 set to have the baseline
        p_sets = [0, 0.05, 0.1, 1]
        print "using %d p_sets: %s" % (len(p_sets), str(p_sets))
        attempts = 1

    nkp_lc_vals = {}
    # initialize the dictionary
    for n in n_sets:
        nkp_lc_vals[n] = {}
        for k in k_sets:
            nkp_lc_vals[n][k] = {}
            for p in p_sets:
                nkp_lc_vals[n][k][p] = {}

    out_file = "%s.txt" % (opts.out_pref)
    if os.path.isfile(out_file):
        print "reading results from %s" % (out_file)
        for n, k, p, attempt, l, c in utils.readColumns(out_file, 1,2,3,4,5,6):
            nkp_lc_vals[int(n)][int(k)][float(p)][int(attempt)] = {'l': float(l), 'c': float(c)}

    # loop through all of the combinations of n k and p 
    for n in n_sets:
        for k in k_sets:
            for p in tqdm(p_sets):
                print "n: %d, k: %d, p: %0.2f" %(n, k, p)
                num_attempts = attempts
                # only need to try once for a p of 0
                if p == 0:
                    num_attempts = 1
                # try each attempt the specified number of times
                for attempt in tqdm(xrange(num_attempts)):
                    if attempt not in nkp_lc_vals[n][k][p]:
                        G = generate_watts_strogatz_graph(n, k, p)
                        #print "# nodes: ", len(G.nodes()), "# edges: ", len(G.edges())
                        #print "nodes: ", len(G.nodes()), G.nodes(), "edges: ", len(G.edges()), G.edges()
                        l = nx.average_shortest_path_length(G)
                        #print "avg_shortest_path_length:", l
                        c = nx.average_clustering(G)
                        #print "clustering_coeff:", c
                        nkp_lc_vals[n][k][p][attempt] = {'l': l, 'c': c}
                        if opts.fig1:
                            fig_file = opts.fig1 + "n%d-k%d-p%d.pdf" % (n, k, p*100)
                            plot_watts_strogatz_graph(G, fig_file)

    if not opts.fig1:
        # write the results after getting all of the values
        print "writing results to %s" % (out_file)
        out = open(out_file, 'w')
        out.write("#n\tk\tp\tattempt\tl\tc\n")
        for n in n_sets:
            for k in k_sets:
                for p in tqdm(p_sets):
                    num_attempts = attempts
                    if p == 0:
                        num_attempts = 1
                    for attempt in tqdm(xrange(num_attempts)):
                        out.write("%d\t%d\t%0.8f\t%d\t%0.8f\t%0.8f\n" % (n, k, p, attempt, nkp_lc_vals[n][k][p][attempt]['l'], nkp_lc_vals[n][k][p][attempt]['c']))

        # now plot the average and std deviation of each of the points (like fig2 in the watts_strogatz paper
        for n in n_sets:
            for k in k_sets:
                #l_ratios, c_ratios = compute_ratios(plc_vals)
                title = "Small-world n: %d k: %d" % (n, k)
                fig_file = "%s%dn-%dk-small-world.png" % (opts.out_pref, n, k)
                plot_plc(nkp_lc_vals[n][k], fig_file=fig_file, title=title)


def parse_args():
    parser = OptionParser()

    parser.add_option('-o','--out-pref',type='string',metavar='STR', default="watts-strogatz-results",
                      help='output file containing results. Default=watts-strogatz-results')
    parser.add_option('-f','--fig1',type='string',metavar='STR',
                      help='option to make rings like in fig 1.')

    (opts, args) = parser.parse_args()

    return opts


if __name__ == "__main__":
    main()
