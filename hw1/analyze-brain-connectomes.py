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
# imported from /home/jeffl/src/python/scripts/trunk
import utilsPoirel as utils

# default dir containing the files
DATADIR = "."
# names of the files
CONNECTOMEFILES = {
    "human_coactivation": "%s/2013-PNAS-Crossley-Cognitive-relevance-coactivation-matrix.txt",
    "macaque_cortex": "%s/1993-Proc-Royal-Society-organization-neural-systems-macaque71.txt",
    "macaque_visual_cortex": "%s/1991-cerebral-cortex-felleman-primate-cerebral-cortex-fv30.txt",
    "macaque_functional": "%s/2007-pnas-honey-network-structure-functional-connectivity-macaque47.txt",
    "cat_cortex": "%s/1995-journal-neuroscience-connectivity-cerebral-cortex-cat-cjall.txt",
    "macaque_interareal": "%s/2012-cerebral-cortex-markov-weighted-directed-interareal-macaque-29x29edgelist.txt",
}
# a dictionary containing a networkx graph for each of the networks
CONNECTOMES = [
    "human_coactivation",
    "macaque_cortex",
    "macaque_visual_cortex",
    "macaque_functional",
    "cat_cortex",
    "macaque_interareal",
]
#1. Human Coactivation (Crossley, 2013)
#2. Macaque Cortex (Young, 1993)
#3. Macaque Visual Cortex (FVE, 1991)
#4. Macaque Functional (Honey et al, 2007)
#5. Cat Cortex (Scannell et al, 1995)
#6. Macaque Interareal (Markov et al, 2013)

CONNECTOMELABELS = {
    "human_coactivation": "Human Coactivation (2013)",    
    "macaque_cortex": "Macaque Cortex (1993)",                           #    Young 1993  # blue star (b*)
    "macaque_visual_cortex": "Macaque Visual Cortex (1991)",                      #    Felleman and Van Essen, 1991  # purple circle
    "macaque_functional": "Macaque Functional (2007)",             #    Honey et al, 2007  # green triangle
    "cat_cortex": "Cat Cortex (1995)",
    "macaque_interareal": "Macaque Interareal (2013)",            #    Markov et al, 2013  # black traingle
}

# connectome dictionary where the value is a tuple of the shape and color
MARKERCOLOR = {
        "human_coactivation": ('r', 's'),       
        "macaque_cortex": ('b', '*'),     
        "macaque_visual_cortex": ('m', 'o'),     
        "macaque_functional": ('g', '>'), 
        "cat_cortex": ('r', 'D'),         
        "macaque_interareal": ('k', '^'), 
}


# Plot the average shortest path length and the average clustering coefficient for all these networks on a single plot.
def plot_avg_l_c(l_avgs, c_avgs, out_file=''):
    fig, ax = plt.subplots()

    index = np.arange(len(CONNECTOMES))
    print "average path lengths:", l_avgs
    print "average clustering coefficients:", c_avgs
    opacity = 0.7
    bar_width = 0.35

    # put everything in a bar plot
    plt.bar(index, l_avgs, bar_width, 
            alpha=opacity, color='b', label="Average path length", )
    plt.bar(index+bar_width, c_avgs, bar_width, 
            alpha=opacity, color='g', label="Average clustering coefficient")

    ax.set_title("Average Connectome Shortest Path Length and Clustering Coefficient")
    ax.set_ylabel("Average path length or \nAverage clustering coefficient")
    ax.set_xticklabels([CONNECTOMELABELS[connectome] for connectome in CONNECTOMES], rotation=15)

    plt.ylim(ymax=2.9)
    plt.legend()
    plt.tight_layout()

    if out_file:
        plt.savefig(out_file)
    else:
        plt.show()
    plt.close()


# simple function to plot a histogram
def plot_hist(vals, bins=10, title='', x_label='', y_label='', out_file=''):
    plt.figure()

    opacity = 0.7

    plt.hist(vals, bins=bins, alpha=opacity)
    plt.title(title)
    plt.xlabel(x_label)
    plt.ylabel(y_label)

    if out_file:
        plt.savefig(out_file)
    else:
        plt.show()
    plt.close()


def main():

    opts = parse_args()

    DATADIR = opts.datadir
    if not opts.datadir:
        print "Using current directory as datadir. Use the --datadir option to specify another location"
        DATADIR = '.'
    OUTDIR = opts.outdir
    if not os.path.isdir(OUTDIR):
        print "Warning: %s does not exist. Creating it" % OUTDIR
        os.makedirs(OUTDIR)

    for connectome in CONNECTOMES:
        CONNECTOMEFILES[connectome] = CONNECTOMEFILES[connectome] % (DATADIR)

    connectome_graphs = {}
    # first parse the edge_lists
    for connectome in CONNECTOMES:
        print "Parsing %s from %s" % (connectome, CONNECTOMEFILES[connectome])
        G = nx.DiGraph()
        for u,v,w in utils.readColumns(CONNECTOMEFILES[connectome], 1,2,3):
            G.add_edge(u,v,weight=float(w))
        connectome_graphs[connectome] = G

    # plot the average shortest path length
    fig_file = "%s/avg_l_c.png" % (OUTDIR)
    if not os.path.isfile(fig_file) or opts.force:
        print "\n" + '-'*20
        print "Computing the average path length and clustering coefficients"
        # TODO the density by avg path length section uses this list
        l_avgs = []
        c_avgs = []
        for connectome in CONNECTOMES:
            G = connectome_graphs[connectome]
            l_avgs.append(nx.average_shortest_path_length(G))
            c_avgs.append(nx.average_clustering(G.to_undirected()))

        print "Plotting the average path length and clustering coefficients to %s" % (fig_file)
        plot_avg_l_c(l_avgs, c_avgs, fig_file)
    else:
        print "%s already exists. Skipping. Use --force to plot anyway" % (fig_file)

    print "\n" + '-'*20
    # Separately for each network, plot a histogram of the average shortest path length from each node (to every other node) and the clustering coefficient of each node.
    for connectome in CONNECTOMES:
        G = connectome_graphs[connectome]
        fig_file = "%s/hist-aspl-%s.png" % (OUTDIR, connectome)
        if not os.path.isfile(fig_file) or opts.force:
            avg_shotest_path_lengths = []
            # calls nx.single_source_shortest_path_length(G, node) to compute the shortest path from each node to every other node
            for node in G.nodes():
                # compute the shortest path lengths from this node to every other node
                shortest_path_lengths = nx.single_source_shortest_path_length(G, node)
                avg_spl = sum(shortest_path_lengths.values()) / float(len(G)-1)
                avg_shotest_path_lengths.append(avg_spl)
            # now plot a histogram of the average shortest path lengths
            print "Plotting:", fig_file
            plot_hist(avg_shotest_path_lengths,
                      title="Average shortest path lengths histogram of %s" % CONNECTOMELABELS[connectome],
                      x_label="average shortest path length", y_label="frequency",
                      out_file=fig_file)
        fig_file = "%s/hist-cc-%s.png" % (OUTDIR, connectome)
        if not os.path.isfile(fig_file) or opts.force:
            # compute the clustering coefficient of each node
            if isinstance(G, nx.DiGraph):
                print "Warning: converting directed graph to undirected to compute clustering coefficient"
            clustering_coefficients = nx.clustering(G.to_undirected()).values()

            # now plot a histogram of the clustering coefficients
            print "Plotting:", fig_file
            plot_hist(clustering_coefficients,
                      title="Clustering coefficients histogram of %s" % CONNECTOMELABELS[connectome],
                      x_label="clustering coefficient", y_label="frequency",
                      out_file=fig_file)

    fig_file = "%s/density_by_avg_path_length.png" % (OUTDIR)
    if not os.path.isfile(fig_file) or opts.force:
        print "\n" + '-'*20
        print "Plotting the average path length against the density of the network"
        # For these networks, replicate the analysis in Figure 1A of Cortical High-Density Counterstream Architectures.
        #Specifically, for each network, plot the average path length against the density of the network.
        densities = []
        for connectome in CONNECTOMES:
            # compute the density of each network
            densities.append(compute_density(connectome_graphs[connectome]))
        # plot the density by average path length
        print "Plotting:", fig_file
        plot_density_vs_apl(l_avgs, densities, fig_file)

    # compute the average path length after systematically lowering the density of the network
    fig_file = "%s/graph_density_edge_removal.png" % (OUTDIR)
    if not os.path.isfile(fig_file) or opts.force:
        print "\n" + '-'*20
        print "Computing the average path length after systematically lowering the density of the network"
        graph_density_edge_removal(connectome_graphs['macaque_interareal'], 
                densities_to_test=range(6,0,-1), 
                num_random_tests=100, 
                fig_file=fig_file)
    print "\n" + '-'*20
    print 'Done'


def compute_density(G):
    # compute the density of each network
    n = len(G.nodes())
    if isinstance(G, nx.DiGraph):
        # total number of edges in a directed graph
        num_complete_graph_edges = (n*(n-1))
    else:
        # total number of edges in an udirected graph
        num_complete_graph_edges = (n*(n-1))/2
    density = len(G.edges()) / float(num_complete_graph_edges)
    return density


# plot the density by average path length
def plot_density_vs_apl(l_avgs, densities, fig_file=''):
    fig, ax = plt.subplots()

    index = 0
    for connectome in CONNECTOMES:
        # plot each of the connectomes with a different shape and color
        #ax.plot(l_avgs[index], densities[index], shape=shape_color[connectome][0], color=shape_color[connectome][1])
        ax.scatter(densities[index], l_avgs[index], 
                marker=MARKERCOLOR[connectome][1], #markersize=10, 
                color=MARKERCOLOR[connectome][0], label=CONNECTOMELABELS[connectome])
        index += 1

    plt.title("Density by Average Path Length")
    plt.xlabel("density")
    plt.ylabel("average path length")
    plt.ylim(ymax=4, ymin=0.9)
    plt.xlim(xmax=0.72, xmin=-.05)
    plt.legend()

    plt.savefig(fig_file)
    plt.close()


    # Perform this analysis only for G29x29 (macaque_weighted). 
    #For each value of density equal to i/10,1<=i<=6, remove a random set of edges from G29x29 to get a graph with density i/10.
    #Compute the average path length of this graph.
    #Repeat this process 100 times for each density value.
    #Plot the average and standard deviation of the average path lengths for each value of density.
    #This analysis mimics that used by the authors to get the "gray area" in Figure 1A.
def graph_density_edge_removal(G, densities_to_test=range(6,0,-1), num_random_tests=100, fig_file=''):
    target_densities = [i/10.0 for i in densities_to_test]
    print "target_densities:", target_densities
    average_path_lengths = {}
    for target_density in tqdm(target_densities):
        if target_density not in average_path_lengths:
            average_path_lengths[target_density] = []
        for j in xrange(num_random_tests):
            # remove a random set of edges to get a graph with density i/10
            less_dense_G = lower_graph_density(G.copy(), target_density)
            average_path_length = nx.average_shortest_path_length(less_dense_G)
            average_path_lengths[target_density].append(average_path_length)
            # TODO try a single run first
            #break

    # compute the average and standard deviation of each using numpy
    l_avg = []
    l_std = []
    for target_density in target_densities:
        l_avg.append(np.mean(average_path_lengths[target_density]))
        l_std.append(np.std(average_path_lengths[target_density]))
        print "target_density:", target_density, "l_avg:", l_avg, "l_std:", l_std
    # now plot the average and std deviation of each of the densities
    fig, ax = plt.subplots()
    plt.errorbar(target_densities, l_avg, yerr=l_std, fmt='o', color='b')
    plt.ylim(ymax=4, ymin=0.9)
    plt.xlim(xmax=0.72, xmin=-.05)
    plt.xlabel("density")
    plt.ylabel("Average shortest path length")
    plt.title("Average path length after edge removal")

    print "writing fig_file: %s" % (fig_file)
    plt.savefig(fig_file)
    plt.close()


def lower_graph_density(G, target_density):
    # TODO do I need to make a copy of G?
    n = len(G.nodes())
    # convert the density to a target number of edges
    target_num_edges = target_density * (n*(n-1))
    num_edges_to_remove = len(G.edges()) - target_num_edges
    # use numpy random choice without replacement to get a set of edges to remove
    indices_of_edges_to_remove = np.random.choice(xrange(len(G.edges())), num_edges_to_remove, replace=False)
    # get the actual edges
    edges_to_remove = []
    for index in indices_of_edges_to_remove:
        edges_to_remove.append(G.edges()[index])
    # now remove those edges
    for u,v in edges_to_remove:
        G.remove_edge(u,v)
    #print "curr_density:", compute_density(G), "target_density:", target_density

    # make sure the graph is still connected
    if not nx.is_connected(G.to_undirected()):
        # If it's not, just take the subgraph of the largest connected component in the undirected graph
        largest_cc = []
        for connected_component in nx.connected_component_subgraphs(G.to_undirected()):
            if len(connected_component) > len(largest_cc):
                largest_cc = connected_component
        subG = G.subgraph(largest_cc)
        print "took the largest connected component of G. # nodes before and after: %d, %d" % (len(G.nodes()), len(subG.nodes()))
        return subG
    else:
        return G



def parse_args():
    parser = OptionParser()

    parser.add_option('-o','--outdir', type='string',metavar='STR', default="viz",
                      help='output dir containing results. Default=viz')
    parser.add_option('','--datadir', type='string',
                      help='Data directory containing the "*.mat" files. Default is current dir.')
    parser.add_option('','--force', action="store_true",
                      help='Force plotting even if the file already exists.')

    (opts, args) = parser.parse_args()

    return opts


if __name__ == "__main__":
    main()
