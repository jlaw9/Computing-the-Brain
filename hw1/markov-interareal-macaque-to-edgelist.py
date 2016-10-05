#! /usr/bin/env python

import networkx as nx

interareal_macaque_file = "./2012-cerebral-cortex-markov-weighted-directed-interareal-macaque.txt"
G = nx.DiGraph()
targets = set()
with open(interareal_macaque_file, 'r') as in_file:
    in_file.readline()
    for line in in_file:
        line = line.rstrip().split('\t')
        targets.add(line[3])

with open(interareal_macaque_file, 'r') as in_file:
    # skip the header line
    # CASE	MONKEY	SOURCE	TARGET	FLNe	NEURONS	STATUS	BIBLIOGRAPHY
    in_file.readline()
    for line in in_file:
        line = line.rstrip().split('\t')
        source = line[2]
        target = line[3]
        w = float(line[4])
        # if this edge is not part of the 29x29 matrix, skip it
        if source not in targets:
            pass
        # if the edge is already in the network, take the average of the two
        elif (source, target) in G.edges():
            print "averaging %s->%s\t%0.3e with %0.3e" %(source, target, G.edge[source][target]['weight'], w)
            G.edge[source][target]['weight'] = (G.edge[source][target]['weight'] + w) / 2.0
        else:
            G.add_edge(source, target, weight=w)
            
print nx.info(G)
out_file = "./2012-cerebral-cortex-markov-weighted-directed-interareal-macaque-29x29edgelist.txt"
print "writing new edge list to: %s" % (out_file)

with open(out_file, 'w') as out:
    out.write('#tail\thead\tedge_weight\n')
    out.write('\n'.join(['%s\t%s\t%s' % (u,v,str(G.edge[u][v]['weight'])) for u,v in sorted(G.edges())]))
