# -*- coding: utf-8 -*-
"""
Created on Thu Sep 15 10:01:43 2016

@author: vladimir
"""

import numpy as np
from scipy.sparse import lil_matrix
from matplotlib import pyplot as plt
import networkx as nx
from fim import apriori
import logging
import sys


def test_data():
    #type: () -> List[int]
    data = [[1, 2, 3], [1, 4, 5], [2, 3, 4], [1, 2, 3, 4], [2, 3], [1, 2, 4], [4, 5], [1, 2, 3, 4],
            [3, 4, 5], [1, 2, 3]]

    data = [[number_to_character(x) for x in list1] for list1 in data]
    return data


def number_to_character(input_number):
    #type: (int) -> str 
    return chr(ord("a") - 1 + input_number)


def plot_matrix(data, num_baskets, num_items):
    H = lil_matrix((num_baskets, num_items), dtype=np.bool)
    for i in range(0, len(data) - 1):
        for j in list(map(int, data[i])):
            H[i, j - 1] = True
    plt.figure(1)
    plt.subplot(121)
    plt.spy(H)
    plt.title('Vector representation')
    plt.xlabel('Items')
    plt.ylabel('Baskets')
    plt.show()


def plot_graph(data, items):
    g = nx.Graph()
    N_baskets = len(data)
    M_items = len(items)
    a = ['b_' + str(i) for i in range(N_baskets)]
    b = ['i_' + str(j) for j in range(M_items)]
    g.add_nodes_from(a, bipartite=0)
    g.add_nodes_from(b, bipartite=1)

    i = 0
    for basket in data:
        for item in basket:
            g.add_edge(a[i], b[list(items).index(item)])
        i += 1

    # Draw this graph
    pos_a = {}
    x = 0.100
    const = 0.100
    y = 1.0
    for i in range(len(a)):
        pos_a[a[i]] = [x, y - i * const]

    xb = 0.500
    pos_b = {}
    for i in range(len(b)):
        pos_b[b[i]] = [xb, y - i * const]

    plt.subplot(121)
    nx.draw_networkx_nodes(g, pos_a, nodelist=a, node_color='r', node_size=300, alpha=0.8)
    nx.draw_networkx_nodes(g, pos_b, nodelist=b, node_color='b', node_size=300, alpha=0.8)

    # edges
    pos = {}
    pos.update(pos_a)
    pos.update(pos_b)
    nx.draw_networkx_edges(g, pos, edgelist=nx.edges(g), width=1, alpha=0.8, edge_color='g')
    nx.draw_networkx_labels(g, pos, font_size=10, font_family='sans-serif')

    plt.title('Graph representation')
    plt.show()


def read_file(file_path, process=lambda x: x):
    contents_list = list()
    logging.info(file_path)
    try:
        with open(file_path, encoding='utf8') as input_file:
            for line in input_file:
                line = process(line)
                if line:
                    contents_list.append(line)

            return contents_list

    except IOError as err:
        logging.warning('Failed to open file {0}'.format(err.message))
        sys.exit(1)


def main():
    ###############################################################################
    # Some basic data analysis
    ###############################################################################

    data = read_file("/Users/zxj/cs535/data/marketing.data", lambda x: x.split(","))
    frequent_itemset = apriori(data, supp=-3, zmin=2, target='s', report='a')
    rules = apriori(data, supp=-3, zmin=2, target='r', report='rCL')
    print("Frequent itemsets are: ")
    print(frequent_itemset)
    print("Rules are:")
    print(rules)


if __name__ == '__main__':
    main()