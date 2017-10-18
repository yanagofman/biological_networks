from goatools import obo_parser
from itertools import repeat
import networkx as nx
import pickle
import datetime
import pandas as pd
from tqdm import tqdm
import os



''' ontotype utilities to be used for constructing input files and parsing GO data'''

def create_GO_terms_dict(obo):
    """
    This method creates a dictionary of each GO term from an obo file using GOATools package
    :param obo: a path to an obo file
    :return: a dictionary
    """

    go = obo_parser.GODag(obo)
    return go

def get_human_go_ids(goa_humanDB):
    """
    get the GO ids which appear in goahuman DB.
    :param goa_humanDB: a data frame with 'GO_ID' field
    :return: a set of the DB's GO ids
    """

    return set(goa_humanDB['GO_ID'].values)

def create_child_2_parent_dict(go_ids, goDB, save=None):
    """
    create a dictionary of child and parent information. the parents are recursively set.
    :param go_ids: a set of GO ids we should model
    :param goDB: a parse obo object, dictionary.
    :param save: a path to save the dictionary to as a binary file. important due to long running time
                of the dictionary construction. default None
    :return: a dictionary with child GO id as a key and a set of corresponding parents GO ids as received by
            the GO DB. each parent should appear in the go_ids list
    """

    if go_ids == None or goDB == None:
        raise Exception('Invalid input for function create_child_2_parent_dict, received None object\n')

    child_2_parent = dict()

    # iterate over the GO ids and add parents to dictionary
    for go_id in tqdm(go_ids):
        GO_term = goDB[go_id]
        parents_from_go = GO_term.get_all_parents()
        parents = set([p for p in parents_from_go if p in go_ids])

        # add to dictionary
        assert(go_id not in child_2_parent)
        child_2_parent[go_id] = parents

    # save as pickle if needed
    if save != None:
        with open(save, 'wb') as o:
            pickle.dump(child_2_parent, o)

    return child_2_parent

def create_child_2_parent_file(child_2_parent_dict, output_dir=None):
    """
    create a data frame with children and their parents.
    :param child_2_parent_dict: a dictionary which for each go id holds a set of parents go ids
    :param output_dir: output directory to save the file in
    :return: a data frame with CHILD and PARENT columns
    """

    if child_2_parent_dict == None:
        raise Exception("Invalid input for function create_child_2_parent_file, received a None dictionary object\n")

    children = []
    parents = []

    for child in child_2_parent_dict:
        children.extend(repeat(child, len(child_2_parent_dict[child])))
        parents.extend(list(child_2_parent_dict[child]))

        if len(children) != len(parents):
            raise Exception("Child and parent dimensions do not fit. {} children and {} parents\n"\
                            .format(len(children),len(parents)))

    # create the new data frame
    child_2_parent = pd.DataFrame({'CHILD': children, 'PARENT':parents})

    # save to file if needed
    if output_dir != None:
        child_2_parent.to_csv(os.path.join(output_dir, 'child_2_parent.txt'), sep='\t', index=False)

    return child_2_parent


def create_basic_gene_2_term_dict(gene_and_terms):
    """
    create a dictionary of gene and its corresponding term in GO based on the goa human db
    :param gene_and_terms: data frame containing gene and its corresponding term
    :return: a dictionary with GO terms as keys and the gene name as value
    """
    if gene_and_terms is None:
        raise Exception("Invalid input for function create_child_2_parent_file, received a None dictionary object\n")

    gene_2_term = dict()

    for term in tqdm(set(gene_and_terms['GO_ID'].values)):
        genes = list(set(gene_and_terms[gene_and_terms['GO_ID'] == term]['DB_Object_Symbol'].values))
        if term not in gene_2_term:
            gene_2_term[term] = genes
        else:
            raise Exception("Create gene 2 term dictionary failed. term is already in dict\n")

    return gene_2_term

def create_gene_2_term_dict_by_all_parents(child_2_parent, gene_2_term):
    """
    create a dictionary of gene 2 term which maps all possible terms to an existing GO annotation.
    every optional parent may have 1-n related terms
    :param child_2_parent: a dictionary with a child as a key and a list of all parents as a value
    :param gene_2_term: a dictionary of all GO ids in goa human DB as keys and all related genes as a value
    :return: a dictionary with the child's GO id and a list of all connected terms of all parents
    """

    rec_gene_2_term = dict()

    for child in tqdm(child_2_parent):
        parents = child_2_parent[child]
        # init with the genes related to the child
        related_genes = gene_2_term[child]
        for parent in parents:
            related_genes.extend(gene_2_term[parent])

        # look at the set of all genes since the list may overlap.
        related_genes = list(set(related_genes))
        # update the result dictionary
        rec_gene_2_term[child] = related_genes

    return rec_gene_2_term

def create_gene_2_term_file(gene_2_term_dict, output_dir=None):
    """
    create a data frame with GO ID and its gene names.
    :param gene_2_term_dict: a dictionary which for each go id holds a set of parents go ids
    :param output_dir: output directory to save the file in
    :return: a data frame with CHILD and PARENT columns
    """

    if gene_2_term_dict == None:
        raise Exception("Invalid input for function create_gene_2_term_file, received a None dictionary object\n")

    terms = []
    genes = []

    for term in tqdm(gene_2_term_dict):
        terms.extend(repeat(term, len(gene_2_term_dict[term])))
        genes.extend(list(gene_2_term_dict[term]))

        if len(terms) != len(genes):
            raise Exception("Terms and genes dimensions do not fit. {} terms and {} genes\n"\
                            .format(len(terms),len(genes)))

    # create the new data frame
    gene_2_term = pd.DataFrame({'TERM': terms, 'GENE':genes})

    # save to file if needed
    if output_dir != None:
        gene_2_term.to_csv(os.path.join(output_dir, 'gene_2_term.txt'), sep='\t', index=False)

    return gene_2_term

def write_ontotype_input_log_file(gene_2_term, child_2_parent, save=None, cycles=0):
    """
    this function creates a file with some information about the constructed input files
    :param save: optional. a path to save the log file, if nor defines then it will be on the lical directory
    :param gene_2_terms: a data frame of gene 2 term
    :param child_2_parent: a data frame of child to parent
    :param cycles: an indicator for cycles in the graph. 0 - has no cycles, 1 o.w
    :return: write the running information to a file
    """

    if gene_2_term is None or child_2_parent is None:
        raise Exception("Invalid inputs, recieved a None data frame object\n")

    with open("ontotype_inputs_log.txt", 'w') as o:
        o.write("--- Constructing ontotype inputs ---\n")
        o.write("{}\n\n".format(datetime.datetime.now()))
        o.write("Created child_2_parent and gene_2_term inpout file\n")
        o.write("--------------------------------------------------\n")
        o.write("CHILD TO PARENT input file contains {} GO IDS\n".format(child_2_parent.shape[0]))
        o.write("Number of unique children {}".format(set(child_2_parent['CHILD'].values)))
        o.write("GENE TO TERM input file contains {} GO IDS\n".format(gene_2_term.shape[0]))
        o.write("Number of unique terms {}".format(set(gene_2_term['TERM'].values)))

        if cycles == 0:
            o.write("No cycles detected\n")
        else:
            o.write("Cycle has been detected! child_2_parent dictionary is not valid\n")
    print("Finished writing ontotype's input log")


def test_for_cycles(child_2_parent, parsed_obo, save=None, nodes=None):
    """
    create a graph of child to parent relations and check that there are no cycles in it
    :param child_2_parent: a child to parent data frame
    :param parsed_obo: a dictionary with
    :param save: a path to save the nods list to as a pickle file. optional, default is none
    :param nodes: a path to a nodes file if saved should be in a binary format (pickle). optional.
    :return: 0 if the graph has no cycles 1 otherwise
    """

    if child_2_parent is None:
        raise Exception("Invalid child 2 parent input file. cannot test for cycles in the graph\n")

    # define a graph object
    g = nx.DiGraph()
    graph_nodes = []

    if nodes == None:
        # need to construct the list of nodes.

        for i in child_2_parent.index:
            # add only direct parents to the graph - an edge will connect a child and its direct parent
            graph_nodes.append((child_2_parent.iloc[i]['CHILD'], child_2_parent.iloc[i]['PARENT']))

        if save != None:
            with open(save, 'wb') as o:
                pickle.dump(graph_nodes, o)

    else:   # read nodes data from pickle
        with open(nodes, 'rb') as o:
            graph_nodes = pickle.load(o)

    # now we have a nodes list
    g.add_edges_from(nodes)

    # test for cycles
    cycles = nx.find_cycle(g)

    if cycles == []: # empty list means no cycles
        return 0


    return 1


