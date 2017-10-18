from collections import defaultdict

import pickle
import csv

LOWER_THRESHOLD = 50
UPPER_THRESHOLD = 500


def get_go_id_out_if_go_number(go_number):
    return int(go_number.split(':')[-1]) if go_number else None


def get_parent_child_mappings(go_to_parent_file_path):
    parent_to_children = defaultdict(list)
    child_to_parents = defaultdict(list)
    with open(go_to_parent_file_path, "r") as f:
        reader = csv.reader(f)
        for row in reader:
            child_go_number, parent_go_number = row
            child_go_id = get_go_id_out_if_go_number(child_go_number)
            parent_go_id = get_go_id_out_if_go_number(parent_go_number)
            parent_to_children[parent_go_id].append(child_go_id)
            child_to_parents[child_go_id].append(parent_go_id)
    return parent_to_children, child_to_parents


def recursive_get_relevant_gos(parent_child_map, go_id, results, go_to_genes=None):
    if not parent_child_map[go_id]:
        return
    for current_go_id in parent_child_map[go_id]:
        results.append(go_to_genes[current_go_id] if go_to_genes else current_go_id)
        recursive_get_relevant_gos(parent_child_map, current_go_id, results, go_to_genes)


def get_relevant_proteins(go_data_file_path, parent_to_children):
    go_to_genes = defaultdict(set)
    with open(go_data_file_path, "r") as f:
        reader = csv.reader(f)
        next(reader)
        for row in reader:
            gene_id, go_number, _ = row
            go_id = get_go_id_out_if_go_number(go_number)
            if go_id:
                go_to_genes[go_id].add(gene_id)

    relevant_proteins = set()
    for go_id, genes in go_to_genes.copy().items():
        list_of_children_genes = list()
        recursive_get_relevant_gos(parent_to_children, go_id, list_of_children_genes, go_to_genes=go_to_genes)
        if LOWER_THRESHOLD <= \
                len(set.union(genes, *list_of_children_genes)) <= UPPER_THRESHOLD:
            relevant_proteins.add(go_id)
    return relevant_proteins


def create_initialization_map(go_data_file, child_to_parent_file_path):
    parent_to_children, child_to_parents = get_parent_child_mappings(child_to_parent_file_path)
    initialization_map = defaultdict(list)
    relevant_gos = get_relevant_proteins(go_data_file, parent_to_children)
    with open(go_data_file, "r") as f:
        reader = csv.reader(f)
        next(reader)
        for row in reader:
            gene_id, go_number, _ = row
            go_id = get_go_id_out_if_go_number(go_number)
            if not go_id or go_id not in relevant_gos:
                continue
            if go_id not in initialization_map[gene_id]:
                initialization_map[gene_id].append(go_id)
                recursive_get_relevant_gos(child_to_parents, go_id, initialization_map[gene_id])
                initialization_map[gene_id] = list(set(initialization_map[gene_id]))

    file_name = "initialization_map.pkl"
    with open(file_name, 'wb') as initialization_map_file_obj:
        pickle.dump(initialization_map, initialization_map_file_obj)


# addition
def get_all_genes(file_name):
    result = set()
    with open(file_name, "r") as f:
        reader = csv.reader(f, delimiter=',')
        next(reader)
        for row in reader:
            result.add(row[0])
    return result


if __name__ == '__main__':
    create_initialization_map("gene_to_go.csv", "go_to_parent.csv")
