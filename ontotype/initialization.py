from collections import defaultdict

from ontotype.ontotype_classes import Go
import pickle
import csv

LOWER_THRESHOLD = 50
UPPER_THRESHOLD = 500


def get_relevant_proteins(go_data_file):
    go_to_count = defaultdict(int)
    with open(go_data_file, "r") as f:
        reader = csv.reader(f)
        next(reader)
        for row in reader:
            _, go_number, _ = row
            go_to_count[go_number] += 1
    return {k for k, v in go_to_count.items() if LOWER_THRESHOLD <= v <= UPPER_THRESHOLD}


def create_initialization_map(go_data_file):
    initialization_map = defaultdict(list)
    relevant_gos = get_relevant_proteins(go_data_file)
    with open(go_data_file, "r") as f:
        reader = csv.reader(f)
        next(reader)
        for row in reader:
            gene_id, go_number, go_name = row
            if go_number not in relevant_gos:
                continue
            go = Go(go_number, go_name)
            if go not in initialization_map[gene_id]:
                initialization_map[gene_id].append(go)
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
    create_initialization_map("mart_export.csv")
