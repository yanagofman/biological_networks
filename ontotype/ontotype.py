import pickle
import random
from collections import defaultdict

LOWER_BOUND = 3
UPPER_BOUND = 10


def get_non_duplicating_list(input_list):
    output = list()
    for x in input_list:
        if x not in output:
            output.append(x)
    return output


class Ontotype(object):
    def __init__(self, initialization_map_file_name):
        self.initialization_map_file_name = initialization_map_file_name
        self.initialization_map = self.load_initialization_map()

    def get_distinct_list_of_protein_ids(self):
        distinct_go_ids = {go_id for go_ids in self.initialization_map.values() for go_id in go_ids}
        return sorted(distinct_go_ids)

    def get_go_list_for_gene_id(self, gene_id):
        return self.initialization_map.get(gene_id, list())

    def create_go_data_map(self, gene_id_list):
        gene_id_non_duplications_list = get_non_duplicating_list(gene_id_list)
        go_data_map = defaultdict(int)
        for gene_id in gene_id_non_duplications_list:
            go_list_for_gene = self.get_go_list_for_gene_id(gene_id)
            for go_id in go_list_for_gene:
                go_data_map[go_id] += 1
        return go_data_map

    def load_initialization_map(self):
        with open(self.initialization_map_file_name, 'rb') as f:
            return pickle.load(f)
    
    def randomize_map(self):
        values = list(self.initialization_map.values())
        keys = self.initialization_map.keys()
        random.shuffle(values)
        self.initialization_map = dict(zip(keys, values))


def main():
    ontotype = Ontotype("./initialization_map.pkl")
    init_map = ontotype.initialization_map.copy()
    ontotype.randomize_map()
    print(ontotype.initialization_map == init_map)


if __name__ == '__main__':
    main()
