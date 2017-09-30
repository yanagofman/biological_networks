import pickle

from collections import defaultdict


class Go(object):
    def __init__(self, go_number, go_name):
        self.go_number = go_number
        self.go_name = go_name

    def __eq__(self, other):
        return self.go_number == other.go_number

    def get_go_id(self):
        return int(self.go_number.split(':')[-1])


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
        distinct_go_ids = {go_obj.get_go_id() for go_objs in self.initialization_map.values() for go_obj in go_objs}
        return sorted(distinct_go_ids)

    def get_go_list_for_gene_id(self, gene_id):
        return self.initialization_map.get(gene_id)

    def create_go_data_map(self, gene_id_list):
        gene_id_non_duplications_list = get_non_duplicating_list(gene_id_list)
        go_data_map = defaultdict(int)
        for gene_id in gene_id_non_duplications_list:
            if gene_id not in self.initialization_map:
                continue
            go_list_for_gene = self.get_go_list_for_gene_id(gene_id)
            for go in go_list_for_gene:
                go_data_map[int(go.get_go_id())] += 1
        return go_data_map

    def load_initialization_map(self):
        with open(self.initialization_map_file_name, 'rb') as f:
            return pickle.load(f)


def main():
    ontotype = Ontotype("./initialization_map.pkl")
    proteins = ontotype.get_distinct_list_of_protein_ids()
    print(proteins[:10])


if __name__ == '__main__':
    main()
