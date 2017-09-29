import copy
import csv
import json
import pickle

import numpy as np
import pandas as pd

from fetch_data.gene_entries import GeneEntries
from ontotype.ontotype_classes import Ontotype
from utils import similar, clean_up

PRINT_THRESHOLD = 10


class BuildTrainingSet(object):
    def __init__(self, initialization_map_file_name):
        self.ontotype_obj = Ontotype(initialization_map_file_name)
        self.gene_entries_fetcher = GeneEntries()

    @staticmethod
    def fetch_cell_lines(reader):
        headers = next(reader)  # fetching the headers row
        cell_lines = headers[1:]
        return cell_lines

    @staticmethod
    def read_fetched_cell_lines_data(fetched_data_path):
        with open(fetched_data_path, "r") as f:
            return json.load(f)

    @staticmethod
    def get_fetched_data_cell_lines_mapping(fetched_cell_lines, input_cell_lines):
        mapping = dict()
        max_similarity = 0
        for input_cell_line in input_cell_lines:
            cleaned_input_cell_line = clean_up(input_cell_line)
            for fetched_cell_line in fetched_cell_lines:
                current_similarity = similar(clean_up(fetched_cell_line), cleaned_input_cell_line)
                if current_similarity > max_similarity:
                    mapping[input_cell_line] = fetched_cell_line
                    max_similarity = current_similarity
            max_similarity = 0
        return mapping

    def fetch_single_cell_line_result(self, fetched_data, cell_line):
        genes = fetched_data[cell_line]
        go_data_map = self.ontotype_obj.create_go_data_map(genes)
        return go_data_map

    def fetch_cell_lines_core_genes(self, fetched_data, mapping, cell_lines):
        cell_lines_data_map = dict()
        for cell_line in cell_lines:
            go_data_map = self.fetch_single_cell_line_result(fetched_data, mapping[cell_line])
            cell_lines_data_map[cell_line] = go_data_map
        return cell_lines_data_map

    @staticmethod
    def merge_two_go_data_maps(go_data_1, go_data_2):
        merged_go_data = copy.deepcopy(go_data_1)
        for go_id, related_genes in go_data_2.items():
            merged_go_data[go_id] += related_genes
        return merged_go_data

    def initialize_cell_lines_data_go_map(self, reader, fetched_data_path):
        cell_lines_data = self.read_fetched_cell_lines_data(fetched_data_path)
        cell_lines = self.fetch_cell_lines(reader)
        cell_lines_mapping = self.get_fetched_data_cell_lines_mapping(cell_lines_data.keys(), cell_lines)
        cell_lines_data_map = self.fetch_cell_lines_core_genes(cell_lines_data, cell_lines_mapping, cell_lines)
        return cell_lines_data_map, cell_lines

    @staticmethod
    def fill_initial_results(results, cell_lines_data_map):
        for cell_line, go_map in cell_lines_data_map.items():
            results.append([cell_line, None, np.array(list(go_map.items())), 1])

    def fill_defective_genes_results(self, results, cell_lines_data_map, cell_lines, reader):
        for j, row in enumerate(reader):
            gene = row[0]
            gene_id = self.gene_entries_fetcher.get_single_gene_id(gene)
            for i, cell_line in enumerate(cell_lines):
                go_map = self.ontotype_obj.create_go_data_map([gene_id])
                merged_go_map = self.merge_two_go_data_maps(cell_lines_data_map[cell_line], go_map)
                results.append([cell_line, gene, np.array(list(merged_go_map.items())), row[i + 1]])
            if j and (j + 1) % PRINT_THRESHOLD == 0:
                print("Done iterating on %s defective genes" % (j + 1))

    def build_training_set(self, csv_path, fetched_data_path):
        results = list()
        with open(csv_path, "r") as f:
            reader = csv.reader(f)
            cell_lines_data_map, cell_lines = self.initialize_cell_lines_data_go_map(reader, fetched_data_path)
            self.fill_initial_results(results, cell_lines_data_map)
            self.fill_defective_genes_results(results, cell_lines_data_map, cell_lines, reader)
        return pd.DataFrame(results)


def save_df_to_pickle(training_set_path, df):
    with open(training_set_path, "w") as f:
        pickle.dump(df, f)


def main(gene_input_path, fetched_data_path, initialization_map_file_name, training_set_path):
    build_training_set_obj = BuildTrainingSet(initialization_map_file_name)
    df = build_training_set_obj.build_training_set(gene_input_path, fetched_data_path)
    save_df_to_pickle(training_set_path, df)

if __name__ == '__main__':
    main("./genes_input.csv", "../fetch_data/data.json", "../ontotype/initialization_map.pkl", "./training_set.pkl")
