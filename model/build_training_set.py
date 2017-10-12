import copy
import csv
import json
import os
from datetime import datetime


from fetch_data.gene_entries import GeneEntries
from ontotype.ontotype_classes import Ontotype
from strings import CELL_LINE, DEFECTIVE_GENE, IS_ALIVE
from utils import similar, clean_up

PRINT_THRESHOLD = 10
WRITE_THRESHOLD = 1000


def get_file_name_for_param(training_set_path, param, is_folder=False):
    file_components = os.path.split(training_set_path)
    file_basename = file_components[-1]
    file_folder = os.path.join(*file_components[:-1])
    if is_folder:
        new_folder = param
        new_full_folder = os.path.join(file_folder, new_folder)
        if not os.path.exists(new_full_folder):
            os.makedirs(new_full_folder)
        return os.path.join(file_folder, new_folder, file_basename)
    else:
        basename_prefix, basename_suffix = file_basename.split('.')
        new_filename = "%s_%s.%s" % (basename_prefix, param, basename_suffix)
        return os.path.join(file_folder, new_filename)


class BuildTrainingSet(object):
    def __init__(self, initialization_map_file_name, gene_name_to_id_path, should_shuffle):
        self.ontotype_obj = Ontotype(initialization_map_file_name)
        if should_shuffle:
            self.ontotype_obj.randomize_map()
        self.gene_entries_fetcher = GeneEntries(gene_name_to_id_path)

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

    def get_results_headers(self, single_csv):
        distinct_proteins = self.ontotype_obj.get_distinct_list_of_protein_ids()
        headers = [DEFECTIVE_GENE, IS_ALIVE] + distinct_proteins
        if single_csv:
            headers.insert(0, CELL_LINE)
        return headers

    @staticmethod
    def fill_initial_results(writer, cell_lines_data_map, cell_lines, single_csv):
        base_dict = {DEFECTIVE_GENE: None, IS_ALIVE: 1}
        for cell_line in cell_lines:
            if single_csv:
                base_dict[CELL_LINE] = cell_line
            writer.writerow({**base_dict,
                             **cell_lines_data_map[cell_line]})

    def fill_defective_genes_results(self, writer, cell_lines_data_map, cell_lines, reader, single_csv):
        now = datetime.now()
        results = list()
        for j, row in enumerate(reader):
            gene = row[0]
            gene_id = self.gene_entries_fetcher.get_single_gene_id(gene)
            if gene_id:
                go_map = self.ontotype_obj.create_go_data_map([gene_id])
                for i, cell_line in enumerate(cell_lines):
                    base_dict = {DEFECTIVE_GENE: gene, IS_ALIVE: row[i + 1]}
                    if single_csv:
                        base_dict[CELL_LINE] = cell_line
                    merged_go_map = self.merge_two_go_data_maps(cell_lines_data_map[cell_line], go_map)
                    merged_go_map.update(base_dict)
                    results.append(merged_go_map)
                if j and (j + 1) % PRINT_THRESHOLD == 0:
                    print("Done iterating on %s defective genes after %s seconds" % (j + 1,
                                                                                     (datetime.now() - now)
                                                                                     .total_seconds()))
            if j and j % WRITE_THRESHOLD == 0:
                writer.writerows(results)
                results = list()
        if results:
            writer.writerows(results)

    def handle_the_whole_cell_lines_in_single_csv(self, training_set_path, results_headers, cell_lines_data_map,
                                                  cell_lines, reader, single_csv):
        with open(training_set_path, "w") as output_file:
            output_writer = csv.DictWriter(output_file, results_headers)
            output_writer.writeheader()
            self.fill_initial_results(output_writer, cell_lines_data_map, cell_lines, single_csv)
            self.fill_defective_genes_results(output_writer, cell_lines_data_map, cell_lines, reader, single_csv)

    def handle_each_cell_line_in_different_csv(self, training_set_path, results_headers, cell_lines_data_map,
                                               cell_lines, reader, reader_file, single_csv):
        for cell_line in cell_lines:
            with open(get_file_name_for_param(training_set_path, cell_line), "w") \
                    as cell_line_output_file:
                output_writer = csv.DictWriter(cell_line_output_file, results_headers)
                output_writer.writeheader()
                self.fill_initial_results(output_writer, cell_lines_data_map, [cell_line], single_csv)
                self.fill_defective_genes_results(output_writer, cell_lines_data_map, [cell_line], reader, single_csv)
            reader_file.seek(1)

    def build_training_set(self, gene_input_path, fetched_data_path, training_set_path, single_csv):
        results_headers = self.get_results_headers(single_csv)
        with open(gene_input_path, "r") as f:
            reader = csv.reader(f)
            cell_lines_data_map, cell_lines = self.initialize_cell_lines_data_go_map(reader, fetched_data_path)
            if single_csv:
                self.handle_the_whole_cell_lines_in_single_csv(training_set_path, results_headers, cell_lines_data_map,
                                                               cell_lines, reader, single_csv)
            else:
                self.handle_each_cell_line_in_different_csv(training_set_path, results_headers, cell_lines_data_map,
                                                            cell_lines, reader, f, single_csv)


def main(gene_input_path, fetched_data_path, initialization_map_file_name, training_set_path, gene_name_to_id_path,
         single_csv=False, number_of_iterations=11):
    for i in range(number_of_iterations):
        should_shuffle = bool(i)
        new_training_set_path = get_file_name_for_param(training_set_path, "shuffle_%s" % i if i else "no_shuffle",
                                                        is_folder=True)
        build_training_set_obj = BuildTrainingSet(initialization_map_file_name, gene_name_to_id_path, should_shuffle)
        build_training_set_obj.build_training_set(gene_input_path, fetched_data_path, new_training_set_path, single_csv)

if __name__ == '__main__':
    main("./genes_input.csv", "../fetch_data/data.json", "../ontotype/initialization_map.pkl",
         "./training_set_files/training_set.csv", "../fetch_data/gene_name_to_id_mapping.json")
