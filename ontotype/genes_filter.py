import csv

from scipy.stats import pearsonr
from numpy import var


def get_row_from_input(file_path):
    with open(file_path, "r") as f:
        reader = csv.reader(f)
        headers = next(reader)
        return headers, list(reader)


class GenesFilter(object):
    def __init__(self, genes_input_path):
        self.genes_input_path = genes_input_path
        self.filters = [(self.should_filter_avg_cs_score, self.get_avg_cs_score_threshold),
                        (self.should_filter_second_lowest_cs_score, None),
                        (self.should_filter_pearson, None),
                        (self.should_filter_variance, self.get_variance_threshold)]

    @staticmethod
    def get_parsed_cs_scores(row):
        return list(map(float, row[1:]))

    def get_variance_threshold(self, rows):
        list_of_variances = self.get_list_of_variances(rows)
        return [self.get_top_2000_variance_threshold(list_of_variances)]

    @staticmethod
    def get_list_of_variances(rows):
        list_of_variances = list()
        for row in rows:
            cs_scores = GenesFilter.get_parsed_cs_scores(row)
            list_of_variances.append(var(cs_scores))
        return list_of_variances

    @staticmethod
    def get_top_2000_variance_threshold(list_of_variances):
        list_of_variances = sorted(list_of_variances)
        return list_of_variances[-2000]

    def get_avg_cs_score_threshold(self, rows):
        avg_cs_list = self.get_list_of_avg_cs_scores(rows)
        return [self.get_15_percent_cs_score_threshold(avg_cs_list)]

    @staticmethod
    def get_list_of_avg_cs_scores(rows):
        avg_cs_list = list()
        for row in rows:
            cs_scores = GenesFilter.get_parsed_cs_scores(row)
            avg_cs_list.append(sum(cs_scores) / len(cs_scores))
        return avg_cs_list

    @staticmethod
    def get_15_percent_cs_score_threshold(avg_cs_list):
        avg_cs_list = sorted(avg_cs_list)
        return avg_cs_list[int((15/100.0) * len(avg_cs_list))]

    def filter_genes(self, output_file_path):
        headers, rows = get_row_from_input(self.genes_input_path)
        for filter_method, pre_process_method in self.filters:
            data = []
            if pre_process_method:
                data = pre_process_method(rows)
            filtered_rows = list()
            for row in rows:
                if filter_method(row, *data):
                    continue
                filtered_rows.append(row)
            rows = filtered_rows
        with open(output_file_path, "w") as output_file:
            writer = csv.writer(output_file)
            writer.writerow(headers)
            writer.writerows(rows)

    def should_filter_avg_cs_score(self, row, avg_cs_score_threshold):
        cs_scores = self.get_parsed_cs_scores(row)
        if sum(cs_scores) / len(cs_scores) < avg_cs_score_threshold:
            return True

    @staticmethod
    def should_filter_second_lowest_cs_score(row):
        cs_scores = GenesFilter.get_parsed_cs_scores(row)
        cs_scores = sorted(cs_scores)
        if cs_scores[1] < -1:
            return True

    @staticmethod
    def should_filter_pearson(row):
        cs_scores = GenesFilter.get_parsed_cs_scores(row)
        for i in range(len(cs_scores)):
            identity_vector = [0] * len(cs_scores)
            identity_vector[i] = 1
            if abs(pearsonr(cs_scores, identity_vector)[0]) > 0.8:
                return True

    @staticmethod
    def should_filter_variance(row, variance_threshold):
        cs_scores = GenesFilter.get_parsed_cs_scores(row)
        if var(cs_scores) < variance_threshold:
            return True


def main(genes_input_file_path):
    genes_filter = GenesFilter(genes_input_file_path)
    genes_filter.filter_genes("../model/2000_genes_input.csv")


if __name__ == '__main__':
    main("../model/genes_input.csv")