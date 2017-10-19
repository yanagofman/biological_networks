import json
import sys
import traceback

import dryscrape

NUM_OF_PROCESSES = 4


class GeneEntries(object):
    def __init__(self, gene_name_to_id_path):
        self.session = dryscrape.Session()
        with open(gene_name_to_id_path, "r") as f:
            self.gene_name_to_id_mapping = json.load(f)

    def get_single_gene_id(self, gene_name):
        if gene_name in self.gene_name_to_id_mapping:
            return self.gene_name_to_id_mapping[gene_name]
        url = "http://www.ensembl.org/Homo_sapiens/Search/Results?q=%s" \
              ";site=ensembl;facet_species=Human;page=1;facet_feature_type=Gene" % gene_name
        try:
            self.session.visit(url)
            self.session.wait_for(lambda: self.session.at_xpath(".//div[@class='table_result']"))
        except Exception:
            traceback.print_exc(file=sys.stdout)
            return
        first_result = self.session.at_xpath(".//div[@class='table_result']")
        id_element = first_result.at_xpath(".//div[@class='green_data']")
        span_element = id_element.at_xpath(".//span")
        return span_element.text()


def main():
    gene_entries = GeneEntries("./gene_name_to_id_mapping.json")
    print(gene_entries.get_single_gene_id('SPEN'))

if __name__ == '__main__':
    main()
