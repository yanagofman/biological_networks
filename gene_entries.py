import dryscrape


class GeneEntries(object):
    def __init__(self):
        self.session = dryscrape.Session()

    def get_single_gene_id(self, gene_name):
        url = "http://www.ensembl.org/Homo_sapiens/Search/Results?q=%s" \
              ";site=ensembl;facet_species=Human;page=1;facet_feature_type=Gene" % gene_name
        self.session.visit(url)
        self.session.wait_for(lambda: self.session.at_xpath(".//div[@class='table_result']"))
        first_result = self.session.at_xpath(".//div[@class='table_result']")
        id_element = first_result.at_xpath(".//div[@class='green_data']")
        span_element = id_element.at_xpath(".//span")
        return span_element.text()


def main():
    gene_entries = GeneEntries()
    print(gene_entries.get_single_gene_id('SPEN'))

if __name__ == '__main__':
    main()