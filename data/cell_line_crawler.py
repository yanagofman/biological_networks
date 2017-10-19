import json
import math
import re

from selenium import webdriver
from selenium.common.exceptions import TimeoutException
from selenium.webdriver.common.by import By
from selenium.webdriver.support import expected_conditions as ec
from selenium.webdriver.support.wait import WebDriverWait

from data.gene_entries import GeneEntries
from utils import similar

NUM_OF_RESULTS_RE = re.compile("\s(\d+)\sentries")

CELL_LINES = ['OCI-AML5', 'TF-1', 'NB-4', 'MOLM-13', 'HEL', 'PL-21',
              'MV-4-11', 'EOL-1', 'OCI-AML2', 'OCI-AML3',
              'P31/FUJ', 'MonoMac1', 'SKM-1', 'THP-1']
MUTATION_TYPES = ['Frame_Shift_Del', 'In_Frame_Del', 'Missense_Mutation', 'Nonsense_Mutation']
GENE = 'Gene'
MUTATION_TYPE = 'Variant Classification'
DEFAULT_NUM_OF_PAGE_RESULTS = 25

cached_genes = dict()


class CellLineCrawler(object):
    def __init__(self):
        self.session = webdriver.Firefox()
        self.session.maximize_window()
        self.login()
        self.gene_entries = GeneEntries()

    def wait_for(self, xpath):
        timeout = 7
        try:
            element_present = ec.presence_of_element_located((By.XPATH, xpath))
            WebDriverWait(self.session, timeout).until(element_present)
        except TimeoutException as e:
            print("Couldn't find results: %s" % e)
            return False
        return True

    def get_homepage(self):
        self.session.get("https://portals.broadinstitute.org/ccle_legacy/home")

    def login(self):
        self.get_homepage()
        login_xpath = ".//div[@id='logIn']"
        if self.wait_for(login_xpath):
            login_div = self.session.find_element_by_xpath(login_xpath)
            login_username = login_div.find_element_by_xpath(".//input[@type='text']")
            login_username.send_keys("yanagofman")
            login_password = login_div.find_element_by_xpath(".//input[@type='password']")
            login_password.send_keys("wpuf8gPk")
            submit_button = login_div.find_element_by_xpath(".//input[contains(@src, 'sign_in')]")
            submit_button.click()

    def handle_single_gene(self, row, header_to_index_dict):
        row_columns = row.find_elements_by_xpath(".//td")
        mutation_type = row_columns[header_to_index_dict[MUTATION_TYPE]]
        if mutation_type.text in MUTATION_TYPES:
            gene = row_columns[header_to_index_dict[GENE]]
            link = gene.find_element_by_xpath(".//a")
            gene_name = link.text
            entry_id = self.gene_entries.get_single_gene_id(gene_name)
            return entry_id

    def read_single_page_mutations_table(self):
        mutations_xpath = ".//tr[contains(@class, 'mutation')]"
        if self.wait_for(mutations_xpath):
            mutations = list()
            mutations_headers = self.session.find_elements_by_xpath(".//th")
            header_to_index_dict = {header.text.replace("\n", " "): i for i, header in enumerate(mutations_headers)}
            rows = self.session.find_elements_by_xpath(mutations_xpath)
            for row in rows:
                entry_id = self.handle_single_gene(row, header_to_index_dict)
                if entry_id:
                    mutations.append(entry_id)
            return mutations

    def get_paginate_buttons(self):
        pagination_xpath = ".//div[@id='sampleMutationDetailTable1_paginate']"
        if self.wait_for(pagination_xpath):
            paginate = self.session.find_element_by_xpath(pagination_xpath)
            span = paginate.find_element_by_xpath(".//span")
            paginate_buttons = span.find_elements_by_xpath(".//a")
            return paginate_buttons

    def get_i_index_paginate_button(self, i):
        paginate_buttons = self.get_paginate_buttons()
        if paginate_buttons:
            return [paginate_button for paginate_button in paginate_buttons if paginate_button.text == str(i + 1)][0]

    def get_paginate_buttons_length(self):
        table_xpath = ".//div[@id='sampleMutationDetailTable1_info']"
        if self.wait_for(table_xpath):
            results_info = self.session.find_element_by_xpath(table_xpath)
            content = results_info.text
            num_of_results = int(NUM_OF_RESULTS_RE.search(content).group(1))
            print("Num of results: %s" % num_of_results)
            return math.ceil(num_of_results / DEFAULT_NUM_OF_PAGE_RESULTS)

    def read_multiple_pages_mutations_table(self):
        num_of_pages = self.get_paginate_buttons_length()
        mutations = []
        if num_of_pages:
            mutations.extend(self.read_single_page_mutations_table())
            for i in range(1, num_of_pages):
                paginate_button = self.get_i_index_paginate_button(i)
                paginate_button.click()
                mutations.extend(self.read_single_page_mutations_table())
        print("Total num of mutations: %s" % len(mutations))
        return mutations

    def click_on_cell_line_tab(self):
        tabs_xpath = ".//td[contains(@class, 'rich-tab-header')]"
        if self.wait_for(tabs_xpath):
            tabs = self.session.find_elements_by_xpath(tabs_xpath)
            for tab in tabs:
                if tab.text.startswith("Cell") and tab.get_attribute("class").strip().endswith("inactive"):
                    tab.click()

    def click_on_best_result(self, cell_line):
        self.click_on_cell_line_tab()
        table_result_xpath = ".//table[@class='tableOnSearchPage']"
        if self.wait_for(table_result_xpath):
            table = self.session.find_elements_by_xpath(table_result_xpath)[-1]
            results = table.find_elements_by_xpath(".//tr[@class='tableOnSearchPage']")
            options = list()
            for result in results:
                relevant_cell = result.find_element_by_xpath(".//td[@class='tableOnSearchPageLeft']")
                link = relevant_cell.find_element_by_xpath(".//a")
                options.append((link, similar(link.text.lower(), cell_line.lower())))
            max(options, key=lambda x: x[1])[0].click()

    def search_for_cell_line(self, cell_line):
        search_box_xpath = ".//input[@id='searchForm:searchText']"
        if self.wait_for(search_box_xpath):
            search_box = self.session.find_element_by_xpath(search_box_xpath)
            search_box.send_keys(cell_line)
            search_button = self.session.find_element_by_xpath(".//input[@id='searchForm:searchPortal']")
            search_button.click()

    def read_single_cell_line_mutations(self, cell_line):
        self.search_for_cell_line(cell_line)
        self.click_on_best_result(cell_line)
        return self.read_multiple_pages_mutations_table()

    def read_whole_mutations_data(self):
        results = dict()
        # Reading the mutations for each cell line
        for i, cell_line in enumerate(CELL_LINES):
            print("Tying to fetch %s(%s)" % (cell_line, i))
            if i:
                self.get_homepage()
            results[cell_line] = self.read_single_cell_line_mutations(cell_line)
        return results


def main():
    b = CellLineCrawler()
    results = b.read_whole_mutations_data()
    with open("./data.json", "w") as f:
        json.dump(results, f)


if __name__ == '__main__':
    main()
