import ontotype_utils
import pandas as pd
import  argparse

def main(args):

    go_terms_dict = ontotype_utils.create_GO_terms_dict(args.in_obo)
    goahumanDB = pd.read_excel(args.data_file)
    goahuman_go_ids = ontotype_utils.get_human_go_ids(goahumanDB)

    # create dictionaries and data files
    child_2_parent_dict = ontotype_utils.create_child_2_parent_dict(goahuman_go_ids, go_terms_dict)
    child_2_parent = ontotype_utils.create_child_2_parent_file(child_2_parent_dict, args.out_dir)

    # parse ID and gene name out of the goa human DB and use it to construct gene 2 term
    goahumanDB = goahumanDB[['GO_ID', 'DB_Object_Symbol']]
    gene_2_term_dict = ontotype_utils.create_basic_gene_2_term_dict(goahumanDB)

    gene_2_term_dict = ontotype_utils.create_gene_2_term_dict_by_all_parents(child_2_parent_dict, gene_2_term_dict)
    gene_2_term = ontotype_utils.create_gene_2_term_file(gene_2_term_dict, args.out_dir)

    # check that the graph has no cycles
    noCycle = ontotype_utils.test_for_cycles(child_2_parent)
    if noCycle == 1:
        print("WARNING:: Detected cycle in child to parent network\n")

    # write to log file
    ontotype_utils.write_ontotype_input_log_file(gene_2_term, child_2_parent)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--in_obo", type=str, help="a path to an obo file",
                        required=True)
    parser.add_argument("-o", "--out_dir", type=str,
                        help="a path to a directory in which the output files will be saved", default=None)
    parser.add_argument("-d", "--data_file", type=str,
                        help="a path to a human data file containing GO information- goa human", default=None)

    args = parser.parse_args()

    main(args)