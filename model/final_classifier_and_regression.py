# Code should be given  at least 2 command line arguments and up to 3
# First argument - type of prediction:
# clf - for RandomForestClassifier
# reg - for RandomForestRegressor
# Second argument - Path to the csv containing the ontotype's results and the sampels' scores
# Third argument(optional) - if you want to use the elimination of gene, write 1

# After building a classifier/regressor, it will start a loop, in which the user should enter a list of defected gene
#  ids, separated by commas The code will run the ontotype on the list, and return the prediction of the
# classifier/regressor on the result

# To end the loop, enter a hashtag instead of gene ids list

from sys import *

from model.data_test import *
from ontotype.ontotype import Ontotype

Ontotype = Ontotype("./ontotype/initialization_map.pkl")


class Classifier(object):
    def __init__(self, data_file_name, genes_list=None):
        self.dt, self.lb = load_data_clf(data_file_name, genes=genes_list)
        self.clf = RandomForestClassifier(n_jobs=-1)
        self.clf.fit(self.dt, self.lb)
        self.onto = Ontotype
        self.ids = Ontotype.get_distinct_list_of_protein_ids()

    def predict(self, gene_id_list):
        onto_vec = np.zeros(len(self.ids))
        onto_map = Ontotype.create_go_data_map(gene_id_list)
        for key in onto_map.keys():
            onto_vec[self.ids.index(key)] = onto_map[key]
        return self.clf.predict(onto_vec.reshape(1, -1))


class Regressor(object):
    def __init__(self, data_file_name, genes_list=None):
        self.dt, self.lb = load_data(data_file_name, genes=genes_list)
        self.clf = RandomForestRegressor(n_jobs=-1)
        self.clf.fit(self.dt, self.lb)
        self.onto = Ontotype
        self.ids = Ontotype.get_distinct_list_of_protein_ids()

    def predict(self, gene_id_list):
        onto_vec = np.zeros(len(self.ids))
        onto_map = Ontotype.create_go_data_map(gene_id_list)
        for key in onto_map.keys():
            onto_vec[self.ids.index(key)] = onto_map[key]
        return self.clf.predict(onto_vec.reshape(1, -1))


def main():
    if len(argv) < 3:
        print(
            "Not enough arguements given!\nFirst arg should be: clf|reg\nSecond arg should be the name of the data "
            "file\naOptional 3rd arg: use alimination of genes(print 1 if wanted)")
        return
    t = str.lower(argv[1])
    file_name = argv[2]
    genes = None
    if len(argv) > 3 and argv[3] == '1':
        genes = load_genes('genes_validation\genes_data_after_second_elimination.csv')
    if t == 'clf':
        rf = Classifier(file_name, genes_list=genes)
    elif t == 'reg':
        rf = Regressor(file_name, gene_list=genes)
    else:
        print("Invalid Arguements!")
        return
    print('To finish, enter #')
    while True:
        lst = input("Please enter a list of defected genes id, seperated by ',': ").split(',')
        if lst[0] == '#':
            break
        for i in range(len(lst)):
            lst[i] = lst[i].strip()
        try:
            res = rf.predict(lst)
            print(rf.__class__.__name__, 'predicted', res)
        except Error as e:
            print("Something went wrong, maybe invalid gene_id_list!!!")


if __name__ == '__main__':
    main()
