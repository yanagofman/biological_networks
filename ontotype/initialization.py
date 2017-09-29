from ontotype.ontotype_classes import Go
import pickle
import csv


def parse_line(line):
    row = line.split(",")
    gene_id = row[0]
    go_number = row[1]
    go_name = row[2]
    return gene_id, go_number, go_name


def is_line_valid(line):
    row = line.split(",")
    if len(row) == 3:
        gene_id = row[0]
        go_number = row[1]
        go_name = row[2]
        if gene_id != '' and gene_id != '\n' and go_number != '' and go_number != '\n' and go_name != '' \
                and go_name != '\n':
            return True
        else:
            return False
    else:
        return False


def parse_go_name(go_name):
    if go_name.endswith('\n'):
        return go_name[0:len(go_name) - 1]
    return go_name


def create_initialization_map(go_data_file):
    initialization_map = dict()
    first_line = True
    for line in open(go_data_file):
        if first_line:
            first_line = False
            continue
        if is_line_valid(line):
            (gene_id, goNumber, go_name) = parse_line(line)
            go_name = parse_go_name(go_name)
            go = Go(goNumber, go_name)
            if gene_id in initialization_map:
                if go not in initialization_map[gene_id]:
                    initialization_map[gene_id].append(go)
            else:
                initialization_map[gene_id] = [go]
    file_name = "initialization_map.pkl"
    with open(file_name, 'wb') as initialization_map_file_obj:
        pickle.dump(initialization_map, initialization_map_file_obj)
    print('Done')


# addition
def get_all_genes(file_name):
    result = set()
    with open(file_name, "r") as f:
        reader = csv.reader(f, delimiter=',')
        next(reader)
        for row in reader:
            result.add(row[0])
    return result


if __name__ == '__main__':
    create_initialization_map("mart_export.txt")
