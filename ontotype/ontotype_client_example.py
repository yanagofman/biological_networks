from ontotype.ontotype_classes import Ontotype

ontotype_obj = Ontotype()  # call once. loading this pickle takes time...

goDataMap1 = ontotype_obj.create_go_data_map(['ENSG00000283951', 'ENSG00000283882', 'ENSG00000284150'])

ontotype_obj.print_go_data_map(goDataMap1)

goDataMap2 = ontotype_obj.create_go_data_map(['ENSG00000284244'])

ontotype_obj.print_go_data_map(goDataMap2)
