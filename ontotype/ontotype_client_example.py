from ontotype_classes import Ontotype


ontotypt = Ontotype() #call once. loading this pickle takes time...

goDataMap1 = ontotypt.createGoDataMap(['ENSG00000283951', 'ENSG00000283882', 'ENSG00000284150'])

ontotypt.printGoDataMap(goDataMap1)

goDataMap2 = ontotypt.createGoDataMap(['ENSG00000284244'])

ontotypt.printGoDataMap(goDataMap2)

#geneIdList = getAllGenes('mart_export.txt')



