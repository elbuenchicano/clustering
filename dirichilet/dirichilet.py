
from utils          import *
from interface      import MainControler
from bupclustering  import *
   
################################################################################
################################################################################

################################################################################
################################################################################
class Controler(MainControler):
    def __init__(self, func_dict, json_file = 'conf.json'):
        return super().__init__(func_dict, json_file)

    #+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    def postProcess(self, collected):
        pass

################################################################################
################################################################################
############################### MAIN ###########################################
if __name__ == '__main__':
    main_functions = {
                        'clustering_bottom_up'  : clusteringBottomUp,
                        'tsne'                  : tsne
                     } 
    control =  Controler(main_functions, 'dirichilet_conf.json')
    control.run()