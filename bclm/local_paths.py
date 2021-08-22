import os


BCLM_FOLDER = os.path.dirname(os.path.realpath(__file__))
DATA_FOLDER = os.path.join(BCLM_FOLDER, 'data')
YAP_OUT_FOLDER = os.path.join(DATA_FOLDER, 'yap_outputs')

TREEBANK_TOKEN_PATHS = {
                'train': os.path.join(YAP_OUT_FOLDER, 'spmrl_train_tokens.txt'),
                'dev': os.path.join(YAP_OUT_FOLDER, 'spmrl_dev_tokens.txt'),
                'test': os.path.join(YAP_OUT_FOLDER, 'spmrl_test_tokens.txt'),
                }

YAP_OUTPUT_PATHS = {
                    'seg': {
                            'train': os.path.join(YAP_OUT_FOLDER, 'spmrl_train_seg.conll'),
                            'dev': os.path.join(YAP_OUT_FOLDER, 'spmrl_dev_seg.conll'),
                            'test': os.path.join(YAP_OUT_FOLDER, 'spmrl_test_seg.conll'),
                    },
                    'map': {
                            'train': os.path.join(YAP_OUT_FOLDER, 'spmrl_train_map.conll'),
                            'dev': os.path.join(YAP_OUT_FOLDER, 'spmrl_dev_map.conll'),
                            'test': os.path.join(YAP_OUT_FOLDER, 'spmrl_test_map.conll'),
                    },
                    'dep': {
                            'train': os.path.join(YAP_OUT_FOLDER, 'spmrl_train_dep.conll'),
                            'dev': os.path.join(YAP_OUT_FOLDER, 'spmrl_dev_dep.conll'),
                            'test': os.path.join(YAP_OUT_FOLDER, 'spmrl_test_dep.conll'),
                    },
                }

LATTICES_PATHS = {
                    'train': os.path.join(YAP_OUT_FOLDER, 'spmrl_train.lattices'),
                    'dev': os.path.join(YAP_OUT_FOLDER, 'spmrl_dev.lattices'),
                    'test': os.path.join(YAP_OUT_FOLDER, 'spmrl_test.lattices'),
                }

DF_PATHS = {
            'spmrl': os.path.join(DATA_FOLDER, 'spdf_fixed.csv.gz'),
            'ud': os.path.join(DATA_FOLDER, 'uddf_fixed.csv.gz'),
            'yap_dev': os.path.join(YAP_OUT_FOLDER, 'yap_dev.csv.gz'),
            'yap_test': os.path.join(YAP_OUT_FOLDER, 'yap_test.csv.gz'),
           }