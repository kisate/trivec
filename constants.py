PARAMS = {'batch_size': 72000,
          'embed_dim': 100,
          'epoch': 8,
          'learning_rate': 1e-2,
          'regularization': 0,
          'num_of_neg_samples': 6,
          'seed': 42}

DATA_CONST = {'work_dir': 'data',
              'drug_train': "/polyphar_train_modified.csv",
              'drug_val': "/polyphar_validate.csv",
              'drug_test': "/polyphar_test.csv",
              'ent_maps': "/ent_maps.csv",
              'rel_maps': "/rel_maps.csv",
              'ppi': "/ppi_data.csv",
              'targets': "/targets_data.csv",
              'save_path': "/trivec_saved/"}

KG_CONST = {'column_names': ['from', 'rel', 'to']}
