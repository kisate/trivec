from typing import List, Tuple
from pandas import concat, DataFrame, read_csv
from constants import DATA_CONST, KG_CONST
import numpy as np


class KnowledgeGraph:
    """
    Class representing Knowledge graph.
    Preprocess and store dataframes for model evaluation.
    Parameters
    ----------
    data_path : str
        Path to the data folder relative to the working directory.
    use_proteins : bool
        Whether to use protein nodes or not.
    use_proteins_on_validation : bool
        Whether to use protein nodes on validation.
    Attributes
    ----------
    data_path : str
        Path to the data folder relative to the working directory.
    use_proteins : bool
        Whether to use protein nodes or not.
    use_prot_on_val : bool
        Whether to use protein nodes on validation.
    use_reversed_edges : bool
        Whether to add edge reverse duplicate or not (for drug-drug relations).
    ent_maps : DataFrame
        Mapping between STITCH ID of node, int id in data.
        WARNING: contains only drugs!
    rel_maps : DataFrame
        Mapping between CID ID of side effect, int id and name.
    _df_train : DataFrame
        DataFrame with training triples.
    _df_val : DataFrame
        DataFrame with validate triples.
    _df_test : DataFrame
        DataFrame with test triples.
    df_drug : np.array
        np.array with all (train, test, validate) triples.
    size_train : int
        Number of training triples.
    size_val : int
        Number of validate triples.
    size_test : int
        Number of test triples.
    """
    def __init__(self, data_path: str = ".",
                 use_proteins: bool = False,
                 use_proteins_on_validation: bool = False,
                 use_reversed_edges: bool = False):
        if use_proteins_on_validation and not use_proteins:
            raise ValueError("Can use proteins on validation only "
                             "if use_proteins = True")
        self.data_path = data_path
        self.use_proteins = use_proteins
        self.use_prot_on_val = use_proteins_on_validation
        self.use_reversed_edges = use_reversed_edges
        self.ent_maps = None
        self.rel_maps = None
        self._df_train, self._df_val, self._df_test, self.df_drug \
            = self.load_polypharmacy_data()
        self.size_train = len(self._df_train)
        self.size_val = len(self._df_val)
        self.size_test = len(self._df_test)

    def load_polypharmacy_data(self) -> Tuple[DataFrame, DataFrame, DataFrame,
                                              np.array]:
        """
        Load polypharmacy data.
        Returns
        -------
        Tuple[DataFrame, DataFrame, DataFrame, np.array] :
            Train, validate, test dataframes and array with all triplec.
        """

        if self.use_reversed_edges:
            # get train data only with drugs
            df_train = self.get_df_with_reversed_edges(
                read_csv(self.data_path + DATA_CONST['drug_train']))

            # get val data only with drugs
            df_val = self.get_df_with_reversed_edges(
                read_csv(self.data_path + DATA_CONST['drug_val']))

            # get test data only with drugs
            df_test = self.get_df_with_reversed_edges(
                read_csv(self.data_path + DATA_CONST['drug_test']))
        else:
            df_train = read_csv(self.data_path + DATA_CONST['drug_train'])

            df_val = read_csv(self.data_path + DATA_CONST['drug_val'])

            df_test = read_csv(self.data_path + DATA_CONST['drug_test'])

        df_train.reset_index(drop=True, inplace=True)
        df_val.reset_index(drop=True, inplace=True)
        df_test.reset_index(drop=True, inplace=True)

        df_drug = concat([df_train, df_val, df_test]).to_numpy()

        self.ent_maps = read_csv(self.data_path + DATA_CONST['ent_maps'])
        self.rel_maps = read_csv(self.data_path + DATA_CONST['rel_maps'])

        if self.use_proteins:
            # load ppi data
            max_drug_rel_id = self.rel_maps['id_in_data'].max()
            df_ppi_train, df_ppi_val = self.load_protein_data(
                DATA_CONST['ppi'],
                self.use_prot_on_val,
                max_drug_rel_id + 1)

            # load targets
            df_tar_train, df_tar_val = self.load_protein_data(
                DATA_CONST['targets'],
                self.use_prot_on_val, max_drug_rel_id + 2)

            df_train = concat(
                [df_train, df_ppi_train, df_tar_train]).reset_index(drop=True)
            df_val = concat([df_val, df_ppi_val, df_tar_val]).reset_index(
                drop=True)

        return df_train, df_val, df_test, df_drug

    @staticmethod
    def get_df_with_reversed_edges(df: DataFrame) -> DataFrame:
        """
        For dataframe with oriented triples return dataframe with original and
        reversed triples.
        Parameters
        ----------
        df : DataFrame
            Dataframe with oriented triples.
        Returns
        -------
        DataFrame :
            Dataframe with original and reversed triples.
        """
        reversed_data = df.copy()
        cols = reversed_data.columns
        reversed_data = reversed_data[[cols[2], cols[1], cols[0]]]
        reversed_data.columns = cols
        return concat([df, reversed_data]).reset_index(drop=True)

    def load_protein_data(self, csv_file: str, use_protein_in_validation: bool,
                          rel_id: int) -> Tuple[DataFrame, DataFrame]:
        df = read_csv(self.data_path + csv_file)
        cols = df.columns
        df[KG_CONST['column_names'][1]] = [rel_id] * len(df)
        df = df[
            [cols[0], KG_CONST['column_names'][1], cols[1]]]
        df.columns = KG_CONST['column_names']

        if use_protein_in_validation:
            len_train = int(len(df) * 0.8)
            df_train = self.get_df_with_reversed_edges(df.iloc[:len_train])
            df_val = self.get_df_with_reversed_edges(df.iloc[len_train:])
        else:
            df_train = self.get_df_with_reversed_edges(df)
            df_val = df.iloc[:0]

        return df_train, df_val

    def get_num_of_ent(self, data_type: str) -> int:
        """
        Get number of different entities types.
        Parameters
        ----------
        data_type : str
            Train, val or test.
        Returns
        -------
        int :
            Number of different entities types.
        Raises
        ------
        ValueError:
            If data_type not one of train, val or test.
        """
        if data_type == "train":
            df = concat([self._df_train, self._df_val, self._df_test])
            # add 1 to max(...), cause max index is smaller by 1
            return max(np.max(df[KG_CONST['column_names'][0]]),
                       np.max(df[KG_CONST['column_names'][2]])) + 1
        if data_type == "val":
            if self.use_prot_on_val:
                return self.get_num_of_ent("train")
            return len(self.ent_maps)
        if data_type == "test":
            return len(self.ent_maps)
        raise ValueError("Unknown data_type!")

    def get_num_of_rel(self, data_type: str):
        """
        Get number of different relatives types.
        Parameters
        ----------
        data_type : str
            Train, val or test.
        Returns
        -------
        int :
            Number of different relatives types.
        Raises
        ------
        ValueError:
            If data_type not one of train, val or test.
        """
        if data_type == "test":
            return len(self.rel_maps)
        if data_type == "train":
            if self.use_proteins:
                # additional types for prot-prot and drug-prot interaction
                return len(self.rel_maps) + 2
            return len(self.rel_maps)
        if data_type == "val":
            if self.use_prot_on_val:
                return len(self.rel_maps) + 2
            return len(self.rel_maps)
        raise ValueError("Unknown datatype!")

    def get_data_by_type(self, data_type: str):
        """
        Get data by type.
        Parameters
        ----------
        data_type : str
            Train, val or test.
        Returns
        -------
        DataFrame :
            Data by type.
        Raises
        ------
        ValueError:
            If data_type not one of train, val or test.
        """
        if data_type == "train":
            return self._df_train
        if data_type == "val":
            return self._df_val
        if data_type == "test":
            return self._df_test
        raise ValueError("Unknown datatype!")


class EdgesPowerKG(KnowledgeGraph):
    """
    Attributes
    ----------
    thresholds : list
    test_pos_weak, test_pos_strong: Dict[int, pd.DataFrame]
        From threshold to corresponding test edges (weak or strong).
    val_pos_weak, val_pos_strong: Dict[int, pd.DataFrame]
        From threshold to corresponding val edges (weak or strong).
        Only drug-drug edges!

    See more in docs for KnowledgeGraph

    """

    def __init__(self, data_path, use_proteins,
                 use_proteins_on_validation, thresholds,
                 use_reversed_edges: bool):
        super().__init__(data_path, use_proteins,
                         use_proteins_on_validation, use_reversed_edges)
        self.thresholds = thresholds
        print('Make Strong and Weak edges dataframes')
        self._df_test_weak, self._df_test_strong = self._get_weak_strong_edges(
            self._df_train, self._df_test, thresholds)
        self._df_val_weak, self._df_val_strong = self._get_weak_strong_edges(
            self._df_train, self._df_val, thresholds)
        print('Done!')

    def _calc_deg(self, triplets_df: DataFrame):
        N_drugs = self.get_num_of_ent(data_type='test')
        degs = np.zeros((N_drugs,))
        for i in range(len(degs)):
            degs[i] += (triplets_df[KG_CONST['column_names'][0]] == i).sum()
            degs[i] += (triplets_df[KG_CONST['column_names'][2]] == i).sum()
        return degs

    def _get_weak_strong_edges(self,
                               triplets_df_train: DataFrame,
                               triplets_df_test: DataFrame,
                               thresholds=None):
        if thresholds is None:
            thresholds = [100]
        N_drugs = self.get_num_of_ent(data_type='test')
        degs_train = self._calc_deg(triplets_df_train)
        degs_test = self._calc_deg(triplets_df_test)

        # Leave only non-isolated in test graph edges
        drugs_with_test_edges = [drug for drug in range(N_drugs) if
                                 degs_test[drug] > 0]

        drug_order = sorted(drugs_with_test_edges, key=lambda i: degs_train[i])

        drugs_weak = {}
        drugs_strong = {}
        for thr in thresholds:
            drugs_weak[thr] = drug_order[:thr]
            drugs_strong[thr] = drug_order[-thr:]

        tripl_test_weak = {}
        tripl_test_strong = {}
        for thr in thresholds:
            tripl_test_weak[thr] = triplets_df_test[np.logical_or(
                triplets_df_test[KG_CONST['column_names'][0]].isin(
                    drugs_weak[thr]),
                triplets_df_test[KG_CONST['column_names'][2]].isin(
                    drugs_weak[thr]))]

            tripl_test_strong[thr] = triplets_df_test[np.logical_or(
                triplets_df_test[KG_CONST['column_names'][0]].isin(
                    drugs_strong[thr]),
                triplets_df_test[KG_CONST['column_names'][2]].isin(
                    drugs_strong[thr]))]

        return tripl_test_weak, tripl_test_strong

    def get_data_by_type(self, data_type: str):
        if data_type == "train":
            return self._df_train
        if data_type == "val":
            return self._df_val
        if data_type == "test":
            return self._df_test
        if data_type == 'test_weak':
            return self._df_test_weak
        if data_type == 'test_strong':
            return self._df_test_strong
        if data_type == 'val_weak':
            return self._df_val_weak
        if data_type == 'val_strong':
            return self._df_val_strong
        raise ValueError("Unknown datatype!")
