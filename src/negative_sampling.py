import math
import scipy.sparse as sp
from typing import List, Optional, Tuple
import numpy as np
import gc
import torch
import torch.utils.data as data
from src.utils import create_rel2idxs


class NegativeSampler:
    def __init__(self):
        pass

    def __call__(self, triplets: torch.Tensor):
        raise NotImplementedError


class NegDataset(data.Dataset):
    def __init__(self, triplets: torch.Tensor, neg_sampler: NegativeSampler):
        self.triplets = triplets
        self.neg_sampler = neg_sampler
        self.neg_triplets = neg_sampler(triplets)

    def __getitem__(self, index):
        return self.neg_triplets[index]

    def __len__(self):
        return len(self.neg_triplets)

    def regenerate(self):
        self.neg_triplets = self.neg_sampler(self.triplets)


class HonestNegativeSampler(NegativeSampler):
    """
    Class for negative sampling. Guaranties that negative samples would not
    contain any positive samples.
    """

    def __init__(self, num_subjects: int, num_objects: int,
                 all_triplets: np.array, neg_per_pos: int):
        super().__init__()
        self.num_subjects = num_subjects
        self.num_objects = num_objects
        self.all_triplets = all_triplets
        self.neg_per_pos = neg_per_pos

        # Relation to corresponding row idxs in self.all_triplets
        self.rel2idxs = create_rel2idxs(all_triplets)

        self.rel2adj = {}
        for rel, idxs in self.rel2idxs.items():
            adj_values = np.ones(len(idxs))
            adj_coords = all_triplets[idxs, 0], all_triplets[idxs, 2]
            adj_shape = (self.num_objects, self.num_subjects)
            self.rel2adj[rel] = sp.csr_matrix((adj_values, adj_coords),
                                              adj_shape)

    @staticmethod
    def _sample_from_zeros(n: int, sparse: sp.csr_matrix) -> List[List[int]]:
        """
        Sample n zeros from sparse matrix.
        Parameters
        ----------
        n : int
            Number of samples to get from matrix.
        sparse : sp.csr_matrix
            Sparse matrix.
        Returns
        -------
        List[List[int]]
            List of 2-D indices of zeros.
        """
        zeros = np.argwhere(np.logical_not(sparse.todense()))
        ids = np.random.choice(range(len(zeros)), size=(n,))
        return zeros[ids].tolist()

    @staticmethod
    def _sample_by_row(num_of_iters_y: int, sparse: sp.csr_matrix,
                       part_of_zero_i: List[float], submatrix_size: int,
                       n_of_samples: int, start_idx: int,
                       end_idx: Optional[int] = None
                       ) -> list:
        """
        Sample zeros from submatrix of sparse of kind: sparse[start_idx:end_idx].
        Parameters
        ----------
        num_of_iters_y : int
        sparse : sp.csr_matrix
            Sparse matrix.
        part_of_zero_i : List[float]
            Part on n samples to get from current part of matrix.
        submatrix_size : int
            Size of submatrix (height and width).
        n_of_samples : int
            Samples to get from matrix.
        start_idx : int
            Start index of submatrix by x.
        end_idx : Optional[int]
            End index of submatrix by x.
        Returns
        -------
        list
            List of samples.
        """
        to_return = []
        for j in range(num_of_iters_y):
            to_sample = math.ceil(n_of_samples * (part_of_zero_i[j]))
            submat = sparse[start_idx:end_idx,
                     j * submatrix_size:(j + 1) * submatrix_size]
            ids_in_submat = HonestNegativeSampler._sample_from_zeros(to_sample,
                                                                     submat)
            ids_in_mat = ids_in_submat + \
                         np.array([start_idx, j * submatrix_size])
            to_return.extend(ids_in_mat)
        j = num_of_iters_y
        if j * submatrix_size < sparse.shape[1]:
            to_sample = math.ceil(n_of_samples * (part_of_zero_i[j]))
            submat = sparse[start_idx:end_idx,
                     j * submatrix_size:]
            ids_in_submat = HonestNegativeSampler._sample_from_zeros(to_sample,
                                                                     submat)
            ids_in_mat = ids_in_submat + \
                         np.array([start_idx, j * submatrix_size])
            to_return.extend(ids_in_mat)
        return to_return

    @staticmethod
    def _get_number_of_zeros_by_row(sparse: sp.csr_matrix,
                                    num_of_iters_y: int,
                                    submatrix_size: int,
                                    elements_in_submatrix: int,
                                    start_idx: int,
                                    end_idx: Optional[int] = None
                                    ) -> List[float]:
        """
        Get number of zeros in submatrix of sparse of kind: sparse[start_idx:end_idx].
        Parameters
        ----------
        sparse : sp.csr_matrix
            Sparse matrix.
        num_of_iters_y : int
        submatrix_size : int
            Size of submatrix (height and width).
        elements_in_submatrix : int
            Number of elements in submatrix.
        start_idx : int
            Start index of submatrix by x.
        end_idx : Optional[int]
            End index of submatrix by x.
        Returns
        -------
        List[float]
            List of number of zeros in each submatrix.
        """
        tmp = []
        for j in range(num_of_iters_y):
            tmp.append(1 - sparse[start_idx:end_idx,
                           j * submatrix_size:(
                                                      j + 1) * submatrix_size].count_nonzero()
                       / elements_in_submatrix)
        j = num_of_iters_y
        if j * submatrix_size < sparse.shape[1]:
            sub_mtr = sparse[start_idx:end_idx, j * submatrix_size:]
            tmp.append(
                1 - sub_mtr.count_nonzero() / (
                        sub_mtr.shape[0] * sub_mtr.shape[1]))
        return tmp

    @staticmethod
    def _not_exist_edges(sparse: sp.csr_matrix, n_of_samples: int,
                         submatrix_size: int = 1000
                         ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Perform negative sampling.
        Parameters
        ----------
        sparse : sp.csr_matrix
            Sparse matrix.
        n_of_samples : int
            Number os samples to get.
        submatrix_size : int
            Size of submatrix (height and width).
        Returns
        -------
        np.ndarray, np.ndarray
            Negative samples (array of begin nodes and array of end nodes).
        """
        num_of_iters_x = sparse.shape[0] // submatrix_size
        num_of_iters_y = sparse.shape[1] // submatrix_size

        # count nonzero elements on each submatrix
        elements_in_submatrix = submatrix_size ** 2
        part_of_zero = []
        for i in range(num_of_iters_x):
            part_of_zero.append(
                HonestNegativeSampler._get_number_of_zeros_by_row(
                    sparse,
                    num_of_iters_y,
                    submatrix_size,
                    elements_in_submatrix,
                    i * submatrix_size,
                    (i + 1) * submatrix_size))
        i = num_of_iters_x
        if num_of_iters_x * submatrix_size < sparse.shape[0]:
            part_of_zero.append(
                HonestNegativeSampler._get_number_of_zeros_by_row(
                    sparse,
                    num_of_iters_y,
                    submatrix_size,
                    elements_in_submatrix,
                    i * submatrix_size))

        norm = sum([sum(i) for i in part_of_zero])
        part_of_zero = [[i / norm for i in lst] for lst in part_of_zero]
        result = []
        for i in range(num_of_iters_x):
            print(f"Progress: {i}/{num_of_iters_x}")
            result.extend(HonestNegativeSampler._sample_by_row(
                num_of_iters_y, sparse, part_of_zero[i],
                submatrix_size, n_of_samples,
                i * submatrix_size, (i + 1) * submatrix_size))
            gc.collect()
        if num_of_iters_x * submatrix_size < sparse.shape[0]:
            result.extend(
                HonestNegativeSampler._sample_by_row(
                    num_of_iters_y, sparse, part_of_zero[i],
                    submatrix_size, n_of_samples,
                    num_of_iters_x * submatrix_size))
        np.random.shuffle(result)
        result = np.vstack(result[:n_of_samples])
        return result[:, 0], result[:, 1]

    def negative_by_type(self, num_of_neg_samples: int, rel: int) \
            -> torch.Tensor:
        """
        Get negative samples for given relative type.
        Guaranteed no positive examples.

        Parameters
        ----------
        kg : KnowledgeGraph
        data_type : str
            Train, val or test.

        Returns
        -------
        torch.Tensor
            Negative samples.
        """

        neg_samples = HonestNegativeSampler._not_exist_edges(
            sparse=self.rel2adj[rel],
            n_of_samples=num_of_neg_samples)

        neg_samp = torch.zeros((num_of_neg_samples, 3))
        neg_samp[:, 0] = torch.tensor(neg_samples[0])
        neg_samp[:, 1] = torch.full((num_of_neg_samples,), rel)
        neg_samp[:, 2] = torch.tensor(neg_samples[1])
        return neg_samp

    def __call__(self, triplets: torch.Tensor):
        rel2idxs = create_rel2idxs(np.array(triplets))
        neg_triplets = []
        for rel, idxs in rel2idxs.items():
            neg_triplets.append(
                self.negative_by_type(self.neg_per_pos * len(idxs), rel))
        return torch.cat(neg_triplets, axis=0)


class RandomNegativeSampler(NegativeSampler):
    """
    Class for negative sampling. Gives no guarantee that negative samples could
    not contain positive one.
    """

    def __init__(self, num_subjects, num_objects, neg_per_pos: int):
        super().__init__()
        self.num_subjects = num_subjects
        self.num_objects = num_objects
        self.neg_per_pos = neg_per_pos

    def __call__(self, triplets: torch.Tensor):
        # Change triple head/tail to random node
        corrupt_subjects_num = self.neg_per_pos // 2
        corrupt_objects_num = self.neg_per_pos - corrupt_subjects_num

        corrupt_objects = torch.randint(
            low=0, high=self.num_objects,
            size=(len(triplets) * corrupt_objects_num, 1))
        corrupt_object_tripls = torch.cat(
            [
                triplets[:, :2].repeat(corrupt_objects_num, 1),
                corrupt_objects.float()
            ],
            axis=1
        )

        corrupt_subjects = torch.randint(
            low=0, high=self.num_subjects,
            size=(len(triplets) * corrupt_subjects_num, 1))
        corrupt_subject_tripls = torch.cat(
            [
                corrupt_subjects.float(),
                triplets[:, 1:].repeat(corrupt_subjects_num, 1)
            ],
            axis=1)

        return torch.cat([corrupt_object_tripls, corrupt_subject_tripls])
