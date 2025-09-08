import typing as t
import random
import os
import numpy as np
import torch
import torch_geometric
import torch.nn.functional as F
from torch_geometric.data import Data, Dataset
from rdkit import Chem
from rdkit.Chem.rdchem import BondType as BT
from .abstract_dataset import AbstractDatasetInfos
from .abstract_dataset import AbstractDataModule

bonds = {"friends": 0}


class FO2DataModule(AbstractDataModule):
    def __init__(
        self,
        cfg,
        auto_train_ratio: float = 0.8,
        file_name: str = None,
        train_path: str = None,
        gen_file: str = None,
    ):
        super().__init__(cfg=cfg)
        assert (
            file_name or train_path
        ), "Either file_name or both train_path must be provided"
        self.file_name = file_name
        self.train_path = train_path
        self.gen_file = gen_file
        self.auto_train_ratio = auto_train_ratio

    def prepare_data(self, _=None) -> None:
        train_file = (
            self.train_path
            or f"/home/jungpete/projects/deepgraphon/data-fo2/{self.file_name}/{self.file_name}.train.txt"
        )

        if self.file_name:
            val_file = f"/home/jungpete/projects/deepgraphon/data-fo2/{self.file_name}/{self.file_name}.val.txt"
            if not os.path.exists(val_file):
                val_file = f"/home/jungpete/projects/deepgraphon/data-fo2/{self.file_name}/{self.file_name}.test.txt"
        else:
            val_file = None

        train_fols: t.List[t.Tuple[str, ...]] = []
        with open(train_file) as f:
            for line in f:
                parsed_line = eval(line)
                train_fols.append(tuple(parsed_line))
                if int(self.cfg.train.n_train_data) > 0 and len(train_fols) >= int(
                    self.cfg.train.n_train_data
                ):
                    break
        fols_deduplicated_train = sorted(set(train_fols))
        print(
            f"Train number of samples: {len(train_fols)}, deduplicated: {len(fols_deduplicated_train)}"
        )
        random.shuffle(fols_deduplicated_train)

        val_fols: t.List[t.Tuple[str, ...]] = []
        if val_file is None:
            split_index = int(self.auto_train_ratio * len(fols_deduplicated_train))
            val_fols = fols_deduplicated_train[split_index:]
            fols_deduplicated_train = fols_deduplicated_train[:split_index]
        else:
            with open(val_file) as f:
                for line in f:
                    parsed_line = eval(line)
                    val_fols.append(tuple(parsed_line))
        fold_deduplicated_val = sorted(set(val_fols))
        print(
            f"Val number of samples: {len(val_fols)}, deduplicated: {len(fold_deduplicated_val)}"
        )

        gen_fols: t.List[t.Tuple[str, ...]] = []
        if self.gen_file:
            with open(self.gen_file) as f:
                n_lines = sum(1 for line in f)
            with open(self.gen_file) as f:
                for idx_line, line in enumerate(f):
                    # Uncomment if only last 10k samples are wanted.
                    # if idx_line < n_lines - 10000:
                    #     continue
                    splited = line.replace(", ", ",").split(" ")
                    parsed_line = eval(splited[2] if len(splited) >= 3 else splited[0])
                    gen_fols.append(tuple(parsed_line))
        fold_deduplicated_gen = list(set(gen_fols))
        random.shuffle(fold_deduplicated_gen)

        # Calculate the sizes available
        gen_size = len(fold_deduplicated_gen)
        train_size = len(fols_deduplicated_train)
        val_size = len(fold_deduplicated_val)

        # Adjust sizes to ensure gen_train will always be the same size as train_fols, and same for val
        adjusted_train_size = min(train_size, int(gen_size * self.auto_train_ratio))
        adjusted_val_size = min(val_size, gen_size - adjusted_train_size)

        # Adjust datasets accordingly without repeating data
        fold_deduplicated_gen_train = fold_deduplicated_gen[:adjusted_train_size]
        fols_deduplicated_train = fols_deduplicated_train[:adjusted_train_size]

        fold_deduplicated_gen_val = fold_deduplicated_gen[
            adjusted_train_size : adjusted_train_size + adjusted_val_size
        ]
        fold_deduplicated_val = fold_deduplicated_val[:adjusted_val_size]

        assert fold_deduplicated_gen_train, f"{len(fold_deduplicated_gen_train)=}"
        assert fold_deduplicated_gen_val, f"{len(fold_deduplicated_gen_val)=}"

        print(
            f"Training real samples: {len(fols_deduplicated_train)}, generated samples: {len(fold_deduplicated_gen_train)}"
        )
        print(
            f"Validation real samples: {len(fold_deduplicated_val)}, generated samples: {len(fold_deduplicated_gen_val)}"
        )

        all_dataset = FO2Dataset(
            fols_deduplicated_train + fold_deduplicated_val + gen_fols, "all"
        )
        train_dataset = FO2Dataset(
            fols_deduplicated_train,
            "train",
            all_dataset.types,
            all_dataset.constant_to_idx,
            all_dataset.edge_to_idx,
        )
        val_dataset = FO2Dataset(
            fold_deduplicated_val,
            "val",
            train_dataset.types,
            train_dataset.constant_to_idx,
            train_dataset.edge_to_idx,
        )
        gen_train_dataset = FO2Dataset(
            fold_deduplicated_gen_train,
            "gen",
            train_dataset.types,
            train_dataset.constant_to_idx,
            train_dataset.edge_to_idx,
        )
        gen_val_dataset = FO2Dataset(
            fold_deduplicated_gen_val,
            "gen",
            train_dataset.types,
            train_dataset.constant_to_idx,
            train_dataset.edge_to_idx,
        )

        train_and_gen_dataset = torch.utils.data.ConcatDataset(
            [train_dataset, gen_train_dataset]
        )
        test_and_gen_dataset = torch.utils.data.ConcatDataset(
            [val_dataset, gen_val_dataset]
        )

        self.types = all_dataset.types
        self.idx_to_type = all_dataset.idx_to_type
        self.constant_to_idx = all_dataset.constant_to_idx
        self.idx_to_constant = all_dataset.idx_to_constant
        self.edge_to_idx = all_dataset.edge_to_idx
        self.idx_to_edge = all_dataset.idx_to_edge

        super().prepare_data(
            datasets={
                "_train": train_dataset,
                "_val": val_dataset,
                "_test": val_dataset,
                "train_and_gen": train_and_gen_dataset,
                "test_and_gen": test_and_gen_dataset,
            }
        )

    def to_sentence(self, nodes, adj_matrix) -> str:
        sentence = []

        constant_to_type = {}
        for idx, node_type in enumerate(nodes):
            constant = self.idx_to_constant[idx]
            type_ = self.idx_to_type[node_type.item()]
            if type_ != "NONE":
                sentence.append(f"{type_}({constant})")
            constant_to_type[constant] = type_

        for from_idx, to_idx in zip(*torch.nonzero(adj_matrix, as_tuple=True)):
            fulledge = self.idx_to_edge[adj_matrix[from_idx, to_idx].item()]
            subedges = fulledge.split(";")
            for edge in subedges:
                from_constant = self.idx_to_constant[from_idx.item()]
                to_constant = self.idx_to_constant[to_idx.item()]
                sentence.append(f"{edge}({from_constant},{to_constant})")

        sentence = sorted(sentence)
        return "[" + ",".join(f"'{x}'" for x in sentence) + "]"


class FO2Dataset(Dataset):
    def __init__(
        self,
        objects: t.List[str],
        split: str,
        types={},
        constant_to_idx={},
        edge_to_idx={},
    ) -> None:
        super().__init__()
        self.objects = objects
        self.split = split
        self.types = types
        self.idx_to_type = {v: k for k, v in self.types.items()}
        self.constant_to_idx = constant_to_idx
        self.idx_to_constant = {v: k for k, v in self.constant_to_idx.items()}
        self.edge_to_idx = edge_to_idx
        self.idx_to_edge = {v: k for k, v in self.edge_to_idx.items()}
        self.data = self.process_objects(self.objects)

    def len(self) -> int:
        return len(self.data)

    def get(self, idx: int) -> Data:
        return self.data[idx]

    def process_objects(self, sentences: t.List[t.List[str]]) -> t.List[Data]:
        data_list = []

        if self.split == "all":
            self.types["NONE"] = self.types.get("NONE", len(self.types))
            self.edge_to_idx["NONE"] = self.edge_to_idx.get(
                "NONE", len(self.edge_to_idx)
            )
            for i, sentence in enumerate(sentences):
                for predicate in sentence:
                    inside_of_parenthetes = predicate.split("(")[1].rstrip(")")
                    constants = inside_of_parenthetes.split(",")

                    for constant in constants:
                        self.constant_to_idx[constant] = self.constant_to_idx.get(
                            constant, len(self.constant_to_idx)
                        )

                    if predicate.count(",") == 0:
                        predicate_symbol = predicate.split("(")[0]
                        self.types[predicate_symbol] = self.types.get(
                            predicate_symbol, len(self.types)
                        )

                    elif predicate.count(",") == 1:
                        edge = predicate.split("(")[0]
                        self.edge_to_idx[edge] = self.edge_to_idx.get(
                            edge, len(self.edge_to_idx)
                        )

                    else:
                        raise RuntimeError(predicate)

        self.idx_to_constant = {v: k for k, v in self.constant_to_idx.items()}
        self.idx_to_type = {v: k for k, v in self.types.items()}
        self.idx_to_edge = {v: k for k, v in self.edge_to_idx.items()}

        for i, sentence in enumerate(sentences):
            adj_matrix = torch.zeros(
                (len(self.idx_to_constant), len(self.idx_to_constant))
            )
            edge_indexes_to_edge_types = {}

            constant_to_type = {}

            edge_type_to_adj_matrix = {
                e: torch.zeros((len(self.idx_to_constant), len(self.idx_to_constant)))
                for e in self.edge_to_idx.keys()
                if e != "NONE"
            }

            for predicate in sentence:
                if predicate.count(",") == 0:
                    predicate_symbol = predicate.split("(")[0]
                    constant = predicate.split("(")[1].rstrip(")")
                    constant_to_type[constant] = self.types[predicate_symbol]

            for predicate in sentence:
                if predicate.count(",") == 0:
                    continue

                assert predicate.count(",") == 1, predicate

                constants = predicate.split("(")[1].rstrip(")").split(",")
                from_, to_ = constants
                from_idx, to_idx = (
                    self.constant_to_idx[from_],
                    self.constant_to_idx[to_],
                )

                edge = predicate.split("(")[0]

                adj_matrix[from_idx, to_idx] = 1
                adj_matrix[to_idx, from_idx] = 1

                edge_type_to_adj_matrix[edge][from_idx, to_idx] = 1
                edge_type_to_adj_matrix[edge][to_idx, from_idx] = 1

                if (from_idx, to_idx) not in edge_indexes_to_edge_types:
                    edge_indexes_to_edge_types[(from_idx, to_idx)] = []
                if (to_idx, from_idx) not in edge_indexes_to_edge_types:
                    edge_indexes_to_edge_types[(to_idx, from_idx)] = []

                if edge not in edge_indexes_to_edge_types[(from_idx, to_idx)]:
                    edge_indexes_to_edge_types[(from_idx, to_idx)].append(edge)
                if edge not in edge_indexes_to_edge_types[(to_idx, from_idx)]:
                    edge_indexes_to_edge_types[(to_idx, from_idx)].append(edge)

            if all(len(x) == 1 for x in edge_indexes_to_edge_types.values()):
                pass
            elif all(len(x) in (1, 2) for x in edge_indexes_to_edge_types.values()):
                first_edge_with_two_types = [
                    x for x in edge_indexes_to_edge_types.values() if len(x) == 2
                ][0]
                new_edge = ";".join(sorted([x for x in first_edge_with_two_types]))
                if new_edge not in self.edge_to_idx:
                    new_edge_idx = len(self.edge_to_idx)
                    self.edge_to_idx[new_edge] = new_edge_idx
                    self.idx_to_edge[new_edge_idx] = new_edge
            else:
                raise RuntimeError(edge_indexes_to_edge_types)

            x = F.one_hot(
                torch.tensor(
                    [
                        constant_to_type.get(
                            self.idx_to_constant[idx], self.types["NONE"]
                        )
                        for idx in range(len(self.constant_to_idx))
                    ]
                ),
                num_classes=len(self.types),
            ).float()
            y = torch.zeros([1, 0]).float()
            edge_index, _ = torch_geometric.utils.dense_to_sparse(adj_matrix)
            edge_attr = torch.zeros(
                edge_index.shape[-1], len(self.edge_to_idx), dtype=torch.float
            )
            for this_edge_index, (from_, to_) in enumerate(zip(*edge_index)):
                edge_types = edge_indexes_to_edge_types[(from_.item(), to_.item())]
                stringified_edge_types = ";".join(sorted(edge_types))
                edge_attr[this_edge_index][self.edge_to_idx[stringified_edge_types]] = 1

            n = adj_matrix.shape[-1]
            num_nodes = n * torch.ones(1, dtype=torch.long)
            data = torch_geometric.data.Data(
                x=x, edge_index=edge_index, edge_attr=edge_attr, y=y, n_nodes=num_nodes
            )
            # Unsqueeze to add batch dimension so that it's concated properly by dataloader.
            data.node_feats = x.unsqueeze(0)
            data.adj_matrix = adj_matrix.unsqueeze(0)
            data.edge_type_to_adj_matrix = {
                k: v.unsqueeze(0) for k, v in edge_type_to_adj_matrix.items()
            }
            data.gan_y = torch.tensor(1.0 if self.split == "gen" else 0.0)
            data_list.append(data)

        return data_list


class FO2DatasetInfos(AbstractDatasetInfos):
    def __init__(self, datamodule, dataset_config):
        self.datamodule = datamodule
        self.name = "nx_graphs"
        self.n_nodes = self.datamodule.node_counts()
        self.node_types = self.datamodule.node_types()
        self.edge_types = self.datamodule.edge_counts()
        super().complete_infos(self.n_nodes, self.node_types)
