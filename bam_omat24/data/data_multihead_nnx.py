"""Multihead data loading for multi-dataset fine-tuning.

Extends the base Dataset to inject a head index into each graph's globals,
enabling multihead RACE training with multiple datasets.
"""

import pickle
import numpy as np
import jraph
from flax import nnx

from bam_omat24.data.data_nnx import Dataset


class DatasetWithHead(Dataset):
    """Dataset that injects a head index into each graph's globals.

    Each graph gets globals["head"] = np.array([head_idx], dtype=np.int32),
    which is automatically stacked by jraph.batch_np during collation.

    Args:
        file_path: Path to pickle file containing graphs.
        head_idx: Integer head index to assign to all graphs in this dataset.
        process_id: Current process/node ID (0 to n_processes-1).
        n_processes: Total number of processes/nodes.
    """

    def __init__(
        self,
        file_path: str = None,
        head_idx: int = 0,
        process_id: int = 0,
        n_processes: int = 1,
    ):
        super().__init__(
            file_path=file_path,
            process_id=process_id,
            n_processes=n_processes,
        )
        self.head_idx = head_idx

        # Inject head index into each graph's globals
        for i, g in enumerate(self.graphs):
            new_globals = dict(g.globals)
            new_globals["head"] = np.array([head_idx], dtype=np.int32)
            self.graphs[i] = g._replace(globals=new_globals)
