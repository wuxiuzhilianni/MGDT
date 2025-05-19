from mmdet.datasets import DATASETS, ConcatDataset, build_dataset

@DATASETS.register_module()
class SemiDataset(ConcatDataset):
    """Wrapper for semisupervised od."""

    def __init__(self, sup: dict, unsup: dict, **kwargs):
        super().__init__([build_dataset(sup), build_dataset(unsup)], **kwargs)

    @property
    def sup(self):
        return self.datasets[0]

    @property
    def unsup(self):
        return self.datasets[1]

@DATASETS.register_module()
class SparseDataset(ConcatDataset):
    """Wrapper for semisupervised od."""

    def __init__(self, sup: dict, unsup_unlabeled: dict, unsup_labeled: dict, **kwargs):
        super().__init__([build_dataset(sup), build_dataset(unsup_unlabeled), build_dataset(unsup_labeled)], **kwargs)

    @property
    def sup(self):
        return self.datasets[0]

    @property
    def unsup_unlabeled(self):
        return self.datasets[1]
    
    @property
    def unsup_labeled(self):
        return self.datasets[2]

@DATASETS.register_module()
class SparseDatasetV2(ConcatDataset):
    """Wrapper for semisupervised od."""

    def __init__(self, sup_sparsely: dict, sup_weakly: dict, unsup: dict, **kwargs):
        super().__init__([build_dataset(sup_sparsely), build_dataset(sup_weakly), build_dataset(unsup)], **kwargs)

    @property
    def sup_sparsely(self):
        return self.datasets[0]

    @property
    def sup_weakly(self):
        return self.datasets[1]
    
    @property
    def unsup(self):
        return self.datasets[2]