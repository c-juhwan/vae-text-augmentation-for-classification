# Standard Library Modules
import pickle
# 3rd-party Modules
from tqdm.auto import tqdm
# Pytorch Modules
from torch.utils.data.dataset import Dataset

class AugmentationDataset(Dataset):
    def __init__(self, data_path:str) -> None:
        super(AugmentationDataset, self).__init__()
        with open(data_path, 'rb') as f:
            data_ = pickle.load(f)

        self.data_list = []
        self.vocabulary = data_['Vocab']
        for idx in tqdm(range(len(data_['Text'])), desc=f'Loading data from {data_path}'):
            self.data_list.append({
                'Text_Tensor': data_['Text'][idx],
                'Label_Tensor': data_['Label'][idx]
            })

        del data_

    def __getitem__(self, index:int) -> dict:
        return self.data_list[index]

    def __len__(self) -> int:
        return len(self.data_list)
