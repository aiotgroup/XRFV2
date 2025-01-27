
from torch.utils.data import Dataset
from dataset.wwadl import WWADLDatasetSingle

class WWADLDatasetMuti(Dataset):
    def __init__(self, dataset_dir, split="train"):
        """
        初始化 WWADL 数据集。
        :param dataset_dir: 数据集所在目录路径。
        :param split: 数据集分割，"train" 或 "test"。
        """
        assert split in ["train", "test"], "split must be 'train' or 'test'"

        self.imu_dataset = WWADLDatasetSingle(dataset_dir, split='train', modality='imu')
        self.wifi_dataset = WWADLDatasetSingle(dataset_dir, split='train', modality='wifi')

        self.labels = self.imu_dataset.labels

    def shape(self):
        wifi_shape = self.wifi_dataset.shape()
        imu_shape = self.imu_dataset.shape()
        return wifi_shape[0], f'{wifi_shape}_{imu_shape}'

    def __len__(self):
        """
        返回数据集的样本数。
        """
        return len(self.wifi_dataset)

    def __getitem__(self, idx):

        wifi_data, wifi_label = self.wifi_dataset[idx]
        imu_data, imu_label = self.imu_dataset[idx]

        # print(wifi_label[0], imu_label[0])

        data = {
            'wifi': wifi_data['wifi'],
            'imu': imu_data['imu']
        }

        return data, imu_label


if __name__ == '__main__':
    dataset_dir = '/root/shared-nvme/dataset/all_30_3'
    test_dataset = WWADLDatasetMuti(dataset_dir)
    from torch.utils.data import DataLoader
    from dataset.wwadl import detection_collate
    # 定义 DataLoader
    batch_size = 4
    train_data_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=1,
        collate_fn=detection_collate,
        pin_memory=True,
        drop_last=True
    )
    class_set = {}
    for i, (data_batch, label_batch) in enumerate(train_data_loader):
        # print(f"Batch {i} data shape: {data_batch.shape}")
        # print(f"Batch {i} labels: {len(label_batch)}")
        # for label in label_batch:
        #     for ll in label:
        #         if int(ll[2]) not in class_set:
        #             class_set[int(ll[2])] = 0
        #         class_set[int(ll[2])] += 1
        #         assert ll[2] < 30, "00"
        if i > 50:
            break
        # print(label_batch)
    # class_set = {8: 2176, 26: 2627, 24: 5569, 19: 514, 0: 1452, 15: 528, 22: 1096, 1: 2935, 6: 1595, 17: 813, 11: 904, 10: 707, 23: 549, 7: 2619, 18: 622, 13: 1207, 20: 1727, 12: 487, 14: 819, 9: 601, 5: 1229, 27: 427, 25: 301, 16: 611, 3: 891, 4: 888, 2: 714, 21: 254, 28: 97, 29: 38}
    # print(sorted(class_set.items(), key=lambda x: x[0]))
        # break


