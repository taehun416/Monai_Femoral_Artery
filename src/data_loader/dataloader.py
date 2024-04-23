from monai.data import CacheDataset, DataLoader, load_decathlon_datalist, ThreadDataLoader
import torch.nn as nn

class TrainThreadDataloader(nn.Module):
    def __init__(
        self, 
        split_json,
        train_transforms,
        val_transforms,
        cache_num,
        cache_rate,
        num_workers
    ) -> None:
        super(TrainThreadDataloader, self).__init__()
        # super().__init__()
        self.split_json = split_json
        self.train_transforms = train_transforms
        self.val_transforms = val_transforms
        self.cache_num = cache_num
        self.cache_rate = cache_rate
        self.num_workers = num_workers

    def dataloader(split_json, train_transforms, val_transforms, cache_num, cache_rate, num_workers):
        datalist = load_decathlon_datalist(split_json, True, "training")
        val_files = load_decathlon_datalist(split_json, True, "validation")
        train_ds = CacheDataset(
            data=datalist,
            transform=train_transforms,
            cache_num=cache_num,
            cache_rate=cache_rate,
            num_workers=num_workers,
        )
        train_loader = ThreadDataLoader(train_ds, num_workers=0, batch_size=1, shuffle=True)
        val_ds = CacheDataset(data=val_files, transform=val_transforms, cache_num=9, cache_rate=1.0, num_workers=4)
        val_loader = ThreadDataLoader(val_ds, num_workers=0, batch_size=1)

        return train_ds, train_loader, val_ds, val_loader

class TrainDataloader(nn.Module):
    def __init__(
        self, 
        split_json,
        train_transforms,
        val_transforms,
        cache_num,
        cache_rate,
        num_workers
    ) -> None:
        super(TrainDataloader, self).__init__()
        # super().__init__()
        self.split_json = split_json
        self.train_transforms = train_transforms
        self.val_transforms = val_transforms
        self.cache_num = cache_num
        self.cache_rate = cache_rate
        self.num_workers = num_workers

    def dataloader(split_json, train_transforms, val_transforms, cache_num, cache_rate, num_workers):
        datalist = load_decathlon_datalist(split_json, True, "training")
        val_files = load_decathlon_datalist(split_json, True, "validation")
        train_ds = CacheDataset(
            data=datalist,
            transform=train_transforms,
            cache_num=cache_num,
            cache_rate=cache_rate,
            num_workers=num_workers,
        )
        train_loader = DataLoader(train_ds, num_workers=0, batch_size=1, shuffle=True,pin_memory=True)
        val_ds = CacheDataset(data=val_files, transform=val_transforms, cache_num=9, cache_rate=1.0, num_workers=4)
        val_loader = DataLoader(val_ds, num_workers=0, batch_size=1, pin_memory=True)

        return train_ds, train_loader, val_ds, val_loader
    
# class TrainDataloader(nn.Module):
#     def __init__(
#         self, 
#         train_files,
#         train_transforms,
#         val_files,
#         val_transforms,
#         cache_rate,
#         num_workers
#     ) -> None:
#         super(TrainDataloader, self).__init__()
#         # super().__init__()
#         self.train_files = train_files
#         self.train_transforms = train_transforms
#         self.val_files = val_files
#         self.val_transforms = val_transforms
#         self.cache_rate = cache_rate
#         self.num_workers = num_workers

#     def dataloader(train_files, train_transforms, val_files, val_transforms, cache_rate, num_workers):
#         train_ds = CacheDataset(data=train_files, transform=train_transforms, 
#                                 cache_rate=cache_rate, num_workers=num_workers)
#         train_loader = DataLoader(train_ds, batch_size=2, shuffle=True, num_workers=num_workers)
#         val_ds = CacheDataset(data=val_files, transform=val_transforms, 
#                               cache_rate=cache_rate, num_workers=num_workers)
#         val_loader = DataLoader(val_ds, batch_size=1, num_workers=num_workers)

#         return train_ds, train_loader, val_ds, val_loader
    

class TestThreadDataloader(nn.Module):
    def __init__(
        self, 
        split_json,
        val_transforms,
        cache_num,
        cache_rate
    ) -> None:
        super(TestThreadDataloader, self).__init__()
        # super().__init__()
        self.split_json = split_json
        self.val_transforms = val_transforms
        self.cache_num = cache_num
        self.cache_rate =cache_rate

    def dataloader(split_json, cache_num, cache_rate, val_transforms):
        val_files = load_decathlon_datalist(split_json, True, "test")
        val_ds = CacheDataset(data=val_files, transform=val_transforms, cache_num= cache_num, cache_rate= cache_rate, num_workers=4)
        val_loader = ThreadDataLoader(val_ds, num_workers=0, batch_size=1)

        return val_ds, val_loader
    
class TestDataloader(nn.Module):
    def __init__(
        self, 
        split_json,
        val_transforms,
        cache_num,
        cache_rate
    ) -> None:
        super(TestDataloader, self).__init__()
        # super().__init__()
        self.split_json = split_json
        self.val_transforms = val_transforms
        self.cache_num = cache_num
        self.cache_rate =cache_rate

    def dataloader(split_json, cache_num, cache_rate, val_transforms):
        val_files = load_decathlon_datalist(split_json, True, "test")
        val_ds = CacheDataset(data=val_files, transform=val_transforms, cache_num= cache_num, cache_rate= cache_rate, num_workers=4)
        val_loader = DataLoader(val_ds, num_workers=0, batch_size=1)

        return val_ds, val_loader
    
# class testDataloader(nn.Module):
#     def __init__(
#         self, 
#         val_files,
#         val_transforms,
#         cache_rate,
#         num_workers
#     ) -> None:
#         super(TrainDataloader, self).__init__()
#         # super().__init__()
#         self.val_files = val_files
#         self.val_transforms = val_transforms
#         self.cache_rate = cache_rate
#         self.num_workers = num_workers

#     def dataloader(train_files, val_files, val_transforms, cache_rate, num_workers):
#         val_ds = CacheDataset(data=val_files, transform=val_transforms, 
#                               cache_rate=cache_rate, num_workers=num_workers)
#         val_loader = DataLoader(val_ds, batch_size=1, num_workers=num_workers)

#         return val_ds, val_loader