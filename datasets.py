from glob import glob
import os
import pandas as pd
from PIL import Image
import torch
from torchvision import transforms as T
from torch.utils.data import random_split
import numpy as np
from torch.utils.data import random_split
from torch.utils.data import Dataset
from torch.utils.data import DataLoader

class FrameImageDataset(torch.utils.data.Dataset):
    def __init__(self,
                 root_dir='/work3/ppar/data/ucf101',
                 split='train',
                 transform=None
                 ):
        self.frame_paths = sorted(glob(f'{root_dir}/frames/{split}/*/*/*.jpg'))
        self.df = pd.read_csv(f'{root_dir}/metadata/{split}.csv')
        self.split = split
        self.transform = transform

    def __len__(self):
        return len(self.frame_paths)

    def _get_meta(self, attr, value):
        return self.df.loc[self.df[attr] == value]

    def __getitem__(self, idx):
        frame_path = self.frame_paths[idx]
        video_name = frame_path.split('/')[-2]
        video_meta = self._get_meta('video_name', video_name)
        label = video_meta['label'].item()

        frame = Image.open(frame_path).convert("RGB")

        if self.transform:
            frame = self.transform(frame)
        else:
            frame = T.ToTensor()(frame)

        return frame, label


class FrameVideoDataset(torch.utils.data.Dataset):
    def __init__(self,
                 root_dir='/work3/ppar/data/ucf101',
                 split='train',
                 transform=None,
                 stack_frames=True
                 ):

        self.video_paths = sorted(glob(f'{root_dir}/videos/{split}/*/*.avi'))
        self.df = pd.read_csv(f'{root_dir}/metadata/{split}.csv')
        self.split = split
        self.transform = transform
        self.stack_frames = stack_frames

        self.n_sampled_frames = 10

    def __len__(self):
        return len(self.video_paths)

    def _get_meta(self, attr, value):
        return self.df.loc[self.df[attr] == value]

    def __getitem__(self, idx):
        video_path = self.video_paths[idx]
        video_name = video_path.split('/')[-1].split('.avi')[0]
        video_meta = self._get_meta('video_name', video_name)
        label = video_meta['label'].item()

        video_frames_dir = self.video_paths[idx].split('.avi')[0].replace('videos', 'frames')
        video_frames = self.load_frames(video_frames_dir)

        if self.transform:
            frames = [self.transform(frame) for frame in video_frames]
        else:
            frames = [T.ToTensor()(frame) for frame in video_frames]

        if self.stack_frames:
            frames = torch.stack(frames).permute(1, 0, 2, 3)

        return frames, label

    def load_frames(self, frames_dir):
        frames = []
        for i in range(1, self.n_sampled_frames + 1):
            frame_file = os.path.join(frames_dir, f"frame_{i}.jpg")
            frame = Image.open(frame_file).convert("RGB")
            frames.append(frame)

        return frames

class FlowFramesNPYDataset(torch.utils.data.Dataset):
    def __init__(self,
                 root_dir='/work3/ppar/data/ucf101',
                 split='train',
                 transform=None,
                 stack_flows=True
                 ):

        self.video_paths = sorted(glob(f'{root_dir}/videos/{split}/*/*.avi'))
        self.df = pd.read_csv(f'{root_dir}/metadata/{split}.csv')
        self.split = split
        self.transform = transform
        self.stack_flows = stack_flows

        self.n_sampled_frames = 10

    def __len__(self):
        return len(self.video_paths)

    def _get_meta(self, attr, value):
        return self.df.loc[self.df[attr] == value]

    def __getitem__(self, idx):
        video_path = self.video_paths[idx]
        video_name = video_path.split('/')[-1].split('.avi')[0]
        video_meta = self._get_meta('video_name', video_name)
        label = video_meta['label'].item()

        video_flows_dir = self.video_paths[idx].split('.avi')[0].replace('videos', 'flows')
        video_flows = self.load_flows(video_flows_dir)

        if self.transform:
            flows = [self.transform(frame) for frame in video_flows]
        else:
            flows = [T.ToTensor()(frame) for frame in video_flows]

        if self.stack_flows:
            flows = torch.stack(flows).permute(1, 0, 2, 3)

        return flows, label

    def load_flows(self, flows_dir):
        flows = []
        for i in range(1, self.n_sampled_frames):
            flow_file = os.path.join(flows_dir, f"flow_{i}_{i+1}.npy")
            flow = np.load(flow_file)
            flows.append(flow)

        return flows


def datasetSingleFrame(batch_size=64, transform=None):
    from torch.utils.data import DataLoader

    root_dir = '/dtu/datasets1/02516/ucf101_noleakage/'

    transform = T.Compose([T.Resize((64, 64)), T.ToTensor()])

    trainset = FrameImageDataset(root_dir=root_dir, split='train', transform=transform)
    testset = FrameImageDataset(root_dir=root_dir, split='test', transform=transform)
    validationset = FrameImageDataset(root_dir=root_dir, split='val', transform=transform)

    train_loader = DataLoader(trainset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(testset, batch_size=batch_size, shuffle=False)
    val_loader = DataLoader(validationset, batch_size=batch_size, shuffle=False)

    return (train_loader, test_loader, val_loader), (trainset, testset, validationset)


def datasetVideoStackFrames(batch_size=64, transform=None):
    from torch.utils.data import DataLoader

    root_dir = '/dtu/datasets1/02516/ucf101_noleakage/'

    transform = T.Compose([T.Resize((64, 64)), T.ToTensor()])

    trainset = FrameVideoDataset(root_dir=root_dir, split='train', transform=transform, stack_frames=True)
    testset = FrameVideoDataset(root_dir=root_dir, split='test', transform=transform, stack_frames=True)
    validationset = FrameVideoDataset(root_dir=root_dir, split='val', transform=transform, stack_frames=True)

    train_loader = DataLoader(trainset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(testset, batch_size=batch_size, shuffle=False)
    val_loader = DataLoader(validationset, batch_size=batch_size, shuffle=False)

    return (train_loader, test_loader, val_loader), (trainset, testset, validationset)


def datasetVideoListFrames(batch_size=64, transform=None):
    from torch.utils.data import DataLoader

    root_dir = '/dtu/datasets1/02516/ucf101_noleakage/'

    transform = T.Compose([T.Resize((64, 64)), T.ToTensor()])

    trainset = FrameVideoDataset(root_dir=root_dir, split='train', transform=transform, stack_frames=False)
    testset = FrameVideoDataset(root_dir=root_dir, split='test', transform=transform, stack_frames=False)
    validationset = FrameVideoDataset(root_dir=root_dir, split='val', transform=transform, stack_frames=False)

    train_loader = DataLoader(trainset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(testset, batch_size=batch_size, shuffle=False)
    val_loader = DataLoader(validationset, batch_size=batch_size, shuffle=False)

    return (train_loader, test_loader, val_loader), (trainset, testset, validationset)

def datasetVideoFlows(batch_size=64, transform=None, stack_flows = False):
    from torch.utils.data import DataLoader

    root_dir = '/dtu/datasets1/02516/ucf101_noleakage/'

    transform = T.Compose([T.Resize((64, 64)), T.ToTensor()])

    trainset = FlowFramesNPYDataset(root_dir=root_dir, split='train', transform=transform, stack_flows=stack_flows)
    testset = FlowFramesNPYDataset(root_dir=root_dir, split='test', transform=transform, stack_flows=stack_flows)
    validationset = FlowFramesNPYDataset(root_dir=root_dir, split='val', transform=transform, stack_flows=stack_flows)

    train_loader = DataLoader(trainset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(testset, batch_size=batch_size, shuffle=False)
    val_loader = DataLoader(validationset, batch_size=batch_size, shuffle=False)

    return (train_loader, test_loader, val_loader), (trainset, testset, validationset)

import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms as T
import numpy as np
from glob import glob
import pandas as pd
from PIL import Image
import os


class DualStreamDataset(Dataset):
    """
    Dataset that loads both RGB frames and optical flow for dual-stream networks.
    Returns: (frames_tensor, flows_tensor, label)
    """
    def __init__(self,
                 root_dir='/work3/ppar/data/ucf101',
                 split='train',
                 transform=None,
                 n_sampled_frames=10):
        
        self.video_paths = sorted(glob(f'{root_dir}/videos/{split}/*/*.avi'))
        self.df = pd.read_csv(f'{root_dir}/metadata/{split}.csv')
        self.split = split
        self.transform = transform
        self.n_sampled_frames = n_sampled_frames

    def __len__(self):
        return len(self.video_paths)

    def _get_meta(self, attr, value):
        return self.df.loc[self.df[attr] == value]

    def __getitem__(self, idx):
        video_path = self.video_paths[idx]
        video_name = video_path.split('/')[-1].split('.avi')[0]
        video_meta = self._get_meta('video_name', video_name)
        label = video_meta['label'].item()

        # Get directories for frames and flows
        video_frames_dir = video_path.split('.avi')[0].replace('videos', 'frames')
        video_flows_dir = video_path.split('.avi')[0].replace('videos', 'flows')

        # Load frames and flows
        frames = self.load_frames(video_frames_dir)
        flows = self.load_flows(video_flows_dir)

        # Apply transforms
        if self.transform:
            frames = [self.transform(frame) for frame in frames]
            flows = [self.transform(flow) for flow in flows]
        else:
            frames = [T.ToTensor()(frame) for frame in frames]
            flows = [T.ToTensor()(flow) for flow in flows]

        # Stack to create [C, T, H, W] tensors
        frames = torch.stack(frames).permute(1, 0, 2, 3)  # [C, T, H, W]
        flows = torch.stack(flows).permute(1, 0, 2, 3)    # [C, T, H, W]

        return (frames, flows), label

    def load_frames(self, frames_dir):
        frames = []
        for i in range(1, self.n_sampled_frames + 1):
            frame_file = os.path.join(frames_dir, f"frame_{i}.jpg")
            frame = Image.open(frame_file).convert("RGB")
            frames.append(frame)
        return frames

    def load_flows(self, flows_dir):
        flows = []
        all_flows_data = []
        
        # First pass: load all flows to compute global normalization
        for i in range(1, self.n_sampled_frames):
            flow_file = os.path.join(flows_dir, f"flow_{i}_{i+1}.npy")
            flow = np.load(flow_file)
            all_flows_data.append(flow)
        
        # Compute global min/max for better normalization
        all_flows_concat = np.concatenate([f.flatten() for f in all_flows_data])
        global_min, global_max = np.percentile(all_flows_concat, [1, 99])  # Use percentiles to avoid outliers
        
        # Second pass: normalize and convert
        for flow in all_flows_data:
            # Convert numpy array to PIL Image for consistent transform pipeline
            # Assuming flow is [H, W, 2] or [H, W, C]
            if flow.ndim == 2:
                # Single channel, expand to 3 channels
                flow = np.stack([flow, flow, flow], axis=-1)
            elif flow.shape[-1] == 2:
                # 2-channel flow (u, v), pad to 3 channels
                flow = np.concatenate([flow, np.zeros_like(flow[:, :, :1])], axis=-1)
            
            # Normalize flow to [0, 255] range using global stats
            if global_max > global_min:
                flow = np.clip((flow - global_min) / (global_max - global_min) * 255, 0, 255).astype(np.uint8)
            else:
                flow = np.zeros_like(flow, dtype=np.uint8)
            
            flow = Image.fromarray(flow, mode='RGB')
            flows.append(flow)
        
        # Pad to have same number of flows as frames (duplicate last flow)
        if len(flows) < self.n_sampled_frames:
            flows.append(flows[-1])
        
        return flows


def datasetDualStream(batch_size=64, transform=None, n_sampled_frames=10):
    """
    Create data loaders for dual-stream model (frames + optical flow).
    
    Returns:
        (train_loader, test_loader, val_loader), (trainset, testset, validationset)
    """
    root_dir = '/dtu/datasets1/02516/ucf101_noleakage/'
    
    if transform is None:
        # Training augmentation
        transform = T.Compose([
            T.Resize((64, 64)),
            T.RandomHorizontalFlip(0.5),
            T.RandomRotation(15),
            T.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
    
    trainset = DualStreamDataset(
        root_dir=root_dir, 
        split='train', 
        transform=transform,
        n_sampled_frames=n_sampled_frames
    )
    testset = DualStreamDataset(
        root_dir=root_dir, 
        split='test', 
        transform=transform,
        n_sampled_frames=n_sampled_frames
    )
    validationset = DualStreamDataset(
        root_dir=root_dir, 
        split='val', 
        transform=transform,
        n_sampled_frames=n_sampled_frames
    )
    
    train_loader = DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=2)
    test_loader = DataLoader(testset, batch_size=batch_size, shuffle=False, num_workers=2)
    val_loader = DataLoader(validationset, batch_size=batch_size, shuffle=False, num_workers=2)
    
    return (train_loader, test_loader, val_loader), (trainset, testset, validationset)


# Test
if __name__ == "__main__":
    # Create dataset
    (train_loader, test_loader, val_loader), (trainset, testset, valset) = datasetDualStream(
        batch_size=4,
        n_sampled_frames=10
    )
    
    # Test loading a batch
    for (frames, flows), labels in train_loader:
        print(f"Frames shape: {frames.shape}")  
        print(f"Flows shape: {flows.shape}")    
        print(f"Labels shape: {labels.shape}")  
        break