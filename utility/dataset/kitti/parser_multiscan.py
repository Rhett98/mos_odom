import os
import sys
path = os.getcwd()
sys.path.append(path)
import random
import numpy as np

import torch
from torch.utils.data import Dataset

from utility.laserscan import LaserScan, SemLaserScan
from utility.dataset.kitti.utils import load_poses, load_calib
from utility.geometry import get_quaternion_from_transformation_matrix, get_translation_from_transformation_matrix,get_euler_from_transformation_matrix

EXTENSIONS_SCAN = ['.bin']
EXTENSIONS_LABEL = ['.label']
EXTENSIONS_RESIDUAL = ['.npy']


def is_scan(filename):
    return any(filename.endswith(ext) for ext in EXTENSIONS_SCAN)


def is_label(filename):
    return any(filename.endswith(ext) for ext in EXTENSIONS_LABEL)


def is_residual(filename):
    return any(filename.endswith(ext) for ext in EXTENSIONS_RESIDUAL)


def my_collate(batch):
    data = [item[0] for item in batch]
    project_mask = [item[1] for item in batch]
    proj_labels = [item[2] for item in batch]
    data = torch.stack(data,dim=0)
    project_mask = torch.stack(project_mask,dim=0)
    proj_labels = torch.stack(proj_labels, dim=0)

    to_augment =(proj_labels == 12).nonzero()
    to_augment_unique_12 = torch.unique(to_augment[:, 0])

    to_augment = (proj_labels == 5).nonzero()
    to_augment_unique_5 = torch.unique(to_augment[:, 0])

    to_augment = (proj_labels == 8).nonzero()
    to_augment_unique_8 = torch.unique(to_augment[:, 0])

    to_augment_unique = torch.cat((to_augment_unique_5,to_augment_unique_8,to_augment_unique_12),dim=0)
    to_augment_unique = torch.unique(to_augment_unique)

    for k in to_augment_unique:
        data = torch.cat((data,torch.flip(data[k.item()], [2]).unsqueeze(0)),dim=0)
        proj_labels = torch.cat((proj_labels,torch.flip(proj_labels[k.item()], [1]).unsqueeze(0)),dim=0)
        project_mask = torch.cat((project_mask,torch.flip(project_mask[k.item()], [1]).unsqueeze(0)),dim=0)

    return data, project_mask,proj_labels


class SemanticKitti(Dataset):

    def __init__(self, root, # directory where data is
                 sequences,     # sequences for this data (e.g. [1,3,4,6])
                 labels,        # label dict: (e.g 10: "car")
                 color_map,     # colors dict bgr (e.g 10: [255, 0, 0])
                 learning_map,  # classes to learn (0 to N-1 for xentropy)
                 learning_map_inv,    # inverse of previous (recover labels)
                 sensor,              # sensor to parse scans from
                 max_points=150000,   # max number of points present in dataset
                 gt=True,             # send ground truth?
                 transform=False,
                 drop_few_static_frames=False):
        # save deats
        self.root = os.path.join(root, "sequences")
        self.sequences = sequences
        self.labels = labels
        self.color_map = color_map
        self.learning_map = learning_map
        self.learning_map_inv = learning_map_inv
        self.sensor = sensor
        self.sensor_img_H = sensor["img_prop"]["height"]
        self.sensor_img_W = sensor["img_prop"]["width"]
        self.sensor_img_means = torch.tensor(sensor["img_means"], dtype=torch.float)
        self.sensor_img_stds = torch.tensor(sensor["img_stds"], dtype=torch.float)
        self.sensor_fov_up = sensor["fov_up"]
        self.sensor_fov_down = sensor["fov_down"]
        self.max_points = max_points
        self.gt = gt
        self.transform = transform

        """
        Added stuff for dynamic object segmentation
        """
        # dictionary for mapping a dataset index to a sequence, frame_id tuple needed for using multiple frames
        self.dataset_size = 0
        self.index_mapping = {}
        dataset_index = 0
        # added this for dynamic object removal
        self.n_input_scans = sensor["n_input_scans"]  # This needs to be the same as in arch_cfg.yaml!
        self.use_residual = sensor["residual"]
        self.transform_mod = sensor["transform"]
        self.use_normal = sensor["use_normal"] if 'use_normal' in sensor.keys() else False
        """"""

        # get number of classes (can't be len(self.learning_map) because there
        # are multiple repeated entries, so the number that matters is how many
        # there are for the xentropy)
        self.nclasses = len(self.learning_map_inv)

        # sanity checks

        # make sure directory exists
        if os.path.isdir(self.root):
            print("Sequences folder exists! Using sequences from %s" % self.root)
        else:
            raise ValueError("Sequences folder doesn't exist! Exiting...")

        assert(isinstance(self.labels, dict))	# make sure labels is a dict
        assert(isinstance(self.color_map, dict)) # make sure color_map is a dict
        assert(isinstance(self.learning_map, dict)) # make sure learning_map is a dict
        assert(isinstance(self.sequences, list)) # make sure sequences is a list

        # placeholder for filenames
        self.scan_files = {}
        self.label_files = {}
        self.poses = {}

        if self.use_residual:
            for i in range(self.n_input_scans):
                exec("self.residual_files_" + str(str(i+1)) + " = {}")

        # fill in with names, checking that all sequences are complete
        for seq in self.sequences:

            seq = '{0:02d}'.format(int(seq)) # to string
            print("parsing seq {}".format(seq))

            # get paths for each
            scan_path = os.path.join(self.root, seq, "velodyne")
            label_path = os.path.join(self.root, seq, "labels")

            if self.use_residual:
                for i in range(self.n_input_scans):
                    exec("residual_path_" + str(i+1) + " = os.path.join(self.root, seq, residual_images_" + str(i+1))

            # get files
            scan_files = [os.path.join(dp, f) for dp, dn, fn in os.walk(
                    os.path.expanduser(scan_path)) for f in fn if is_scan(f)]
            label_files = [os.path.join(dp, f) for dp, dn, fn in os.walk(
                    os.path.expanduser(label_path)) for f in fn if is_label(f)]

            if self.use_residual:
                for i in range(self.n_input_scans):
                    exec("residual_files_" + str(i+1) + " = " + '[os.path.join(dp, f) for dp, dn, fn in '
                             'os.walk(os.path.expanduser(residual_path_' + str(i+1) + '))'
                             ' for f in fn if is_residual(f)]')

            ### Get poses and transform them to LiDAR coord frame for transforming point clouds
            # load poses
            pose_file = os.path.join(self.root, seq, "poses.txt")
            poses = np.array(load_poses(pose_file))
            inv_frame0 = np.linalg.inv(poses[0])

            # load calibrations
            calib_file = os.path.join(self.root, seq, "calib.txt")
            T_cam_velo = load_calib(calib_file)
            T_cam_velo = np.asarray(T_cam_velo).reshape((4, 4))
            T_velo_cam = np.linalg.inv(T_cam_velo)
            
            # convert kitti poses from camera coord to LiDAR coord
            new_poses = []
            for pose in poses:
                new_poses.append(T_velo_cam.dot(inv_frame0).dot(pose).dot(T_cam_velo))
            self.poses[seq] = np.array(new_poses)
            # check all scans have labels
            if self.gt:
                assert(len(scan_files) == len(label_files))

            """
            Added for dynamic object segmentation
            """
            # fill index mapper which is needed when loading several frames
            n_used_files = max(0, len(scan_files) - self.n_input_scans + 1)  # this is used for multi-scan attach
            # n_used_files = max(0, len(scan_files))  # this is used for multi residual images
            for start_index in range(n_used_files):
                self.index_mapping[dataset_index] = (seq, start_index)
                dataset_index += 1
            self.dataset_size += n_used_files
            """"""

            # extend list
            scan_files.sort()
            label_files.sort()

            self.scan_files[seq] = scan_files
            self.label_files[seq] = label_files

            if self.use_residual:
                for i in range(self.n_input_scans):
                    exec("residual_files_" + str(i+1) + ".sort()")
                    exec("self.residual_files_" + str(i+1) + "[seq]" + " = " + "residual_files_" + str(i+1))
        # print("\033[32m No model directory found.\033[0m")

        print(f"\033[32m There are {self.dataset_size} frames in total. \033[0m")
        if drop_few_static_frames:
            self.remove_few_static_frames()
            print(f"\033[32m Remove {self.total_remove} frames. \n New use {self.dataset_size} frames. \033[0m")

        print(f"\033[32m Using {self.dataset_size} scans from sequences {self.sequences}\033[0m")

    def __getitem__(self, dataset_index):

        # Get sequence and start
        seq, start_index = self.index_mapping[dataset_index]  # seq: '05' start_index: 1856
        current_index = start_index + self.n_input_scans - 1  # this is used for multi-scan attach
        # current_index = start_index  # this is used for multi residual images
        if current_index > len(self.poses[seq])-1:
            start_index += 1-self.n_input_scans
            current_index += 1-self.n_input_scans
        current_pose = self.poses[seq][current_index]
        proj_full = torch.Tensor()
        trans_list = []
        quaternion_list = []
        # index is now looping from first scan in input sequence to current scan
        for index in range(start_index, current_index+1):
        # for index in range(start_index, start_index + 1):
            # get item in tensor shape
            scan_file = self.scan_files[seq][index]
            if self.gt:
                label_file = self.label_files[seq][index]

            if self.use_residual:
                for i in range(self.n_input_scans):
                    exec("residual_file_" + str(i+1) + " = " + "self.residual_files_" + str(i+1) + "[seq][index]")
            index_pose = self.poses[seq][index]
            # get translation and _quaternion between index_pose and current_pose
            if index != current_index:
                relative_pose = np.linalg.inv(index_pose).dot(current_pose)
                trans_list.append(get_translation_from_transformation_matrix(relative_pose))
                # quaternion_list.append(get_quaternion_from_transformation_matrix(relative_pose))
                quaternion_list.append(get_euler_from_transformation_matrix(relative_pose))
            # open a semantic laserscan
            DA = False
            flip_sign = False
            rot = False
            drop_points = False
            if self.transform:
                if random.random() > 0.5:
                    if random.random() > 0.5:
                            DA = True
                    if random.random() > 0.5:
                            flip_sign = True
                    if random.random() > 0.5:
                            rot = True
                    drop_points = random.uniform(0, 0.5)

            if self.gt:
                scan = SemLaserScan(self.color_map,
                                    project=True,
                                    H=self.sensor_img_H,
                                    W=self.sensor_img_W,
                                    fov_up=self.sensor_fov_up,
                                    fov_down=self.sensor_fov_down,
                                    DA=DA,
                                    flip_sign=flip_sign,
                                    drop_points=drop_points,
                                    use_normal=self.use_normal)
            else:
                scan = LaserScan(project=True,
                                 H=self.sensor_img_H,
                                 W=self.sensor_img_W,
                                 fov_up=self.sensor_fov_up,
                                 fov_down=self.sensor_fov_down,
                                 DA=DA,
                                 rot=rot,
                                 flip_sign=flip_sign,
                                 drop_points=drop_points,
                                 use_normal=self.use_normal)

            # open and obtain (transformed) scan
            scan.open_scan(scan_file, index_pose, current_pose, if_transform=self.transform_mod)

            if self.gt:
                scan.open_label(label_file)
                # map unused classes to used classes (also for projection)
                scan.sem_label = self.map(scan.sem_label, self.learning_map)
                scan.proj_sem_label = self.map(scan.proj_sem_label, self.learning_map)

            # make a tensor of the uncompressed data (with the max num points)
            unproj_n_points = scan.points.shape[0]
            unproj_xyz = torch.full((self.max_points, 3), -1.0, dtype=torch.float)
            unproj_xyz[:unproj_n_points] = torch.from_numpy(scan.points)
            unproj_range = torch.full([self.max_points], -1.0, dtype=torch.float)
            unproj_range[:unproj_n_points] = torch.from_numpy(scan.unproj_range)
            unproj_remissions = torch.full([self.max_points], -1.0, dtype=torch.float)
            unproj_remissions[:unproj_n_points] = torch.from_numpy(scan.remissions)
            if self.gt:
                unproj_labels = torch.full([self.max_points], -1.0, dtype=torch.int32)
                unproj_labels[:unproj_n_points] = torch.from_numpy(scan.sem_label)
            else:
                unproj_labels = []

            # get points and labels
            proj_range = torch.from_numpy(scan.proj_range).clone()
            proj_xyz = torch.from_numpy(scan.proj_xyz).clone()
            proj_remission = torch.from_numpy(scan.proj_remission).clone()
            proj_mask = torch.from_numpy(scan.proj_mask)
            if self.gt:
                proj_labels = torch.from_numpy(scan.proj_sem_label).clone()
                proj_labels = proj_labels * proj_mask
            else:
                proj_labels = []
            proj_x = torch.full([self.max_points], -1, dtype=torch.long)
            proj_x[:unproj_n_points] = torch.from_numpy(scan.proj_x)
            proj_y = torch.full([self.max_points], -1, dtype=torch.long)
            proj_y[:unproj_n_points] = torch.from_numpy(scan.proj_y)

            if self.use_residual:
                for i in range(self.n_input_scans):
                    exec("proj_residuals_" + str(i+1) + " = torch.Tensor(np.load(residual_file_" + str(i+1) + "))")
            
            proj = torch.cat([proj_range.unsqueeze(0).clone(),      # torch.Size([1, 64, 2048])
                            proj_xyz.clone().permute(2, 0, 1),    # torch.Size([3, 64, 2048])
                            proj_remission.unsqueeze(0).clone(),]) # torch.Size([1, 64, 2048])
            proj = (proj - self.sensor_img_means[:, None, None]) / self.sensor_img_stds[:, None, None]
            
            if self.use_normal:
                proj_normal = torch.from_numpy(scan.normal_map).clone().permute(2, 0, 1)
                proj = torch.cat([proj, proj_normal,]) 

            proj_full = torch.cat([proj_full, proj.view(1,proj.shape[0],proj.shape[1],proj.shape[2])])

        # add residual channel
        if self.use_residual:
            for i in range(self.n_input_scans):
                proj_full = torch.cat([proj_full, torch.unsqueeze(eval("proj_residuals_" + str(i+1)), 0)])

        proj_full = proj_full * proj_mask.float()

        # get name and sequence
        path_norm = os.path.normpath(scan_file)
        path_split = path_norm.split(os.sep)
        path_seq = path_split[-3]
        path_name = path_split[-1].replace(".bin", ".label")

        return proj_full, proj_mask, proj_labels, unproj_labels, path_seq, path_name, proj_x, proj_y, proj_range, \
                     unproj_range, proj_xyz, unproj_xyz, proj_remission, unproj_remissions, unproj_n_points, trans_list, quaternion_list

    def __len__(self):
        return self.dataset_size

    @staticmethod
    def map(label, mapdict):
        # put label from original values to xentropy
        # or vice-versa, depending on dictionary values
        # make learning map a lookup table
        maxkey = 0
        for key, data in mapdict.items():
            if isinstance(data, list):
                nel = len(data)
            else:
                nel = 1
            if key > maxkey:
                maxkey = key
        # +100 hack making lut bigger just in case there are unknown labels
        if nel > 1:
            lut = np.zeros((maxkey + 100, nel), dtype=np.int32)
        else:
            lut = np.zeros((maxkey + 100), dtype=np.int32)
        for key, data in mapdict.items():
            try:
                lut[key] = data
            except IndexError:
                print("Wrong key ", key)
        # do the mapping
        return lut[label]

    def remove_few_static_frames(self):
        # Developed by Jiadai Sun 2021-11-07
        # This function is used to clear some frames, because too many static frames will lead to a long training time
        # There are several main dicts that need to be modified and processed
        #   self.scan_files, self.label_files, self.residual_files_1 ....8
        #   self.poses, self.index_mapping
        #   self.dataset_size (int number)

        remove_mapping_path = os.path.join(os.path.dirname(__file__), "../../../config/train_split_dynamic_pointnumber.txt")
        with open(remove_mapping_path) as fd:
            lines = fd.readlines()
            lines = [line.strip() for line in lines]

        pending_dict = {}
        for line in lines:
            if line != '':
                seq, fid, _ = line.split()
                if int(seq) in self.sequences:
                    if seq in pending_dict.keys():
                        if fid in pending_dict[seq]:
                            raise ValueError(f"!!!! Duplicate {fid} in seq {seq} in .txt file")
                        pending_dict[seq].append(fid)
                    else:
                        pending_dict[seq] = [fid]

        self.total_remove = 0
        self.index_mapping = {} # Reinitialize
        dataset_index = 0
        self.dataset_size = 0
        for seq in self.sequences:
            seq = '{0:02d}'.format(int(seq))
            if seq in pending_dict.keys():
                raw_len = len(self.scan_files[seq])

                # lidar scan files
                scan_files = self.scan_files[seq]
                useful_scan_paths = [path for path in scan_files if os.path.split(path)[-1][:-4] in pending_dict[seq]]
                self.scan_files[seq] = useful_scan_paths

                # label_files
                label_files = self.label_files[seq]
                useful_label_paths = [path for path in label_files if os.path.split(path)[-1][:-6] in pending_dict[seq]]
                self.label_files[seq] = useful_label_paths

                # poses_file
                self.poses[seq] = self.poses[seq][list(map(int, pending_dict[seq]))]

                assert len(useful_scan_paths) == len(useful_label_paths)
                assert len(useful_scan_paths) == self.poses[seq].shape[0]

                # the index_mapping and dataset_size is used in dataloader __getitem__
                n_used_files = max(0, len(useful_scan_paths))  # this is used for multi residual images
                for start_index in range(n_used_files):
                    self.index_mapping[dataset_index] = (seq, start_index)
                    dataset_index += 1
                self.dataset_size += n_used_files

                # More elegant implementation
                if self.use_residual:
                    for i in range(self.n_input_scans):
                        tmp_residuals = eval(f"self.residual_files_{i+1}[\'{seq}\']")
                        tmp_pending_list = eval(f"pending_dict[\'{seq}\']")
                        tmp_usefuls = [path for path in tmp_residuals if os.path.split(path)[-1][:-4] in tmp_pending_list]
                        exec(f"self.residual_files_{i+1}[\'{seq}\'] = tmp_usefuls")
                        new_len = len(eval(f"self.residual_files_{i+1}[\'{seq}\']"))
                        print(f"  Drop residual_images_{i+1} in seq{seq}: {len(tmp_residuals)} -> {new_len}")
                        if i >= 2:
                            exec(f"assert len(self.residual_files_{i-1}[\'{seq}\']) == len(self.residual_files_{i}[\'{seq}\'])")

                new_len = len(self.scan_files[seq])
                print(f"Seq {seq} drop {raw_len - new_len}: {raw_len} -> {new_len}")
                self.total_remove += raw_len - new_len


class Parser():
    # standard conv, BN, relu
    def __init__(self,
                 root,              # directory for data
                 train_sequences,   # sequences to train
                 valid_sequences,   # sequences to validate.
                 test_sequences,    # sequences to test (if none, don't get)
                 split,             # split (train, valid, test)
                 labels,            # labels in data
                 color_map,         # color for each label
                 learning_map,      # mapping for training labels
                 learning_map_inv,  # recover labels from xentropy
                 sensor,            # sensor to use
                 max_points,        # max points in each scan in entire dataset
                 batch_size,        # batch size for train and val
                 workers,           # threads to load data
                 gt=True,           # get gt?
                 shuffle_train=False):  # shuffle training set?
        super(Parser, self).__init__()

        # if I am training, get the dataset
        self.root = root
        self.train_sequences = train_sequences
        self.valid_sequences = valid_sequences
        self.test_sequences = test_sequences
        self.split = split
        self.labels = labels
        self.color_map = color_map
        self.learning_map = learning_map
        self.learning_map_inv = learning_map_inv
        self.sensor = sensor
        self.max_points = max_points
        self.batch_size = batch_size
        self.workers = workers
        self.gt = gt
        self.shuffle_train = shuffle_train

        # number of classes that matters is the one for xentropy
        self.nclasses = len(self.learning_map_inv)

        # Data loading code
        if self.split == 'train':
            self.train_dataset = SemanticKitti(root=self.root,
                                               sequences=self.train_sequences,
                                               labels=self.labels,
                                               color_map=self.color_map,
                                               learning_map=self.learning_map,
                                               learning_map_inv=self.learning_map_inv,
                                               sensor=self.sensor,
                                               max_points=max_points,
                                               transform=True,
                                               gt=self.gt,
                                               drop_few_static_frames=False)

            self.trainloader = torch.utils.data.DataLoader(self.train_dataset,
                                                           batch_size=self.batch_size,
                                                           shuffle=self.shuffle_train,
                                                           # shuffle=False,
                                                           num_workers=self.workers,
                                                           pin_memory=True,
                                                           drop_last=True)
            assert len(self.trainloader) > 0
            self.trainiter = iter(self.trainloader)

            self.valid_dataset = SemanticKitti(root=self.root,
                                               sequences=self.valid_sequences,
                                               labels=self.labels,
                                               color_map=self.color_map,
                                               learning_map=self.learning_map,
                                               learning_map_inv=self.learning_map_inv,
                                               sensor=self.sensor,
                                               max_points=max_points,
                                               gt=self.gt)

            self.validloader = torch.utils.data.DataLoader(self.valid_dataset,
                                                           batch_size=self.batch_size,
                                                           shuffle=False,
                                                           num_workers=self.workers,
                                                           pin_memory=True,
                                                           drop_last=True)
            assert len(self.validloader) > 0
            self.validiter = iter(self.validloader)

        if self.split == 'valid':
            self.valid_dataset = SemanticKitti(root=self.root,
                                               sequences=self.valid_sequences,
                                               labels=self.labels,
                                               color_map=self.color_map,
                                               learning_map=self.learning_map,
                                               learning_map_inv=self.learning_map_inv,
                                               sensor=self.sensor,
                                               max_points=max_points,
                                               gt=self.gt)

            self.validloader = torch.utils.data.DataLoader(self.valid_dataset,
                                                           batch_size=self.batch_size,
                                                           shuffle=False,
                                                           num_workers=self.workers,
                                                           pin_memory=True,
                                                           drop_last=True)
            assert len(self.validloader) > 0
            self.validiter = iter(self.validloader)

        if self.split == 'test':
            if self.test_sequences:
                self.test_dataset = SemanticKitti(root=self.root,
                                                  sequences=self.test_sequences,
                                                  labels=self.labels,
                                                  color_map=self.color_map,
                                                  learning_map=self.learning_map,
                                                  learning_map_inv=self.learning_map_inv,
                                                  sensor=self.sensor,
                                                  max_points=max_points,
                                                  gt=False)

                self.testloader = torch.utils.data.DataLoader(self.test_dataset,
                                                              batch_size=self.batch_size,
                                                              shuffle=False,
                                                              num_workers=self.workers,
                                                              pin_memory=True,
                                                              drop_last=True)
                assert len(self.testloader) > 0
                self.testiter = iter(self.testloader)

    def get_train_batch(self):
        scans = self.trainiter.next()
        return scans

    def get_train_set(self):
        return self.trainloader

    def get_valid_batch(self):
        scans = self.validiter.next()
        return scans

    def get_valid_set(self):
        return self.validloader

    def get_test_batch(self):
        scans = self.testiter.next()
        return scans

    def get_test_set(self):
        return self.testloader

    def get_train_size(self):
        return len(self.trainloader)

    def get_valid_size(self):
        return len(self.validloader)

    def get_test_size(self):
        return len(self.testloader)

    def get_n_classes(self):
        return self.nclasses

    def get_original_class_string(self, idx):
        return self.labels[idx]

    def get_xentropy_class_string(self, idx):
        return self.labels[self.learning_map_inv[idx]]

    def to_original(self, label):
        # put label in original values
        return SemanticKitti.map(label, self.learning_map_inv)

    def to_xentropy(self, label):
        # put label in xentropy values
        return SemanticKitti.map(label, self.learning_map)

    def to_color(self, label):
        # put label in original values
        label = SemanticKitti.map(label, self.learning_map_inv)
        # put label in color
        return SemanticKitti.map(label, self.color_map)
    
from scipy.spatial.transform import Rotation as R

def get_x_q(pose: torch.Tensor):
    """ Get x, q vectors from pose matrix 
    Args:
        pose (Bx4x4 array): relative pose
    Returns:
        x (Bx3x1 array): translation 
        q (Bx4x1 array): quarternion
    """
    x = pose[:, :-1, -1]
    rot = pose[:, :-1, :-1] 
    r = R.from_matrix(rot.detach().numpy())
    q = torch.from_numpy(r.as_quat())
    
    return x.float(), q.float()

def get_pose(x, q):
    """ Get 4x4 pose from x and q numpy vectors
    Args:
        x (3x1 array): translation 
        q (4x1 array): quarternion
    Returns:
        pose (4x4 array): transformation pose
    """
    pose = np.identity(4)
    r = R.from_quat(q)
    rot = r.as_matrix()
    pose[:-1, :-1] = rot
    pose[:-1, -1] = x
    
    return pose
    
if __name__ == '__main__':
    import yaml
    from utility.geometry import get_transformation_matrix_euler
    from utility.dataset.kitti.utils import write_poses, load_calib
    ARCH = yaml.safe_load(open('config/arch/mos-motion.yml', 'r'))
    DATA = yaml.safe_load(open('config/data/local-test.yaml', 'r'))
    data = '../dataset'
    # DATA = yaml.safe_load(open('config/labels/kitti-toy.yaml', 'r'))
    # data = '/home/robot/Repository/data_odometry_velodyne/dataset'
    parse = Parser(root=data,
                            train_sequences=DATA["split"]["train"],
                            valid_sequences=DATA["split"]["valid"],
                            test_sequences=None,
                            split='train',
                            labels=DATA["labels"],
                            color_map=DATA["color_map"],
                            learning_map=DATA["learning_map"],
                            learning_map_inv=DATA["learning_map_inv"],
                            sensor=ARCH["dataset"]["sensor"],
                            max_points=ARCH["dataset"]["max_points"],
                            batch_size=ARCH["train"]["batch_size"],
                            workers=ARCH["train"]["workers"],
                            gt=True,
                            shuffle_train=True)
    loader = parse.get_train_set()
    # assert len(loader) > 0
    # load calibrations
    calib_file = os.path.join("/home/yu/Resp/dataset/sequences/08/calib.txt")
    T_cam_velo = load_calib(calib_file)
    T_cam_velo = np.asarray(T_cam_velo).reshape((4, 4))
    T_velo_cam = np.linalg.inv(T_cam_velo)
    last_pose = np.eye(4)
    # loader = parse.train_dataset
    # for i in range(5,10):
    #     proj_in, proj_mask,proj_labels, _, path_seq, path_name, p_x, p_y, proj_range, unproj_range, _, _, _, _, npoints,trans,rot = loader.__getitem__(i)
    #     print('********* '+ str(i) + " ***********")
    #     print(trans[-1],rot[-1])
    #     print(get_transformation_matrix_euler(trans[-1],rot[-1]))
    #     print(torch.norm(rot[-1]))
    for i, (proj_in, proj_mask,proj_labels, _, path_seq, path_name, p_x, p_y, proj_range, unproj_range, _, _, _, _, npoints,trans,rot) in enumerate(loader):
        # relative_matrix = get_transformation_matrix_quaternion(trans[-1], rot[-1])
        # last_pose = write_poses("posetest.txt", np.dot(T_cam_velo,np.dot(relative_matrix,T_velo_cam)), last_pose)
        print('********* '+ str(i) + " ***********")
        print(trans[-1],rot[-1])
        print(get_transformation_matrix_euler(trans[-1],rot[-1]))
        print(torch.norm(rot[-1]))
    # pose_file = os.path.join("/home/yu/Resp/dataset/sequences/06/poses.txt")
    # poses = np.array(load_poses(pose_file))
    # inv_frame0 = np.linalg.inv(poses[0])
    # print(poses.shape)
    # print(poses[9, :])
    # print(poses[10, :])

    # # load calibrations
    # calib_file = os.path.join("/home/yu/Resp/dataset/sequences/08/calib.txt")
    # T_cam_velo = load_calib(calib_file)
    # T_cam_velo = np.asarray(T_cam_velo).reshape((4, 4))
    # T_velo_cam = np.linalg.inv(T_cam_velo)
    # print(poses[0])
    # # convert kitti poses from camera coord to LiDAR coord
    # new_poses = []
    # for pose in poses:
    #     new_poses.append(T_velo_cam.dot(inv_frame0).dot(pose).dot(T_cam_velo))
    # poses = np.array(new_poses)
    
    # relative_poses = []
    # for i in range(len(poses)-1):
    #     p = np.linalg.inv(poses[i]).dot(poses[i+1])
    #     relative_poses.append(p)
    #     # print(p)
    #     # print('****')
    #     # print(get_quaternion_from_transformation_matrix(p))
    #     # print(get_translation_from_transformation_matrix(p))
        