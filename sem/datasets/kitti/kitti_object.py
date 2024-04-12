
import numpy as np
from pathlib import Path
import pickle
import visual_utils.visualize_utils as V
import mayavi.mlab as mlab
import pcdet.utils.common_utils as pc_utils
from spconv.utils import VoxelGeneratorV2 as VoxelGenerator
import torch.utils.data as torch_data

class KittiObject(torch_data.Dataset):
    def __init__(self, dataset_cfg=None, ):
        super(KittiObject, self).__init__()
        dbinfo_file = dataset_cfg['dbinfo_file']
        self.root_path = Path(dataset_cfg['root_path'])
        self.all_names = dataset_cfg['all_names']
        self.class_name = dataset_cfg['class_name']
        with open(dbinfo_file, 'rb') as f:
            self.dbinfos = pickle.load(f)

        self.cls_infos = self.dbinfos[self.class_name]

    def __len__(self):
        return len(self.cls_infos)

    def get_lidar(self,lidar_file):
        lidar_file = self.root_path / lidar_file
        assert lidar_file.exists()
        return np.fromfile(str(lidar_file),dtype=np.float32).reshape(-1,4)

    def __getitem__(self, index):
        infos = self.cls_infos[index]
        print(' sample : {}'.format(index))
        obj_pts = self.get_lidar(lidar_file=infos['path'])
        box3d_lidar = infos['box3d_lidar']
        obj_pts = pc_utils.rotate_points_along_z(obj_pts[np.newaxis,...],angle=-box3d_lidar[np.newaxis,6])[0]

        # V.draw_scenes(points=obj_pts)
        # mlab.show(stop=True)

        return obj_pts

if __name__ == '__main__':
    dbinfo_file = "/home/zlin/3dcv/openpcd/data/kitti/infos/ct3d/kitti_dbinfos_trainval.pkl"

    root_path = "/home/zlin/3dcv/openpcd/data/kitti"

    dataset_cfg = {
        'root_path': root_path,
        'dbinfo_file': dbinfo_file,
        'all_names': ['Pedestrian','Car','Cyclist','Van','Truck','Tram','Misc','Person_sitting'],
        'class_name': 'Cyclist'
    }

    cls_file = "/home/zlin/3dcv/openpcd/data/kitti/infos/ct3d/model/{}_model.bin".format(
        dataset_cfg['class_name'])

    dataset = KittiObject(dataset_cfg=dataset_cfg)

    # obj_pts_list = [dataset[i] for i in range(len(dataset))]
    # cls_pts = np.concatenate(obj_pts_list,axis=0)
    # with open(cls_file,'wb') as f:
    #     print('save model file to {}'.format(cls_file))
    #     cls_pts.tofile(f)
    # print(cls_pts.shape)
    # exit()


    cls_pts = np.fromfile(str(cls_file),dtype=np.float32).reshape(-1,4)

    # np.random.shuffle(cls_pts)

    np.set_printoptions(precision=3)
    for i in range(4): # (x,y,z,i)
        min_p = np.min(cls_pts[:,i])
        max_p = np.max(cls_pts[:,i])
        print(min_p,' ',max_p)

    point_cloud_range = np.array([-4, -4, -2, 4, 4, 2], dtype=np.float32)
    voxel_size = np.array([0.05, 0.05, 0.05], dtype=np.float32)
    voxel_generator = VoxelGenerator(
                    voxel_size=voxel_size,
                    point_cloud_range=point_cloud_range,
                    max_num_points=2000,max_voxels=150000)
    grid_size = voxel_generator.grid_size

    voxel_output = voxel_generator.generate(cls_pts)
    voxels = voxel_output['voxels']
    coordinates = voxel_output['coordinates']
    num_points_per_voxel = voxel_output['num_points_per_voxel']

    select_ = num_points_per_voxel > np.mean(num_points_per_voxel)   # Car: 98.9
    select_voxel = coordinates[select_] # zyx format

    offset = np.array([-4, -4, -2], dtype=np.float32)
    select_pt_xyz = select_voxel[:,[2,1,0]] * voxel_size + offset + 0.5 * voxel_size
    select_pt_xyz = select_pt_xyz.astype(np.float32)
    print(select_pt_xyz.shape)

    # length = 100000
    # cls_pts = cls_pts[:length]

    V.draw_scenes(points=select_pt_xyz)

    mlab.show(stop=True)

    with open(cls_file,'wb') as f:
        print('save model file to {}'.format(cls_file))
        select_pt_xyz.tofile(f)

    print()



