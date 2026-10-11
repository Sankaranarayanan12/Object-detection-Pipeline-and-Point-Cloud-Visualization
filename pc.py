import numpy as np
import open3d as o3d

if __name__ == "__main__":
    dataset = o3d.data.SampleRedwoodRGBDImages() 
    depth_raw = o3d.io.read_image(dataset.depth_paths[0])
    intrinsic = o3d.io.read_pinhole_camera_intrinsic(dataset.camera_intrinsic_path)

    d = np.asarray(depth_raw)
    print("depth shape:", d.shape, "dtype:", d.dtype)
    print("depth min/max:", d.min(), d.max(), ", zero (invalid) pixels:", int((d == 0).sum()))
    print("intrinsics:", intrinsic.width, "x", intrinsic.height)
    print(intrinsic.intrinsic_matrix)

    pcd = o3d.geometry.PointCloud.create_from_depth_image(
        depth_raw,
        intrinsic,
        depth_scale=1000.0,   
        depth_trunc=4.0,      
        stride=1,
    )
    print("points after conversion:", len(pcd.points))
    pcd.transform([[1, 0, 0, 0],
                   [0, -1, 0, 0],
                   [0, 0, -1, 0],
                   [0, 0, 0, 1]])
    pcd, _ = pcd.remove_statistical_outlier(nb_neighbors=20, std_ratio=2.0)
    pcd = pcd.voxel_down_sample(voxel_size=0.01)
    print("points after cleaning:", len(pcd.points))

    z = np.asarray(pcd.points)[:, 2]
    t = (z - z.min()) / (z.max() - z.min() + 1e-9)
    pcd.colors = o3d.utility.Vector3dVector(np.stack([1 - t, np.zeros_like(t), t], axis=1))
    o3d.visualization.draw([pcd])