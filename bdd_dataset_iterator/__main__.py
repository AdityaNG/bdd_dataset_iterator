import cv2
import open3d as o3d
import os
import numpy as np
import time

def get_2D_visual_frame(rgb_frame, disparity_frame, img_scale_factor=0.3):
    # Convert disparity to an integer img to visualize
    disparity_frame_norm = (disparity_frame - disparity_frame.min()) / (
        disparity_frame.max() - disparity_frame.min()
    )
    disparity_frame_int = (disparity_frame_norm * 255).astype("uint8")
    disparity_frame_color = cv2.applyColorMap(
        disparity_frame_int, cv2.COLORMAP_JET
    )

    visual = np.hstack([rgb_frame, disparity_frame_color])
    visual_small = cv2.resize(visual, (0,0), fx=img_scale_factor, fy=img_scale_factor)

    return visual_small


def main(args):
    from .bengaluru_driving_dataset import BengaluruOccupancyDatasetIterator

    args.dataset = os.path.expanduser(args.dataset)
    args.calib = os.path.expanduser(args.calib)

    print('Loading dataset from ', args.dataset)
    print('Loading calibration from ', args.calib)

    dataset = BengaluruOccupancyDatasetIterator(
        dataset_path=args.dataset,
        settings_doc=args.calib,

        pc_scale=(1.0, 1.0, 1.0),
        pc_shift=(0.0, 0.0, 0.0),
    )

    vis = o3d.visualization.Visualizer()
    vis.create_window("BDD Dataset Pointcloud")

    pcd = o3d.geometry.PointCloud()


    for index in range(0, len(dataset), 10):
        frame = dataset[index]
        print(frame.keys())
        rgb_frame = frame["rgb_frame"]      # (H, W, 3) uint8
        disparity_frame = frame["disparity_frame"]  # (H, W) Float32
        points = frame["points"]  # (N, 3) Float32
        points_colors = frame["points_colors"]  # (N, 3) Float32
        print('points_colors', points_colors.shape)
        # occupancy_points = frame["occupancy_points"]  # (N, 3) Float32

        visual_small = get_2D_visual_frame(rgb_frame, disparity_frame)

        cv2.imshow("BDD Dataset", visual_small)

        # Points min and max distances
        print("x: ", points[:, 0].min(), points[:, 0].max())
        print("y: ", points[:, 1].min(), points[:, 1].max())
        print("z: ", points[:, 2].min(), points[:, 2].max())


        points = points[::10, :]
        points_colors = points_colors[::10, :]

        points[:, 2] = -points[:, 2]  # Flip z axis
        points[:, 1] = -points[:, 1]  # Flip y axis
        # Swap x, y axes
        # points[:, [0, 1]] = points[:, [1, 0]]

        # Filter points which are more than 30 units from the origin
        distances = np.sqrt(np.sum(points**2, axis=1))
        mask = distances <= 0.005
        points = points[mask]
        points_colors = points_colors[mask]

        print("x: ", points[:, 0].min(), points[:, 0].max())
        print("y: ", points[:, 1].min(), points[:, 1].max())
        print("z: ", points[:, 2].min(), points[:, 2].max())

        pcd.points = o3d.utility.Vector3dVector(points)
        pcd.colors = o3d.utility.Vector3dVector(points_colors)
        pcd.remove_non_finite_points()

        vis.add_geometry(pcd)
        
        vis.update_geometry(pcd)

        key = ""
        while key != ord("n"):
            vis.poll_events()
            vis.update_renderer()
        
            key = cv2.waitKey(1)
            if key == ord("q"):
                return


if __name__ == "__main__":
    import argparse
    from .bdd_helper import DEFAULT_DATASET, DEFAULT_CALIB

    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default=DEFAULT_DATASET)
    parser.add_argument("--calib", type=str, default=DEFAULT_CALIB)

    args = parser.parse_args()

    main(args)