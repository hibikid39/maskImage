import sys
import os
import argparse
import cv2
import numpy as np
import open3d as o3d

# recover depth from rgb color
# https://dev.intelrealsense.com/docs/depth-image-compression-by-colorization-for-intel-realsense-depth-cameras
def rgb2depth(r, g, b, d_min=0.3, d_max=2.0):
    if b + g + r < 255:
        return 0.0
    elif r >= g and r >= b:
        if g >= b:
            d_rnormal = g - b
        else:
            d_rnormal = (g - b) + 1529
    elif g >= r and g >= b:
        d_rnormal = b - r + 510
    elif b >= g and b >= r:
        d_rnormal = r - g + 1020
    else:
        exit(1)

    return d_min + (d_max - d_min) * d_rnormal / 1529.0
    

def load_image(rgb_video_path, depth_frames_path, frame_idx):
    # load video
    vc = cv2.VideoCapture(rgb_video_path)

    frame_cnt = int(vc.get(cv2.CAP_PROP_FRAME_COUNT)) # フレーム数
    fps = int(vc.get(cv2.CAP_PROP_FPS)) # FPS
    print(f"frames:{frame_cnt}, fps:{fps}")

    # set frame position
    vc.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
    frame_pos = int(vc.get(cv2.CAP_PROP_POS_FRAMES)) 
    print(f"frame_pos:{frame_pos}/{frame_cnt}")

    # read frame
    ret, rgb = vc.read()

    vc.release()
    
    # load depth
    depth_path = os.path.join(depth_frames_path, f"{frame_idx:05}.jpg")
    depth_color_uint8 = cv2.imread(depth_path, cv2.IMREAD_UNCHANGED)
    depth_color_int32 = depth_color_uint8.astype(np.int32)
    H, W, _ = depth_color_int32.shape
    depth = np.zeros((H, W))
    for i in range(H):
        for j in range(W):
            depth[i, j] = rgb2depth(depth_color_int32[i, j, 0], depth_color_int32[i, j, 1], depth_color_int32[i, j, 2])
    
    return rgb, depth


def depth2pcd(depth, W, H, fx, fy, cx, cy):
    u, v = np.meshgrid(np.linspace(0, W-1, W), np.linspace(0, H-1, H), indexing='xy')
    dirs = np.stack(
        [(u-cx)/fx, -(v-cy)/fy, np.ones_like(u)], -1
    )
    pcd_np = (dirs * np.tile(depth.reshape(H, W, 1), (1, 1, 3))).reshape(-1, 3)
    
    return pcd_np


def viz(pcd_np):
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(pcd_np)
    o3d.visualization.draw_geometries([pcd])
    

def mask_rgbImage_fromDepth(rgb, depth):
    pass
    #return mask, masked_image


def main():
    parser = argparse.ArgumentParser(prog='maskImage', description='mask RGB image from depth')
    parser.add_argument('rgb_video_path', type=str, help='rgb video path')
    parser.add_argument('depth_frames_path', type=str, help='depth frames path')
    parser.add_argument('frame_idx', type=int, help='frame index')
    args = parser.parse_args()
    
    rgb, depth = load_image(args.rgb_video_path, args.depth_frames_path, args.frame_idx)
    
    depth_height, depth_width = depth.shape
    depth_FOV_H = 73.0
    depth_FOV_V = 59.0
    depth_fx = (depth_width / 2.0) / np.tan(np.deg2rad(depth_FOV_H / 2.0))
    depth_fy = (depth_height / 2.0) / np.tan(np.deg2rad(depth_FOV_V / 2.0))
    depth_cx = (depth_width - 1) / 2.0
    depth_cy = (depth_height - 1) / 2.0
    
    pcd_np = depth2pcd(depth, depth_width, depth_height, depth_fx, depth_fy, depth_cx, depth_cy)
    viz(pcd_np)
    #mask, masked_image = mask_rgbImage_fromDepth(rgb, depth)


if __name__ == "__main__":
    main()