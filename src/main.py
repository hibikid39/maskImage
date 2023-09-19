import sys
import os
import argparse
import cv2
import numpy as np
import open3d as o3d
import copy

class CameraModel:
    def __init__(self, H, W, fx, fy, cx, cy):
        self.H = H
        self.W = W
        self.fx = fx
        self.fy = fy
        self.cx = cx
        self.cy = cy

# recover depth from rgb color
# https://dev.intelrealsense.com/docs/depth-image-compression-by-colorization-for-intel-realsense-depth-cameras
def rgb2depth(r, g, b, d_min=0.3, d_max=2.0):
    if b + g + r < 127: # fix
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
        print("Error.")
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
    depth_color = cv2.imread(depth_path, cv2.IMREAD_UNCHANGED)

    return rgb, depth_color


def recovery_depth(depth_color):
    depth_color_int32 = depth_color.astype(np.int32)
    H, W, _ = depth_color_int32.shape
    depth = np.zeros((H, W))
    for i in range(H):
        for j in range(W):
            depth[i, j] = rgb2depth(depth_color_int32[i, j, 0], depth_color_int32[i, j, 1], depth_color_int32[i, j, 2])
    
    return depth


def depth2pcd(depth, depth_camera):
    u, v = np.meshgrid(np.linspace(0, depth_camera.W-1, depth_camera.W),
                       np.linspace(0, depth_camera.H-1, depth_camera.H), indexing='xy')
    dirs = np.stack(
        [(u-depth_camera.cx)/depth_camera.fx,
         -(v-depth_camera.cy)/depth_camera.fy,
         np.ones_like(u)],
        -1
    )
    pcd_np = (dirs * np.tile(depth.reshape(depth_camera.H, depth_camera.W, 1), (1, 1, 3))).reshape(-1, 3)
    
    return pcd_np


def viz_pcd(pcd_list):
    o3d.visualization.draw_geometries(pcd_list)
    
    
def draw_registration_result(source_pcd, target_pcd, transformation):
    source_pcd_temp = copy.deepcopy(source_pcd)
    target_pcd_temp = copy.deepcopy(target_pcd)
    source_pcd_temp.paint_uniform_color([1, 0.706, 0])
    target_pcd_temp.paint_uniform_color([0, 0.651, 0.929])
    source_pcd_temp.transform(transformation)
    o3d.visualization.draw_geometries([source_pcd_temp, target_pcd_temp])
    
    
def detect_hand(pcd, depth_camera, thre_dis=0.03):
    target_pcd = pcd
    
    # set target pcd
    plane_n = 100
    u, v = np.meshgrid(np.linspace(-100, depth_camera.W + 100, plane_n), np.linspace(-100, depth_camera.H + 100, plane_n), indexing='xy')
    dirs = np.stack(
        [(u-depth_camera.cx)/depth_camera.fx,
         -(v-depth_camera.cy)/depth_camera.fy,
         np.ones_like(u)],
        -1
    )
    depth = np.full((plane_n, plane_n), 2.0)
    source_pcd_np = (dirs * np.tile(depth.reshape(plane_n, plane_n, 1), (1, 1, 3))).reshape(-1, 3)
    source_pcd = o3d.geometry.PointCloud()
    source_pcd.points = o3d.utility.Vector3dVector(source_pcd_np)
    
    threshold = 0.02
    trans_init = np.asarray([[1.0, 0.0, 0.0, 0.0],
                             [0.0, 1.0, 0.0, 0.0],
                             [0.0, 0.0, 1.0, 0.0],
                             [0.0, 0.0, 0.0, 1.0]])
    reg_p2l = o3d.pipelines.registration.registration_icp(
                source_pcd, target_pcd, threshold, trans_init,
                o3d.pipelines.registration.TransformationEstimationPointToPoint())
    
    #draw_registration_result(source_pcd, target_pcd, reg_p2l.transformation)
    source_pcd.transform(reg_p2l.transformation)

    kdtree = o3d.geometry.KDTreeFlann(source_pcd) # desk
    hand_flag = []
    #hand_idx = []
    for i in range(len(target_pcd.points)):
        [k, idx, _] = kdtree.search_radius_vector_3d(target_pcd.points[i], thre_dis)
        if k == 0:
            hand_flag.append(True)
            #hand_idx.append(i)
        else:
            hand_flag.append(False)
    
    #hand_pcd = target_pcd.select_by_index(hand_idx)
    #desk_pcd = target_pcd.select_by_index(hand_idx, invert=True)
    #viz_pcd([hand_pcd, desk_pcd])
    
    return hand_flag

def mask_rgbImage_fromDepth(rgb, mask):
    def crop_mask(mask, bottom=58, up=66, left=0, right=34):
        return mask[up:mask.shape[0]-bottom, left:mask.shape[1]-right]
    def crop_rgb(rgb, left=40):
        return rgb[:, left:, ]
    croped_mask = crop_mask(mask)
    croped_rgb = crop_rgb(rgb)
    
    mask_img = np.zeros((croped_mask.shape[0], croped_mask.shape[1], 3), np.uint8)
    mask_img[croped_mask] = (255, 255, 255)
    
    kernel = np.ones((3, 3), np.uint8)
    mask_img = cv2.erode(mask_img, kernel, iterations=3)
    mask_img = cv2.dilate(mask_img, kernel, iterations=20)

    rgb_small = cv2.resize(croped_rgb, (mask_img.shape[1], mask_img.shape[0]))
    masked_rgb = cv2.bitwise_and(rgb_small, mask_img)
    
    return masked_rgb


def check_cropSize(rgb, depth_color):
    bottom = 65
    up = 68
    left = 0
    right = 34
    croped_depth_color = depth_color[up:depth_color.shape[0]-bottom, left:depth_color.shape[1]-right, ]
    
    left_rgb = 40
    croped_rgb = rgb[:, left_rgb:, ]
    
    rgb_small = cv2.resize(croped_rgb, (croped_depth_color.shape[1], croped_depth_color.shape[0]))
    img_v = cv2.vconcat([rgb_small, croped_depth_color])
    cv2.imshow('img_v', cv2.resize(img_v, None, None, 0.5, 0.5))
    cv2.waitKey(0)
    img_h = cv2.hconcat([rgb_small, croped_depth_color])
    cv2.imshow('img_h', cv2.resize(img_h, None, None, 0.5, 0.5))
    cv2.waitKey(0)


def main():
    parser = argparse.ArgumentParser(prog='maskImage', description='mask RGB image from depth')
    parser.add_argument('rgb_video_path', type=str, help='rgb video path')
    parser.add_argument('depth_frames_path', type=str, help='depth frames path')
    parser.add_argument('output_path', type=str, help='output path')
    args = parser.parse_args()
    
    
    vc = cv2.VideoCapture(args.rgb_video_path)
    frame_cnt = int(vc.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = int(vc.get(cv2.CAP_PROP_FPS))
    print(f"frames:{frame_cnt}, fps:{fps}")

    for frame_idx in range(1, frame_cnt, 1000):
        # load rgb & depth
        vc.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        frame_pos = int(vc.get(cv2.CAP_PROP_POS_FRAMES)) 
        print(f"frame_pos:{frame_pos}/{frame_cnt}")
        is_image, rgb = vc.read()
        depth_path = os.path.join(args.depth_frames_path, f"{frame_idx:05}.jpg")
        depth_color = cv2.imread(depth_path, cv2.IMREAD_UNCHANGED)
    
        #check_cropSize(rgb, depth_color)
        #return 0
    
        # recovery depth from color coded depth image
        depth = recovery_depth(depth_color)
        
        # calc depth param
        depth_height, depth_width = depth.shape
        depth_FOV_H = 73.0 # 68 (RGB) 68/73=0.932 
        depth_FOV_V = 59.0 # 41.5 (RGB) 41.5/59=0.703
        depth_fx = (depth_width / 2.0) / np.tan(np.deg2rad(depth_FOV_H / 2.0))
        depth_fy = (depth_height / 2.0) / np.tan(np.deg2rad(depth_FOV_V / 2.0))
        depth_cx = (depth_width - 1) / 2.0
        depth_cy = (depth_height - 1) / 2.0
        depth_camera = CameraModel(depth_height, depth_width, depth_fx, depth_fy, depth_cx, depth_cy)
        
        # make point cloud
        pcd_np = depth2pcd(depth, depth_camera)
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(pcd_np)
        #viz_pcd(pcd)
        
        # detect hand
        hand_flag = detect_hand(pcd, depth_camera)
        
        # make mask
        depth_0_mask = depth > 0.2
        hand_mask = np.asarray(hand_flag).reshape(depth.shape[0], depth.shape[1])
        mask = depth_0_mask & hand_mask
        
        # mask rgb image
        masked_rgb = mask_rgbImage_fromDepth(rgb, mask)
        
        # write masked rgb image
        output_file = os.path.join(args.output_path, f"masked_{frame_idx:05}.jpg")
        cv2.imwrite(output_file, masked_rgb)
        
    vc.release()


if __name__ == "__main__":
    main()