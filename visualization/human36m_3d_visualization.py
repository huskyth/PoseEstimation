import cv2
import numpy as np
import transformations

from tqdm import tqdm
import time


def read_yml(fs, key, dt='mat'):
    if dt == 'mat':
        outputs = fs.getNode(key).mat()
    else:
        n = fs.getNode(key)
        results = []
        for i in range(n.size()):
            val = n.at(i).string()
            if val == '':
                val = str(int(n.at(i).real()))
            if val != 'none':
                results.append(val)
        outputs = results
    return outputs

def get_parameters(intri_path, extri_path):
    fs_in = cv2.FileStorage(intri_path, cv2.FILE_STORAGE_READ)
    fs_ex = cv2.FileStorage(extri_path, cv2.FILE_STORAGE_READ)

    camnames = read_yml(fs_in, 'names', 'list')
    cameras = {}
    for key in camnames:
        cam = {}
        cam['K'] = read_yml(fs_in, 'K_{}'.format(key))
        cam['dist'] = read_yml(fs_in, 'dist_{}'.format(key))
        cam['R'] = read_yml(fs_ex, 'Rot_{}'.format(key))
        cam['T'] = read_yml(fs_ex, 'T_{}'.format(key))
        cameras[key] = cam
    return cameras


class Camera:
    def __init__(self, R, t, K, dist=None, name=""):
        self.R = R.copy()
        self.t = t.copy()
        self.K = K.copy()
        self.dist = dist

        self.name = name

    def update_after_crop(self, bbox):
        left, upper, right, lower = bbox

        cx, cy = self.K[0, 2], self.K[1, 2]

        new_cx = cx - left
        new_cy = cy - upper

        self.K[0, 2], self.K[1, 2] = new_cx, new_cy

    def update_after_resize(self, image_shape, new_image_shape):
        height, width = image_shape
        new_width, new_height = new_image_shape

        fx, fy, cx, cy = self.K[0, 0], self.K[1, 1], self.K[0, 2], self.K[1, 2]

        new_fx = fx * (new_width / width)
        new_fy = fy * (new_height / height)
        new_cx = cx * (new_width / width)
        new_cy = cy * (new_height / height)

        self.K[0, 0], self.K[1, 1], self.K[0, 2], self.K[1, 2] = new_fx, new_fy, new_cx, new_cy

    @property
    def projection(self):
        return self.K.dot(self.extrinsics)

    @property
    def extrinsics(self):
        return np.hstack([self.R, self.t])


def generate_grid_mesh(start, end, step=1.0):
    num_point_per_line = int((end - start) // step + 1)
    its = np.linspace(start, end, num_point_per_line)
    line = []
    color = []
    common_line_color = [192, 192, 192]
    for i in range(num_point_per_line):
        line.append([its[0], its[i], 0, its[-1], its[i], 0])
        if its[i] == 0:
            color.append([0, 255, 0])
        else:
            color.append(common_line_color)

    for i in range(num_point_per_line):
        line.append([its[i], its[-1], 0, its[i], its[0], 0])
        if its[i] == 0:
            color.append([0, 0, 255])
        else:
            color.append(common_line_color)

    return np.array(line, dtype=np.float32), np.array(color, dtype=np.uint8)


def euclidean_to_homogeneous(points):
    if isinstance(points, np.ndarray):
        return np.hstack([points, np.ones((len(points), 1))])
    else:
        raise TypeError("Works only with numpy arrays")


def homogeneous_to_euclidean(points):
    if isinstance(points, np.ndarray):
        return (points.T[:-1] / points.T[-1]).T
    else:
        raise TypeError("Works only with numpy arrays")


def projection_to_2d_plane(vertices, projection_matrix, view_matrix=None, scale=None):
    if view_matrix is not None:
        vertices = (homogeneous_to_euclidean(
            (euclidean_to_homogeneous(vertices) @ view_matrix.T) @ projection_matrix.T)[:, :2]) * scale

        vertices[:, 1] = scale - vertices[:, 1]
        vertices[:, 0] = vertices[:, 0] + scale
    else:
        vertices = euclidean_to_homogeneous(vertices) @ projection_matrix.T
        vertices = homogeneous_to_euclidean(vertices)
    return vertices.astype(np.int32)


def look_at(eye, center, up):
    f = unit_vector(center - eye)
    u = unit_vector(up)
    s = unit_vector(np.cross(f, u))
    u = np.cross(s, f)

    result = transformations.identity_matrix()
    result[:3, 0] = s
    result[:3, 1] = u
    result[:3, 2] = -f
    result[3, 0] = -np.dot(s, eye)
    result[3, 1] = -np.dot(u, eye)
    result[3, 2] = np.dot(f, eye)
    return result.T


def unit_vector(vector):
    return vector / np.linalg.norm(vector)


def update_camera_vectors():
    global front
    front_temp = np.zeros((3,))
    front_temp[0] = np.cos(np.radians(yaw)) * np.cos(np.radians(pitch))
    front_temp[1] = np.sin(np.radians(pitch))
    front_temp[2] = np.sin(np.radians(yaw)) * np.cos(np.radians(pitch))
    front = unit_vector(front_temp)
    global right
    right = unit_vector(np.cross(front, world_up))


camera_vertices = np.array([[0, 0, 0], [-1, -1, 2],
                            [0, 0, 0], [1, 1, 2],
                            [0, 0, 0], [1, -1, 2],
                            [0, 0, 0], [-1, 1, 2],
                            [-1, 1, 2], [-1, -1, 2],
                            [-1, -1, 2], [1, -1, 2],
                            [1, -1, 2], [1, 1, 2],
                            [1, 1, 2], [-1, 1, 2]], dtype=np.float32)

human36m_connectivity_dict = [(0, 1), (1, 2), (2, 6), (5, 4), (4, 3), (3, 6), (6, 7), (7, 8), (8, 16), (9, 16), (8, 12),
                              (11, 12), (10, 11), (8, 13), (13, 14), (14, 15)]

multiview_data = np.load("/home/zjlab/lizao/human3.6-small/extra/human36m-multiview-labels-GTbboxes.npy", allow_pickle=True).tolist()
subject_name, camera_name, action_name, camera_configs, labels = multiview_data['subject_names'], multiview_data[
    'camera_names'], multiview_data['action_names'], multiview_data['cameras'], multiview_data['table']

camera_name = [str(i) for i, c in enumerate(camera_name)]

# subject_name ['S1', 'S5', 'S6', 'S7', 'S8', 'S9', 'S11']
# action_name ['Directions-1', 'Directions-2', 'Discussion-1', 'Discussion-2', 'Eating-1', 'Eating-2', 'Greeting-1', 'Greeting-2', 'Phoning-1', 'Phoning-2', 'Posing-1', 'Posing-2', 'Purchases-1', 'Purchases-2', 'Sitting-1', 'Sitting-2', 'SittingDown-1', 'SittingDown-2', 'Smoking-1', 'Smoking-2', 'TakingPhoto-1', 'TakingPhoto-2', 'Waiting-1', 'Waiting-2', 'Walking-1', 'Walking-2', 'WalkingDog-1', 'WalkingDog-2', 'WalkingTogether-1', 'WalkingTogether-2']


# specific_subject = "S9"
# specific_action = "WalkingDog-2"
# mask_subject = labels['subject_idx'] == subject_name.index(specific_subject)
# actions = [action_name.index(specific_action)]
# mask_actions = np.isin(labels['action_idx'], actions)
# mask_subject = mask_subject & mask_actions
# indices = []
# indices.append(np.nonzero(mask_subject)[0])
# specific_label = labels[np.concatenate(indices)]
# specific_3d_skeleton = specific_label['keypoints']
# specific_3d_skeleton = np.load("/home/ubuntu/Pavel/repos/real-time-pose-estimation/temp_folder/inference_pipeline_2cam/test_3d_output.npy")
# specific_3d_skeleton = np.load("/home/ubuntu/Pavel/data/keypoints_to_compare/test_3d_output_v134.npy")
# specific_3d_skeleton = np.load("/root/repos/real-time-pose-estimation/temp_folder/inference_pipeline_no_det/test_3d_output_no_det.npy")
# specific_3d_skeleton = np.load("/home/zjlab/Pavel/repos/rtpe_temp/imu_kpts_convert.npy")
# specific_3d_skeleton = np.load('/home/zjlab/Pavel/data/keypoints_to_compare/test_3d_output_v134.npy')

specific_3d_skeleton = np.load("/home/zjlab/dataset/video_paper_10.30/video5/3D-Pose.npy")
# specific_3d_skeleton = np.load("/home/ubuntu/Pavel/repos/rtpe/synchronize/imu_9.8/imu_2023_10_13_patient1.npy")

specific_3d_skeleton_2 = None

# specific_3d_skeleton = np.load("/home/ubuntu/Pavel/repos/rtpe/synchronize/cv_9.8/xxy_v3_1-2cams.npy")
# specific_3d_skeleton_2 = np.load("/home/zjlab/lizao/from_Pavel/imu_yijia0.npy")
# specific_3d_skeleton_2 = np.load("/home/ubuntu/Pavel/repos/rtpe/synchronize/imu_9.8/imu_xiyan3.npy")


# camera_intri = "/home/ubuntu/Pavel/data/calib4c_9_08/intri.yml"
# camera_extri = "/home/ubuntu/Pavel/data/calib4c_9_08/extri_be.yml"
# cameras_all = get_parameters(camera_intri, camera_extri)

# print(cameras_all)

# specific_camera_config = [
#     Camera(cameras_all[f'0{v}']['R'], cameras_all[f'0{v}']['T'], cameras_all[f'0{v}']['K']) for v in range(1, 5)
# ]


# specific_3d_skeleton = np.load('D:/3d/code/VideoTo3dPoseAndBvh/test_3d_output_old.npy')


# R1 = np.array([[0.235684, 0.970691, -0.047027], [0.380989, -0.136804, -0.914403], [-0.894036, 0.197594, -0.402065]])
# t1 = np.array([[0.280419, 0.576737, 2.790456]])
# K1 = np.array([[1005.963003, 0.000000, 976.153958],
#       [0.000000, 1003.991297, 576.992033],
#       [0.000000, 0.000000, 1.000000]])
# dist1 = [-0.050492, 0.080261, 0.001106, 0.006625, 0.000000]

# R2 = np.array([[-0.413168, 0.909910, -0.036819], [0.276664, 0.086901, -0.957029], [-0.867611, -0.405600, -0.287644]])
# t2 = np.array([[-0.886750, 0.921617, 2.741842]])
# K2 = np.array([[1006.505518, 0.000000, 1006.348360],
#       [0.000000, 1003.046864, 526.469265],
#       [0.000000, 0.000000, 1.000000]])
# dist2 = [-0.053939, 0.105380, 0.001648, 0.002457, 0.000000]

# specific_camera_config = [
#     Camera(R1, t1, K1),
#     Camera(R2, t2, K2)
# ]
# camera_name = [str(i) for i in range(2)]


specific_camera_config = camera_configs[subject_name.index("S9")]
print(specific_camera_config["R"][0])
print(specific_camera_config["t"][0])
print(specific_camera_config["K"][0])
specific_camera_config = [
    Camera(specific_camera_config["R"][i], specific_camera_config["t"][i], specific_camera_config["K"][i]) for i in
    range(len(camera_name))]

# first person setup
yaw = -125
pitch = -15
world_up = np.array([0.0, 1.0, 0.0])
position = np.array([5000, 2500, 7557])
front = np.array([0.0, 0.0, -1.0])
right = np.array([0.0, 0.0, 0.0])

grid_vertices, grid_color = generate_grid_mesh(-4, 4, step=1)
grid_vertices = grid_vertices.reshape(-1, 3)

rorate_x_90 = transformations.rotation_matrix(np.radians(-90), (1, 0, 0))

frame_size = 900
original_video_frame_size = 1000
frame = np.zeros([frame_size, frame_size])

for i in range(len(camera_name)):
    specific_camera_config[i].update_after_resize((original_video_frame_size,) * 2,
                                                  (frame_size,) * 2)

update_camera_vectors()
view_matrix = look_at(position, position + front, world_up)

projection_matrix = np.array([[2.41421, 0, 0, 0],
                              [0, 2.41421, 0, 0],
                              [0, 0, -1, -0.2],
                              [0, 0, -1, 0]], dtype=np.float32)

o_view_matrix = view_matrix.copy()
o_projection_matrix = projection_matrix.copy()

total_frame = specific_3d_skeleton.shape[0]
print(total_frame)
frame_index = 0

view_camera_index = -1
frames = []
prev = specific_3d_skeleton[0].reshape(-1, 3)
while True:
    if frame_index == total_frame:
        break
        frame_index = 0

    # if frame_index % 4 != 0:  ## 4 flame show 1
    #     frame_index += 1
    #     continue


    frame = np.full((frame_size, frame_size, 3), 255, dtype=np.uint8) # np.zeros([frame_size, frame_size, 3])
    if view_camera_index >= 0:
        view_matrix = None
        projection_matrix = specific_camera_config[view_camera_index].projection
    else:
        view_matrix = o_view_matrix
        projection_matrix = o_projection_matrix

    grid_vertices_project = grid_vertices @ (np.eye(3) if view_matrix is None else rorate_x_90[:3, :3].T)
    grid_vertices_project = grid_vertices_project @ transformations.scale_matrix(650)[:3, :3].T
    grid_vertices_project = projection_to_2d_plane(grid_vertices_project, projection_matrix, view_matrix,
                                                   int(frame_size / 2)).reshape(-1, 4)

    # draw line
    for index, line in enumerate(grid_vertices_project):
        cv2.line(frame, (line[0], line[1]), (line[2], line[3]), grid_color[index].tolist())

    # draw camera
    for camera_index, conf in enumerate(specific_camera_config):
        if view_camera_index == camera_index:
            continue
        m_rt = transformations.identity_matrix()
        r = np.array(conf.R, dtype=np.float32).T
        m_rt[:-1, -1] = -r @ np.array(conf.t, dtype=np.float32).squeeze()
        m_rt[:-1, :-1] = r

        m_s = transformations.identity_matrix() * 200
        m_s[3, 3] = 1

        camera_vertices_convert = homogeneous_to_euclidean(
            euclidean_to_homogeneous(camera_vertices) @ (
                    (np.eye(4) if view_matrix is None else rorate_x_90) @ m_rt @ m_s).T)

        camera_vertices_convert = projection_to_2d_plane(camera_vertices_convert, projection_matrix, view_matrix,
                                                         int(frame_size / 2))
        camera_vertices_convert = camera_vertices_convert.reshape(-1, 4)
        for index, line in enumerate(camera_vertices_convert):
            cv2.line(frame, (line[0], line[1]), (line[2], line[3]), (0, 153, 255), thickness=1)
        cv2.putText(frame, camera_name[camera_index],
                    (camera_vertices_convert[1, 0], camera_vertices_convert[1, 1] - 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255))


    specific_3d_skeleton_project = specific_3d_skeleton[frame_index].reshape(-1, 3)

    specific_3d_skeleton_project *= 1000

    # print(specific_3d_skeleton_project)

    # ttt = sum(specific_3d_skeleton_project - prev)
    # threshold = 5000
    # print(ttt)
    # print(abs(ttt)[0] > threshold or abs(ttt)[1] > threshold or abs(ttt)[2] > threshold)
    # if abs(ttt)[0] > threshold or abs(ttt)[1] > threshold or abs(ttt)[2] > threshold:
    #     specific_3d_skeleton_project = prev
    # # print('*'*30)
    # # print(specific_3d_skeleton_project)
    # # print(sum(specific_3d_skeleton_project))
    # # print(sum(specific_3d_skeleton_project - prev) > 0)
    # prev = specific_3d_skeleton_project
    specific_3d_skeleton_project = specific_3d_skeleton_project @ (
        np.eye(3) if view_matrix is None else rorate_x_90[:3, :3]).T
    specific_3d_skeleton_project = specific_3d_skeleton_project @ np.eye(3, dtype=np.float32) * 1
    specific_3d_skeleton_project = projection_to_2d_plane(specific_3d_skeleton_project, projection_matrix, view_matrix,
                                                          int(frame_size / 2)).reshape(17, 2)
    for c in human36m_connectivity_dict:
        cv2.line(frame, (*specific_3d_skeleton_project[c[0]],), (*specific_3d_skeleton_project[c[1]],),
                 (100, 155, 255), thickness=2)
        cv2.circle(frame, (*specific_3d_skeleton_project[c[0]],), 3, (0, 0, 255), -1)
        cv2.circle(frame, (*specific_3d_skeleton_project[c[1]],), 3, (0, 0, 255), -1)

    if specific_3d_skeleton_2 is not None:
        specific_3d_skeleton_project = specific_3d_skeleton_2[frame_index - 27].reshape(-1, 3)

        specific_3d_skeleton_project *= 1000

        specific_3d_skeleton_project = specific_3d_skeleton_project @ (
            np.eye(3) if view_matrix is None else rorate_x_90[:3, :3]).T
        specific_3d_skeleton_project = specific_3d_skeleton_project @ np.eye(3, dtype=np.float32) * 1
        specific_3d_skeleton_project = projection_to_2d_plane(specific_3d_skeleton_project, projection_matrix, view_matrix,
                                                            int(frame_size / 2)).reshape(17, 2)
        for c in human36m_connectivity_dict:
            cv2.line(frame, (*specific_3d_skeleton_project[c[0]],), (*specific_3d_skeleton_project[c[1]],),
                    (155, 100, 0), thickness=2)
            cv2.circle(frame, (*specific_3d_skeleton_project[c[0]],), 3, (0, 255, 0), -1)
            cv2.circle(frame, (*specific_3d_skeleton_project[c[1]],), 3, (0, 255, 0), -1)


    # time.sleep(0.15)

    # if frame_index > 1000 and frame_index % 5 == 0:
    frames.append(np.uint8(frame))
    # print(frame.dtype)
    # exit()
    frame_index += 1
    cv2.imshow("Directions-1", frame)
    wkey = cv2.waitKey(50)
    if wkey == ord('1'):
        view_camera_index += 1
        if view_camera_index == 4:
            view_camera_index = -1

    if wkey == ord('q'):
        break


size = (frame_size, frame_size)
fps = 15
    
# Below VideoWriter object will create
# a frame of above defined The output 
# is stored in 'filename.avi' file.
out_fn = "/home/zjlab/dataset/video_paper_10.30/video5/3D-Pose5.avi"
result = cv2.VideoWriter(out_fn,
                        cv2.VideoWriter_fourcc(*'DIVX'),
                        fps, size)
for i, frame in enumerate(frames):
    result.write(frame)
result.release()