import cv2

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

    
