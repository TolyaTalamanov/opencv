import cv2 as cv
import argparse
import numpy as np

def weight_path(model_path):
    return model_path.split('.')[0] + '.bin'


def union(a, b):
    x = min(a[0], b[0])
    y = min(a[1], b[1])
    w = max(a[0] + a[2], b[0] + b[2]) - x
    h = max(a[1] + a[3], b[1] + b[3]) - y
    return (x, y, w, h)


def intersection(a, b):
    x = max(a[0], b[0])
    y = max(a[1], b[1])
    w = min(a[0] + a[2], b[0] + b[2]) - x
    h = min(a[1] + a[3], b[1] + b[3]) - y
    if w < 0 or h < 0: return (0, 0, 0, 0)
    # if w < 0 or h < 0: return None
    return (x, y, w, h)


def eyeBox(face_rc, p1_x, p1_y, p2_x, p2_y, scale=1.8):
    up = np.array([face_rc[2], face_rc[3]])
    p1 = np.array([p1_x * up[0], p1_y * up[1]])
    p2 = np.array([p2_x * up[0], p2_y * up[1]])

    size = np.linalg.norm(p1 - p2)
    midpoint = (p1 + p2) / 2


    width  = scale * size
    height = width
    x = face_rc[0] + midpoint[0] - (width / 2)
    y = face_rc[1] + midpoint[1] - (height / 2)

    return (int(x), int(y), int(width), int(height))


def processPoses(angles_y, angles_p, angles_r):
    def processPoses_meta(desc_y, desc_p, desc_r):
        return cv.empty_array_desc()
    op = cv.gapi_op('custom.processPoses', processPoses_meta, angles_y, angles_p, angles_r)
    return op.getGArray(cv.gapi.CV_GMAT)


def processPoses_kernel(in_ys, in_ps, in_rs):
    out_poses = []
    sz = len(in_ys)
    for i in range(sz):
        out_poses.append(np.array([in_ys[i][0], in_ps[i][0], in_rs[i][0]]).T)

    return out_poses


def parseEyes(g_landm, g_roi, g_sz):
    def parseEyes_meta(landm_desc, roi_desc, sz_desc):
        return cv.empty_array_desc(), cv.empty_array_desc()

    op = cv.gapi_op('custom.parseEyes', parseEyes_meta, g_landm, g_roi, g_sz)
    gleft_eyes  = op.getGArray(cv.gapi.CV_RECT)
    gright_eyes = op.getGArray(cv.gapi.CV_RECT)
    return gleft_eyes, gright_eyes


def parseEyes_kernel(in_landm_per_face, in_face_rcs, frame_size):
    left_eyes  = []
    right_eyes = []
    num_faces = len(in_landm_per_face)
    surface = (0, 0, *frame_size)

    for i in range(num_faces):
        lm = in_landm_per_face[i]
        rc = in_face_rcs[i]
        left_eyes.append( intersection(surface, eyeBox(rc, lm[0], lm[1], lm[2], lm[3])))
        right_eyes.append(intersection(surface, eyeBox(rc, lm[4], lm[5], lm[6], lm[7])))

    return left_eyes, right_eyes



if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='This is an OpenCV-based version of Gaze Estimation example')

    parser.add_argument('--input',
            help='Path to the input video file')
    parser.add_argument('--facem',
            default='face-detection-retail-0005.xml',
            help='Path to OpenVINO face detection model (.xml)')
    parser.add_argument('--faced',
            default='CPU',
            help='Target device for the face detection (e.g. CPU, GPU, VPU, ...)')
    parser.add_argument('--headm',
            default='head-pose-estimation-adas-0001.xml',
            help='Path to OpenVINO head pose estimation model (.xml)')
    parser.add_argument('--headd',
            default='CPU',
            help='Target device for the head pose estimation inference (e.g. CPU, GPU, VPU, ...)')
    parser.add_argument('--landm',
            default='facial-landmarks-35-adas-0002.xml',
            help='Path to OpenVINO landmarks detector model (.xml)')
    parser.add_argument('--landd',
            default='CPU',
            help='Target device for the landmarks detector (e.g. CPU, GPU, VPU, ...)')
    parser.add_argument('--gazem',
            default='gaze-estimation-adas-0002.xml',
            help='Path to OpenVINO gaze vector estimaiton model (.xml)')
    parser.add_argument('--gazed',
            default='CPU',
            help='Target device for the gaze vector estimation inference (e.g. CPU, GPU, VPU, ...)')

    arguments = parser.parse_args()

    # Read image
    img = cv.imread(arguments.input)

    # Build the graph
    g_in = cv.GMat()

    # Detect faces
    face_inputs = cv.GInferInputs()
    face_inputs.setInput('input.1', g_in)
    face_outputs  = cv.gapi.infer('face-detection', face_inputs)
    faces = face_outputs.at('527')

    # Parse faces
    sz = cv.gapi.streaming.size(g_in)
    faces_rc = cv.gapi.parseSSD(faces, sz, 0.5, False, False)

    # Detect poses
    head_inputs = cv.GInferInputs()
    head_inputs.setInput('data', g_in)
    face_outputs = cv.gapi.infer('head-pose', faces_rc, head_inputs)
    angles_y = face_outputs.at('angle_y_fc')
    angles_p = face_outputs.at('angle_p_fc')
    angles_r = face_outputs.at('angle_r_fc')

    # Parse poses
    heads_pos = processPoses(angles_y, angles_p, angles_r)

    # Detect landmarks
    landmark_inputs = cv.GInferInputs()
    landmark_inputs.setInput('data', g_in)
    landmark_outputs  = cv.gapi.infer('facial-landmarks', landmark_inputs)
    landmark = landmark_outputs.at('align_fc3')

    # Parse landmarks
    left_eyes, right_eyes = parseEyes(landmark, faces_rc, sz)

    # Gaze estimation
    gaze_inputs = cv.GInferListInputs()
    gaze_inputs.setInput('left_eye_image'  , left_eyes)
    gaze_inputs.setInput('right_eye_image' , right_eyes)
    gaze_inputs.setInput('head_pose_angles', heads_pos)
    gaze_outputs  = cv.gapi.infer2('gaze-estimation', g_in, gaze_inputs)
    gaze_vectors  = gaze_outputs.at('gaze_vector')

    comp = cv.GComputation(cv.GIn(g_in), cv.GOut(gaze_vectors))

    face_net      = cv.gapi.ie.params('face-detection'  , arguments.facem, weight_path(arguments.facem), arguments.faced)
    head_pose_net = cv.gapi.ie.params('head-pose'       , arguments.headm, weight_path(arguments.headm), arguments.headd)
    landmarks_net = cv.gapi.ie.params('facial-landmarks', arguments.landm, weight_path(arguments.landm), arguments.landd)
    gaze_net      = cv.gapi.ie.params('gaze-estimation' , arguments.gazem, weight_path(arguments.gazem), arguments.gazed)

    kernels = cv.kernels((processPoses_kernel, 'custom.processPoses'),
                         (parseEyes_kernel   , 'custom.parseEyes'))

    nets = cv.gapi.networks(face_net, head_pose_net, landmarks_net, gaze_net)

    out = comp.apply(cv.gin(img), args=cv.compile_args(kernels, nets))
    print(out)
