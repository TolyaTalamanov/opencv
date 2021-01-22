import cv2 as cv
import argparse
import numpy as np

def weight_path(model_path):
    return model_path.split('.')[0] + '.bin'

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
    sz = cv.gapi.streaming.size(g_in)
    faces_rc = cv.gapi.parseSSD(faces, sz, 0.5, False, False)

    # Detect poses
    head_inputs = cv.GInferInputs()
    head_inputs.setInput('data', g_in)
    face_outputs = cv.gapi.infer('head-pose', faces_rc, head_inputs)
    angles_y = face_outputs.at('angle_y_fc')
    angles_p = face_outputs.at('angle_p_fc')
    angles_r = face_outputs.at('angle_r_fc')

    # # Detect landmarks
    # lm_inputs = cv.GInferInputs()
    # lm_inputs.setInput('data', g_in)
    # lm_outputs = cv.gapi.infer('facial-landmarks', lm_inputs)
    # landmarks = lm_outputs.at('align_fc3')

    # comp = cv.GComputation(cv.GIn(g_in), cv.GOut(landmarks))
    comp = cv.GComputation(cv.GIn(g_in), cv.GOut(angles_r))

    face_net      = cv.gapi.ie.params('face-detection'  , arguments.facem, weight_path(arguments.facem), arguments.faced)
    head_pose_net = cv.gapi.ie.params('head-pose'       , arguments.headm, weight_path(arguments.headm), arguments.headd)
    landmarks_net = cv.gapi.ie.params('facial-landmarks', arguments.landm, weight_path(arguments.landm), arguments.landd)
    gaze_net      = cv.gapi.ie.params('gaze-estimation' , arguments.landm, weight_path(arguments.landm), arguments.landd)

    out = comp.apply(cv.gin(img), args=cv.compile_args(
        cv.gapi.networks(face_net, head_pose_net, landmarks_net, gaze_net)))
    print(out)


    # sz = cv.GOpaqueT(cv.gapi.CV_SIZE)
    # cv::GOpaque<cv::Size> sz = custom::Size::on(in); // FIXME
    # cv::GArray<cv::Rect> faces_rc = custom::ParseSSD::on(faces, sz, true);
    # cv::GArray<cv::GMat> angles_y, angles_p, angles_r;
    # std::tie(angles_y, angles_p, angles_r) = cv::gapi::infer<custom::HeadPose>(faces_rc, in);
    # cv::GArray<cv::GMat> heads_pos = custom::ProcessPoses::on(angles_y, angles_p, angles_r);
    # cv::GArray<cv::GMat> landmarks = cv::gapi::infer<custom::Landmarks>(faces_rc, in);
    # cv::GArray<cv::Rect> left_eyes, right_eyes;
    # std::tie(left_eyes, right_eyes) = custom::ParseEyes::on(landmarks, faces_rc, sz);
    # cv::GArray<cv::GMat> gaze_vectors = cv::gapi::infer2<custom::Gaze>( in
                                                                      # , left_eyes
                                                                      # , right_eyes
                                                                      # , heads_pos);
