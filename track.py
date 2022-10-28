import ros_numpy
import rospy
from sensor_msgs.msg import Image
from sensor_msgs.msg import PointCloud2
from cv_bridge import CvBridge, CvBridgeError
import os
import sys
import numpy as np
from pathlib import Path
import torch
import cv2
import message_filters
import tf
# limit the number of cpus used by high performance libraries
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"
FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]  # yolov5 strongsort root directory
WEIGHTS = ROOT / 'weights'
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH
if str(ROOT / 'yolov5') not in sys.path:
    sys.path.append(str(ROOT / 'yolov5'))  # add yolov5 ROOT to PATH
if str(ROOT / 'trackers' / 'strong_sort') not in sys.path:
    sys.path.append(str(ROOT / 'trackers' / 'strong_sort'))  # add strong_sort ROOT to PATH
if str(ROOT / 'trackers' / 'ocsort') not in sys.path:
    sys.path.append(str(ROOT / 'trackers' / 'ocsort'))  # add strong_sort ROOT to PATH
if str(ROOT / 'trackers' / 'strong_sort' / 'deep' / 'reid' / 'torchreid') not in sys.path:
    sys.path.append(str(ROOT / 'trackers' / 'strong_sort' / 'deep' / 'reid' / 'torchreid'))  # add strong_sort ROOT to PATH
ROOT = Path(os.path.relpath(ROOT, Path.cwd()))  # relative
from yolov5.models.common import DetectMultiBackend
from yolov5.utils.general import (LOGGER, check_img_size, non_max_suppression, scale_coords, check_requirements, print_args)
from yolov5.utils.torch_utils import select_device, time_sync
from yolov5.utils.plots import Annotator
from trackers.multi_tracker_zoo import create_tracker

class Track:
    def __init__(self) -> None:
        rospy.init_node('percept_fusion_tracker')
        check_requirements(requirements=ROOT / 'requirements.txt', exclude=('tensorboard', 'thop'))
        self.tf = tf.TransformBroadcaster()
        self.bridge = CvBridge()
        pass
    def listener(self):
        sync1 = message_filters.Subscriber('/kinect2_down/qhd/image_color',Image, queue_size=1000)
        sync2 = message_filters.Subscriber("/kinect2_down/qhd/points", PointCloud2)
        sync = message_filters.ApproximateTimeSynchronizer([sync1, sync2], 10, 1, allow_headerless=True)
        sync.registerCallback(self.sync_callback)
        rospy.spin()
    def sync_callback(self, image, depth):
        cv_image = self.bridge.imgmsg_to_cv2(image, "bgr8")
        self.yolo_track_pred(cv_image)
        for obejcts in self.predict_results.items():
            period = np.array(obejcts[1]['time'])
            time_bias = rospy.Time.now().to_sec() - 3
            valid_period = np.where(time_bias<period)
            obejcts[1]['time'] = np.array(obejcts[1]['time'])[valid_period].tolist()
            obejcts[1]['bbox'] = np.array(obejcts[1]['bbox'])[valid_period].tolist()

        pc = ros_numpy.numpify(depth)
        height = 540
        width = 960
        np_points = np.zeros((height * width, 3), dtype=np.float32)
        np_points[:, 0] = np.resize(pc['x'], height * width)
        np_points[:, 1] = np.resize(pc['y'], height * width)
        np_points[:, 2] = np.resize(pc['z'], height * width)
        for obejcts in self.predict_results.items():
            name = obejcts[0]
            if not obejcts[1]['time']:
                continue
            time = obejcts[1]['time'][-1]
            bbox = obejcts[1]['bbox'][-1]
            xmax =  bbox[2]
            xmin =  bbox[0]
            ymax =  bbox[3]
            ymin =  bbox[1]
            length = xmax - xmin
            height = ymax - ymin
            xlist = []
            ylist = []
            zlist = []
            for ix in range(int(xmin + length / 2), int(xmax - length / 3)):
                # print(ix)
                for iy in range(int(ymin + height / 3), int(ymax - height / 3)):
                    index = int((iy - 1) * width + ix)
                    # index = ix * 540 + iy
                    position = np_points[index]
                    if not np.isnan(position[0]):
                        xlist.append(position[0])
                    if not np.isnan(position[1]):
                        ylist.append(position[1])
                    if not np.isnan(position[2]):
                        zlist.append(position[2])
            if xlist and ylist and zlist:
                x = np.array(xlist)[np.where(abs(np.array(xlist))<4.5)].mean()
                y = np.array(ylist)[np.where(abs(np.array(ylist))<4.5)].mean()
                z = np.array(zlist)[np.where(abs(np.array(zlist))<4.5)].mean()
                self.tf.sendTransform((x, y, z), tf.transformations.quaternion_from_euler(0, 0, 0), rospy.Time.now(),
                                     name, 'kinect2_down_link')
    @torch.no_grad()
    def init_yolo_track(self,
        yolo_weights=WEIGHTS / 'yolov5m.pt',  # model.pt path(s),
        appearance_descriptor_weights=WEIGHTS / 'osnet_x0_25_msmt17.pt',  # model.pt path,
        tracking_method='strongsort',
        imgsz=(640, 640),  # inference size (height, width)
        device='0',  # cuda device, i.e. 0 or 0,1,2,3 or cpu
        half=False,  # use FP16 half-precision inference
        dnn=False,  # use OpenCV DNN for ONNX inference
):
        self.predict_results = {}
        self.device = select_device(device)
        # device = '0'
        self.model = DetectMultiBackend(yolo_weights, device=self.device, dnn=dnn, data=None, fp16=half)
        stride, names, pt = self.model.stride, self.model.names, self.model.pt
        imgsz = check_img_size(imgsz, s=stride)  # check image size
        nr_sources = 1
        vid_path, vid_writer, txt_path = [None] * nr_sources, [None] * nr_sources, [None] * nr_sources

        # Create as many strong sort instances as there are video sources
        self.tracker_list = []
        for i in range(nr_sources):
            tracker = create_tracker(tracking_method, appearance_descriptor_weights, self.device, half)
            self.tracker_list.append(tracker, )
        self.outputs = [None] * nr_sources

        # Run tracking
        self.model.warmup(imgsz=(1 if pt else nr_sources, 3, *imgsz))  # warmup
        dt, seen = [0.0, 0.0, 0.0, 0.0], 0
        self.curr_frames, self.prev_frames = [None] * nr_sources, [None] * nr_sources
    @torch.no_grad()
    def yolo_track_pred(self,source):
        from yolov5.utils.augmentations import letterbox
        im = letterbox(source, auto=self.model.pt)[0]  # padded resize
        im = im.transpose((2, 0, 1))[::-1]  # HWC to CHW, BGR to RGB
        im = np.ascontiguousarray(im)  # contiguous
        im = torch.from_numpy(im).to(self.device)
        im = im.float()  # uint8 to fp16/32
        im /= 255.0  # 0 - 255 to 0.0 - 1.0
        if len(im.shape) == 3:
            im = im[None]  # expand for batch dim
        # Inference
        # visualize = increment_path(save_dir / Path(path[0]).stem, mkdir=True) if visualize else False
        pred = self.model(im)
        # Apply NMS
        pred = non_max_suppression(pred, conf_thres=0.25, iou_thres=0.45, classes=None,  max_det=1000)
        seen = 0
        # Process detections
        
        for i, det in enumerate(pred):  # detections per image
            seen += 1
            annotator = Annotator(source, line_width=2, pil=not ascii)
            if det is not None and len(det):
                # Rescale boxes from img_size to im0 size
                det[:, :4] = scale_coords(im.shape[2:], det[:, :4], source.shape).round()  # xyxy
                s = ''
                # Print results
                for c in det[:, -1].unique():
                    n = (det[:, -1] == c).sum()  # detections per class
                    s += f"{n} {self.model.names[int(c)]}{'s' * (n > 1)}, "  # add to string

                # pass detections to strongsort
                t4 = time_sync()
                self.outputs[i] = self.tracker_list[i].update(det.cpu(), source)

                # draw boxes for visualization
                if len(self.outputs[i]) > 0:
                    for j, (output, conf) in enumerate(zip(self.outputs[i], det[:, 4])):
                        bboxes = output[0:4]
                        id = output[4]
                        cls = output[5]
                        hide_labels = False
                        hide_conf = False
                        hide_class = False

                        label = None if hide_labels else (f'{id} {self.model.names[id]}' if hide_conf else \
                            (f'{id} {conf:.2f}' if hide_class else f'{id} {self.model.names[cls]} {conf:.2f}'))
                        # annotator.box_label(bboxes, label)
                        if not self.predict_results.get(self.model.names[cls]+str(id)):
                            self.predict_results[self.model.names[cls]+str(id)] = {'bbox':[],'time':[]}
                        self.predict_results[self.model.names[cls]+str(id)]['bbox'].append(bboxes)
                        self.predict_results[self.model.names[cls]+str(id)]['time'].append(rospy.Time.now().to_sec())
            else:
                rospy.loginfo('No detections')
            # im0 = annotator.result()
            # cv2.imshow('yolo_track_predictions', im0)
            # cv2.waitKey(10)  # 1 millisecond

            self.prev_frames[i] = self.curr_frames[i]


if __name__ == "__main__":
    tracker = Track()
    tracker.init_yolo_track()
    tracker.listener()