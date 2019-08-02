import argparse
import time
from sys import platform

from models import *
from utils.datasets import *
from utils.utils import *

class CParser():
    def __init__(self):
        self.cfg = "cfg/yolov3.cfg"
        self.data_cfg = 'data/coco.data'
        self.weights = 'weights/yolov3.weights'
        self.images = "data/samples"
        self.img_size = 416
        self.conf_thres = 0.4
        self.nms_thres = 0.4

class CDetector():
    def __init__(self):
        parser = argparse.ArgumentParser()
        parser.add_argument('--cfg', type=str, default='cfg/yolov3.cfg', help='cfg file path')
        parser.add_argument('--data-cfg', type=str, default='data/coco.data', help='coco.data file path')
        parser.add_argument('--weights', type=str, default='weights/yolov3.weights', help='path to weights file')
        parser.add_argument('--images', type=str, default='data/samples', help='path to images')
        parser.add_argument('--img-size', type=int, default=416, help='size of each image dimension')
        parser.add_argument('--conf-thres', type=float, default=0.4, help='object confidence threshold')
        parser.add_argument('--nms-thres', type=float, default=0.4, help='iou threshold for non-maximum suppression')
        # self._opt = parser.parse_args()
        self._opt = CParser()

        self._device = torch_utils.select_device()
        self._model = Darknet(self._opt.cfg, self._opt.img_size)
        if self._opt.weights.endswith('.pt'):  # pytorch format
            self._model.load_state_dict(torch.load(self._opt.weights, map_location=self._device)['model'])
        else:  # darknet format
            _ = load_darknet_weights(self._model, self._opt.weights)
        self._model.to(self._device).eval()

    def make_torch_image(self, localimage, img_size=416):
        self._height = img_size
        img0 = cv2.imread(localimage)  # BGR
        # Padded resize
        img, *_ = letterbox(img0, new_shape=img_size)

        # Normalize RGB
        img = img[:, :, ::-1].transpose(2, 0, 1)  # BGR to RGB
        img = np.ascontiguousarray(img, dtype=np.float32)  # uint8 to float32
        img /= 255.0  # 0 - 255 to 0.0 - 1.0
        return img, img0

    def do(self, localimage):
        objects = []
        classes = load_classes(parse_data_cfg("data/coco.data")['names'])

        img, img0 = self.make_torch_image(localimage)
        img = torch.from_numpy(img).unsqueeze(0).to(self._device)
        with torch.no_grad():
            pred, _ = self._model(img)
        detections = non_max_suppression(pred, self._opt.conf_thres, self._opt.nms_thres)[0]
        if detections is not None and len(detections) > 0:
            # print(detections)
            detections[:, :4] = scale_coords(img.shape[2:], detections[:, :4], img0.shape).round()
            # print(detections)
            for c in detections[:, -1].unique():
                # print("c : ", c)
                n = (detections[:, -1] == c).sum()
                print('%g %ss' % (n, classes[int(c)]), end=', ')
                objects.append((classes[int(c)], n.item()))
        return objects


def mydetect_test(localimage):
    det = CDetector()
    ret = det.do(localimage)
    print(ret)
    return ret

def detect(
        cfg,
        data_cfg,
        weights,
        images,
        output='output',  # output folder
        img_size=416,
        conf_thres=0.5,
        nms_thres=0.5,
        save_txt=False,
        save_images=True,
        webcam=False
):
    device = torch_utils.select_device()
    if os.path.exists(output):
        shutil.rmtree(output)  # delete output folder
    os.makedirs(output)  # make new output folder

    # Initialize model
    model = Darknet(cfg, img_size)

    # Load weights
    if weights.endswith('.pt'):  # pytorch format
        model.load_state_dict(torch.load(weights, map_location=device)['model'])
    else:  # darknet format
        _ = load_darknet_weights(model, weights)

    model.to(device).eval()

    # Set Dataloader
    vid_path, vid_writer = None, None
    if webcam:
        save_images = False
        dataloader = LoadWebcam(img_size=img_size)
    else:
        dataloader = LoadImages(images, img_size=img_size)

    # Get classes and colors
    classes = load_classes(parse_data_cfg(data_cfg)['names'])
    colors = [[random.randint(0, 255) for _ in range(3)] for _ in range(len(classes))]

    for i, (path, img, im0, vid_cap) in enumerate(dataloader):
        t = time.time()
        save_path = str(Path(output) / Path(path).name)

        # Get detections
        img = torch.from_numpy(img).unsqueeze(0).to(device)
        if ONNX_EXPORT:
            torch.onnx.export(model, img, 'weights/model.onnx', verbose=True)
            return
        pred, _ = model(img)
        detections = non_max_suppression(pred, conf_thres*0.3, nms_thres*0.3)[0]
        if detections is not None and len(detections) > 0:
            # Rescale boxes from 416 to true image size
            scale_coords(img_size, detections[:, :4], im0.shape).round()

            # Print results to screen
            for c in detections[:, -1].unique():
                n = (detections[:, -1] == c).sum()
                print('%g %ss' % (n, classes[int(c)]), end=', ')

            # Draw bounding boxes and labels of detections
            for *xyxy, conf, cls_conf, cls in detections:
                if save_txt:  # Write to file
                    with open(save_path + '.txt', 'a') as file:
                        file.write(('%g ' * 6 + '\n') % (*xyxy, cls, conf))

                # Add bbox to the image
                label = '%s %.2f' % (classes[int(cls)], conf)
                print(label)
                plot_one_box(xyxy, im0, label=label, color=colors[int(cls)])

        print('Done. (%.3fs)' % (time.time() - t))

        if webcam:  # Show live webcam
            cv2.imshow(weights, im0)

        if save_images:  # Save generated image with detections
            if dataloader.mode == 'video':
                if vid_path != save_path:  # new video
                    vid_path = save_path
                    if isinstance(vid_writer, cv2.VideoWriter):
                        vid_writer.release()  # release previous video writer
                    width = int(vid_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                    height = int(vid_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                    fps = vid_cap.get(cv2.CAP_PROP_FPS)
                    vid_writer = cv2.VideoWriter(save_path, cv2.VideoWriter_fourcc(*'avc1'), fps, (width, height))
                vid_writer.write(im0)

            else:
                cv2.imwrite(save_path, im0)

    if save_images and platform == 'darwin':  # macos
        os.system('open ' + output + ' ' + save_path)


def test():
    parser = argparse.ArgumentParser()
    parser.add_argument('--cfg', type=str, default='cfg/yolov3-spp.cfg', help='cfg file path')
    parser.add_argument('--data-cfg', type=str, default='data/coco.data', help='coco.data file path')
    parser.add_argument('--weights', type=str, default='weights/yolov3-spp.weights', help='path to weights file')
    parser.add_argument('--images', type=str, default='data/samples', help='path to images')
    parser.add_argument('--img-size', type=int, default=416, help='size of each image dimension')
    parser.add_argument('--conf-thres', type=float, default=0.5, help='object confidence threshold')
    parser.add_argument('--nms-thres', type=float, default=0.5, help='iou threshold for non-maximum suppression')
    opt = parser.parse_args()
    print(opt)

    with torch.no_grad():
        detect(
            opt.cfg,
            opt.data_cfg,
            opt.weights,
            opt.images,
            img_size=opt.img_size,
            conf_thres=opt.conf_thres,
            nms_thres=opt.nms_thres
        )


if __name__ == '__main__':
    mydetect_test("d:/1563186799886973631.jpg")
