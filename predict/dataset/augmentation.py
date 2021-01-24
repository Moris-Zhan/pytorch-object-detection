import numpy as np
from random import uniform
import cv2
import pickle

# classes = ['aeroplane', 'bicycle', 'bird', 'boat', 'bottle', 'bus', 'car', 'cat', 'chair', 'cow',
#                         'diningtable', 'dog', 'horse', 'motorbike', 'person', 'pottedplant', 'sheep', 'sofa', 'train',
#                         'tvmonitor']  

# classes = ["person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck", "boat",
#            "traffic light", "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat", "dog",
#            "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella",
#            "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball", "kite",
#            "baseball bat", "baseball glove", "skateboard", "surfboard", "tennis racket", "bottle",
#            "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange",
#            "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair", "couch", "potted plant",
#            "bed", "dining table", "toilet", "tv", "laptop", "mouse", "remote", "keyboard", "cell phone",
#            "microwave", "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase", "scissors",
#            "teddy bear", "hair drier", "toothbrush"]   
# 
classes = ['aquarium', 'bottle', 'bowl', 'box', 'bucket', 'plastic_bag', 'plate', 'styrofoam', 'tire', 'toilet',
                        'tub', 'washing_machine', 'water_tower']                      

class Compose(object):

    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, data):
        for function_ in self.transforms:
            data = function_(data)
        return data


class Crop(object):

    def __init__(self, max_crop=0.1):
        super().__init__()
        self.max_crop = max_crop

    def __call__(self, data):
        image, label = data
        height, width = image.shape[:2]
        xmin = width
        ymin = height
        xmax = 0
        ymax = 0
        for lb in label:
            xmin = min(xmin, lb[0])
            ymin = min(ymin, lb[1])
            xmax = max(xmax, lb[2])
            ymax = max(ymax, lb[2])
        cropped_left = uniform(0, self.max_crop)
        cropped_right = uniform(0, self.max_crop)
        cropped_top = uniform(0, self.max_crop)
        cropped_bottom = uniform(0, self.max_crop)
        new_xmin = int(min(cropped_left * width, xmin))
        new_ymin = int(min(cropped_top * height, ymin))
        new_xmax = int(max(width - 1 - cropped_right * width, xmax))
        new_ymax = int(max(height - 1 - cropped_bottom * height, ymax))

        image = image[new_ymin:new_ymax, new_xmin:new_xmax, :]
        label = [[lb[0] - new_xmin, lb[1] - new_ymin, lb[2] - new_xmin, lb[3] - new_ymin, lb[4]] for lb in label]

        return image, label


class VerticalFlip(object):

    def __init__(self, prob=0.5):
        super().__init__()
        self.prob = prob

    def __call__(self, data):
        image, label = data
        if uniform(0, 1) >= self.prob:
            image = cv2.flip(image, 1)
            width = image.shape[1]
            label = [[width - lb[2], lb[1], width - lb[0], lb[3], lb[4]] for lb in label]
        return image, label


class HSVAdjust(object):

    def __init__(self, hue=30, saturation=1.5, value=1.5, prob=0.5):
        super().__init__()
        self.hue = hue
        self.saturation = saturation
        self.value = value
        self.prob = prob

    def __call__(self, data):

        def clip_hue(hue_channel):
            hue_channel[hue_channel >= 360] -= 360
            hue_channel[hue_channel < 0] += 360
            return hue_channel

        image, label = data
        adjust_hue = uniform(-self.hue, self.hue)
        adjust_saturation = uniform(1, self.saturation)
        if uniform(0, 1) >= self.prob:
            adjust_saturation = 1 / adjust_saturation
        adjust_value = uniform(1, self.value)
        if uniform(0, 1) >= self.prob:
            adjust_value = 1 / adjust_value
        image = image.astype(np.float32) / 255
        image = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
        image[:, :, 0] += adjust_hue
        image[:, :, 0] = clip_hue(image[:, :, 0])
        image[:, :, 1] = np.clip(adjust_saturation * image[:, :, 1], 0.0, 1.0)
        image[:, :, 2] = np.clip(adjust_value * image[:, :, 2], 0.0, 1.0)

        image = cv2.cvtColor(image, cv2.COLOR_HSV2RGB)
        image = (image * 255).astype(np.float32)

        return image, label


class Resize(object):

    def __init__(self, image_size):
        super().__init__()
        self.image_size = image_size

    def __call__(self, data):
        colors = pickle.load(open("dataset//pallete", "rb"))
        image, label = data
        height, width = image.shape[:2]
        image = cv2.resize(image, (self.image_size, self.image_size))
        width_ratio = float(self.image_size) / width
        height_ratio = float(self.image_size) / height
        new_label = []
        for lb in label:
            resized_xmin = lb[0] * width_ratio
            resized_ymin = lb[1] * height_ratio
            resized_xmax = lb[2] * width_ratio
            resized_ymax = lb[3] * height_ratio
            resize_width = resized_xmax - resized_xmin
            resize_height = resized_ymax - resized_ymin
            new_label.append([resized_xmin, resized_ymin, resize_width, resize_height, lb[4]])
         
        #     # Test Mark
        #     orign_image = image
        #     xmin, ymin, xmax, ymax = int(resized_xmin), int(resized_ymin), int(resized_xmax), int(resized_ymax)
        #     cls_id = int(lb[4])
        #     color = colors[cls_id]
        #     cv2.rectangle(orign_image, (xmin, ymin), (xmax, ymax), color, 2)
        #     text_size = cv2.getTextSize(classes[cls_id] , cv2.FONT_HERSHEY_PLAIN, 1, 1)[0]
        #     cv2.rectangle(orign_image, (xmin, ymin), (xmin + text_size[0] + 3, ymin + text_size[1] + 4), color, -1)
        #     cv2.putText(
        #         orign_image, classes[cls_id],
        #         (xmin, ymin + text_size[1] + 4), cv2.FONT_HERSHEY_PLAIN, 1,
        #         (255, 255, 255), 1)
        #     print("Object: {}, Bounding box: ({},{}) ({},{})".format(classes[cls_id], xmin, xmax, ymin, ymax))    

        # cv2.imshow("CoCo1", orign_image)
        # cv2.waitKey()
        return image, new_label