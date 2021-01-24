import os
import cv2
os.environ['CUDA_LAUNCH_BLOCKING'] = "1" 
import os
import pickle


colors = pickle.load(open("dataset//pallete", "rb")) 

def mark_target(img, targets, dataset, index):
    # img = np.array(img.permute(1, 2, 0).cpu()*255, dtype=np.uint8) # Re multiply
    # pred_img = copy.deepcopy(img)
    # mark target
    for target in targets:
        target = target.numpy()
        if target[0] == index:
            box = target[2:]
            cls_id = int(target[1])
            color = colors[cls_id]
            xmin, ymin, xmax, ymax = int(box[0]), int(box[1]), int(box[2]), int(box[3])
            xmax += xmin
            ymax += ymin
            cv2.rectangle(img, (xmin, ymin), (xmax, ymax), color, 2)
            text_size = cv2.getTextSize(dataset.classes[cls_id] , cv2.FONT_HERSHEY_PLAIN, 1, 1)[0]
            cv2.rectangle(img, (xmin, ymin), (xmin + text_size[0] + 3, ymin + text_size[1] + 4), color, -1)
            cv2.putText(
                img, dataset.classes[cls_id],
                (xmin, ymin + text_size[1] + 4), cv2.FONT_HERSHEY_PLAIN, 1,
                (255, 255, 255), 1)
            # print("Object: {}, Bounding box: ({},{}) ({},{})".format(classes[cls_id], xmin, xmax, ymin, ymax))  
    
    # cv2.imshow('win', img)
    # cv2.waitKey()
    return img

def mark_pred(pred_img, pred_boxes, dataset):
    # pred_img = np.array(pred_img.permute(1, 2, 0).cpu()*255, dtype=np.uint8) # Re multiply
    # for pred_boxes in suppress_output:
    if type(None) == type(pred_boxes): return pred_img 
    for target in pred_boxes:
        target = target.cpu().numpy() # (x1, y1, x2, y2, object_conf, class_score, class_pred)
        box = target[:4]
        cls_id = int(target[6])
        color = colors[cls_id]
        xmin, ymin, xmax, ymax = int(box[0]), int(box[1]), int(box[2]), int(box[3])
        xmax += xmin
        ymax += ymin
        cv2.rectangle(pred_img, (xmin, ymin), (xmax, ymax), color, 2)
        text_size = cv2.getTextSize(dataset.classes[cls_id] , cv2.FONT_HERSHEY_PLAIN, 1, 1)[0]
        cv2.rectangle(pred_img, (xmin, ymin), (xmin + text_size[0] + 3, ymin + text_size[1] + 4), color, -1)
        cv2.putText(
            pred_img, dataset.classes[cls_id],
            (xmin, ymin + text_size[1] + 4), cv2.FONT_HERSHEY_PLAIN, 1,
            (255, 255, 255), 1)
        # print("Object: {}, Bounding box: ({},{}) ({},{})".format(classes[cls_id], xmin, xmax, ymin, ymax)) 

    # cv2.imshow('win', pred_img)
    # cv2.waitKey()
    return pred_img
