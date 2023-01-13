#-----------------------------------------------------------------------#
#   predict.py將單張圖片預測、攝像頭檢測、FPS測試和目錄遍歷檢測等功能
#   整合到了一個py文件中，通過指定mode進行模式的修改。
#-----------------------------------------------------------------------#
import time
import os
import cv2
import numpy as np
from PIL import Image
import importlib
import argparse

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Attribute Learner')
    parser.add_argument('--config', type=str, default="configs.yolox_base" 
                        ,help = 'Path to config .opt file. Leave blank if loading from opts.py')
    parser.add_argument("--mode", type=str, default="video" , help="predict or video")  
    parser.add_argument("--video_fps", type=float, default=25.0, help="video_fps")  
    parser.add_argument("--test_interval", type=int, default=100, help="test_interval") 

    parser.add_argument("--video_path", type=str, 
                                        default="/home/leyan/DataSet/LANEdevkit/Drive-View-Noon-Driving-Taipei-Taiwan.mp4", 
                                        )  
    parser.add_argument("--video_save_path", type=str, 
                                        default="pred_out/coco.mp4", 
                                        ) 
    parser.add_argument("--dir_origin_path", type=str, 
                                        default="img/", 
                                        )  
    parser.add_argument("--dir_save_path", type=str, 
                                        default="img_out/", 
                                        )  

    conf = parser.parse_args() 
    opt = importlib.import_module(conf.config).get_opts(Train=False)
    for key, value in vars(conf).items():     
        setattr(opt, key, value)
    
    d=vars(opt)

    model = opt.Model_Pred(classes_path=opt.classes_path)   
    mode = opt.mode

    #----------------------------------------------------------------------------------------------------------#
    video_path      = opt.video_path
    video_save_path = opt.video_save_path
    video_fps       = opt.video_fps
    test_interval = opt.test_interval
    #----------------------------------------------------------------------------------------------------------#
    dir_origin_path = opt.dir_origin_path
    dir_save_path   = opt.dir_save_path
    fps_image_path  = "img/fps.jpg"
    #-------------------------------------------------------------------------#

    if mode == "predict":
        '''
        1、如果想要進行檢測完的圖片的保存，利用r_image.save("img.jpg")即可保存，直接在predict.py里進行修改即可。 
        2、如果想要獲得預測框的坐標，可以進入yolo.detect_image函數，在繪圖部分讀取top，left，bottom，right這四個值。
        3、如果想要利用預測框截取下目標，可以進入yolo.detect_image函數，在繪圖部分利用獲取到的top，left，bottom，right這四個值
        在原圖上利用矩陣的方式進行截取。
        4、如果想要在預測圖上寫額外的字，比如檢測到的特定目標的數量，可以進入yolo.detect_image函數，在繪圖部分對predicted_class進行判斷，
        比如判斷if predicted_class == 'car': 即可判斷當前目標是否為車，然後記錄數量即可。利用draw.text即可寫字。
        '''
        while True:
            img = dir_origin_path + input('Input image filename:')
            try:
                image = Image.open(img)
            except:
                print('Open Error! Try again!')
                continue
            else:
                r_image = model.detect_image(image)
                r_image.show()

    elif mode == "video":
        capture = cv2.VideoCapture(video_path)
        if video_save_path!="":
            fourcc  = cv2.VideoWriter_fourcc(*'XVID')
            size    = (int(capture.get(cv2.CAP_PROP_FRAME_WIDTH)), int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT)))
            out     = cv2.VideoWriter(video_save_path, fourcc, video_fps, size)

        ref, frame = capture.read()
        # if not ref:
        #     raise ValueError("未能正確讀取攝像頭（視頻），請注意是否正確安裝攝像頭（是否正確填寫視頻路徑）。")

        fps = 0.0
        drawline = False
        while(True):
            t1 = time.time()
            # 讀取某一幀
            ref, frame = capture.read()
            if not ref:
                break
            # 格式轉變，BGRtoRGB
            frame = cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)
            # 轉變成Image
            frame = Image.fromarray(np.uint8(frame))
            # 進行檢測            
            if drawline:
                frame = np.array(model.detect_image_custom_center(frame))
            else:
                frame = np.array(model.detect_image(frame))
            # RGBtoBGR滿足opencv顯示格式
            frame = cv2.cvtColor(frame,cv2.COLOR_RGB2BGR)
            
            fps  = ( fps + (1./(time.time()-t1)) ) / 2
            print("fps= %.2f"%(fps))
            frame = cv2.putText(frame, "fps= %.2f"%(fps), (0, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

            if drawline:
                # Center coordinates 
                center_coordinates = int(1280/2), int(720/3*2)
                
                # Radius of circle 
                radius = 2
                
                # Blue color in BGR 
                color = (0, 0, 255) 
                
                # Line thickness of 2 px 
                thickness = 2
                
                # Draw a circle with blue line borders of thickness of 2 px 
                frame = cv2.circle(frame, center_coordinates, radius, color, thickness) 
                frame = cv2.line(frame, (int(1280/5*2),0), (int(1280/5*2),720), color, thickness)
                frame = cv2.line(frame, (int(1280/6*4),0), (int(1280/6*4),720), color, thickness)
            
            cv2.imshow("video",frame)
            c= cv2.waitKey(1) & 0xff 
            if video_save_path!="":
                out.write(frame)

            if c==27:
                capture.release()
                break

        print("Video Detection Done!")
        capture.release()
        if video_save_path!="":
            print("Save processed video to the path :" + video_save_path)
            out.release()
        cv2.destroyAllWindows()

    elif mode == "fps":
        img = Image.open('img/street.jpg')
        tact_time = model.get_FPS(img, test_interval)
        print(str(tact_time) + ' seconds, ' + str(1/tact_time) + 'FPS, @batch_size 1')

    elif mode == "dir_predict":
        import os
        from tqdm import tqdm

        img_names = os.listdir(dir_origin_path)
        for img_name in tqdm(img_names):
            if img_name.lower().endswith(('.bmp', '.dib', '.png', '.jpg', '.jpeg', '.pbm', '.pgm', '.ppm', '.tif', '.tiff')):
                image_path  = os.path.join(dir_origin_path, img_name)
                image       = Image.open(image_path)
                r_image     = model.detect_image(image)
                if not os.path.exists(dir_save_path):
                    os.makedirs(dir_save_path)
                r_image.save(os.path.join(dir_save_path, img_name))
                
    else:
        raise AssertionError("Please specify the correct mode: 'predict', 'video', 'fps' or 'dir_predict'.")
