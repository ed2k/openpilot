import numpy as np

import cv2
from tools.sim.camera import transform_img
from common.transformations.camera import eon_intrinsics
from common.transformations.model import medmodel_intrinsics
from tensorflow.keras.models import load_model
from tools.sim.parser import parser

md_prev_frame = np.zeros((6,128,256))
md_desire = np.zeros((1,8))
md_traffic_convention = np.zeros((1,2))
md_state = np.zeros((1,512))
md_supercombo = load_model('../../models/supercombo.keras')
#md_supercombo = load_model('supercombo.keras')
#print(md_supercombo.summary())
#from keras.utils.vis_utils import plot_model
#plot_model(md_supercombo, to_file='supercombo.png', show_shapes=True, show_layer_names=True)

def frames_to_tensor(frames):
  H = (frames.shape[1]*2)//3                                                                                                
  W = frames.shape[2]                                                                                                       
  in_img1 = np.zeros((frames.shape[0], 6, H//2, W//2), dtype=np.uint8)                                                      
                                                                                                                            
  in_img1[:, 0] = frames[:, 0:H:2, 0::2]                                                                                    
  in_img1[:, 1] = frames[:, 1:H:2, 0::2]                                                                                    
  in_img1[:, 2] = frames[:, 0:H:2, 1::2]                                                                                    
  in_img1[:, 3] = frames[:, 1:H:2, 1::2]                                                                                    
  in_img1[:, 4] = frames[:, H:H+H//4].reshape((-1, H//2,W//2))                                                              
  in_img1[:, 5] = frames[:, H+H//4:H+H//2].reshape((-1, H//2,W//2))
  return in_img1

def show_parsed(img, data, color=(0,255,0), delta = None, alpha = 0.5):
    img2 = img.copy()
    #print(delta.shape)
    for i in range(192):
        y = 873 - 3*i
        x = int(1163/2 - data[i]*10)
        d = 0
        if delta is not None:
            d = int(delta[i])
        cv2.rectangle(img2, (x-d, y-1), (x+d,y), color, -1)

    img = cv2.addWeighted(img2, alpha, img, 1-alpha, 0)
    return img

# output:12, 4, input: (1, 12, 128, 256)
def modeld(frame):
    #cv2.imwrite('test.png', frame)
    global md_prev_frame, md_desire, md_state
    img_yuv = cv2.cvtColor(frame, cv2.COLOR_BGR2YUV_I420)
    img = img_yuv.reshape((874*3//2, 1164))
    img_med_model = transform_img(img, from_intr=eon_intrinsics, to_intr=medmodel_intrinsics, yuv=True,
                                    output_size=(512,256))
    frame_tensor = frames_to_tensor(np.array([img_med_model])).astype(np.float32)/128.0 - 1.0
    frame_tensors = [md_prev_frame, frame_tensor[0]]
    i = np.vstack(frame_tensors)[None]
    print(i.shape)
    inputs = [i, md_desire, md_traffic_convention, md_state]
    #inputs = [np.vstack(frame_tensors)[None], md_desire, md_state]
    outs = md_supercombo.predict_on_batch(inputs)
    parsed = parser(outs)

    #print(parsed)
    # Important to refeed the state
    md_state = outs[-1]
    pose = outs[-2]
    md_prev_frame = frame_tensor[0]
    return parsed

import sys, os, random
def main():
    img = cv2.imread('test.png')
    if len(sys.argv) > 1:
        img = cv2.imread(sys.argv[1])
    parsed = modeld(img[:,:,::-1])
    print(parsed['path_stds'])
    #return

    clean_img = img.copy()
    while True:
        img = clean_img.copy()
        parsed = modeld(img[:,:,::-1])
        # cv2 uses bgr
        img = show_parsed(img, parsed['lll'][0], color=(0,0,255), delta=parsed['lll_stds'][0], alpha=parsed['lll_prob'][0])
        img = show_parsed(img, parsed['rll'][0], color=(255,0,0), delta=parsed['rll_stds'][0], alpha=parsed['rll_prob'][0])
        img = show_parsed(img, parsed['path'][0], delta=parsed['path_stds'][0])
        cv2.imshow("modeld", img)
        key = cv2.waitKey(10)
        if key & 0xFF == ord('q'):
            break
        elif key & 0xFF == ord('n'):
            base_path = '../../../comma10k/imgs/'
            fname = random.choice(os.listdir(base_path))
            print(fname)
            img = cv2.imread(base_path+fname)
            clean_img = img.copy()

main()