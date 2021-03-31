from train_srgan import SRGAN
from PIL import Image
import numpy as np
import cv2
import os


srgan = SRGAN()
model = srgan.build_generator()
model.load_weights(r"weights/DIV2K_train_LR_bicubic_X2/gen_epoch4950.h5")
imgList = os.listdir("datasets/DIV2K_train_LR_bicubic_X2/test/")
avg_pnsr = 0
for file in imgList:
    # 原图
    src_image = Image.open(r"datasets/DIV2K_train_LR_bicubic_X2/test/"+file)
    # 低分率图
    lr_image = src_image.resize((128,128))
    input_image = np.array(lr_image)/127.5 - 1
    input_image = np.expand_dims(input_image,axis=0)
    # 生成高分率图像
    fake = (model.predict(input_image)*0.5 + 0.5)*255
    # 原高分率图像
    hr_image = np.array(src_image.resize((512,512)))
    # 低分辨率通过双线性插值转换为高分率图像
    lr_image_b = np.array(lr_image.resize((512,512)))
    # 计算PSNR
    diff = hr_image - lr_image_b
    mse = np.mean(np.square(diff))
    psnr = 10 * np.log10(255 * 255 / mse)
    avg_pnsr += psnr/len(imgList)
    # 显示三者图像
    image_show = np.hstack((hr_image,lr_image_b,np.uint8(fake[0])))
    cv2.imshow("1",cv2.cvtColor(image_show,cv2.COLOR_RGB2BGR))
    cv2.waitKey(0)
print(avg_pnsr)

