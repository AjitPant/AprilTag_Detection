
import cv2
import numpy as np
import random
import os
from helper import add_noise, add_shadow

class backgroundOverlayer(object):
    """
    Overlay's april tag on the background image
    """
    def __init__(self, apriltag_generator , mx_tags):

        self.generator = apriltag_generator
        self.mx_tags = mx_tags

    def __call__(self, background_img):


        tags_to_overlay = random.randint(0, self.mx_tags)
        out_response = np.zeros(background_img.shape[:2], dtype = np.uint8)
        real_out_response = np.full(background_img.shape[:2],0, dtype = np.uint8)

        #It attemps to generate as many tags as possible till the upper_limit tags_to_overlay, but sometimes two might overlap it will just remove the later one

        for tag in range(tags_to_overlay):
            index = random.randrange(len(self.generator))
            result = self.generator[index]

            response = result["response"]
            mask     = result["mask"]
            tag_img  = result["image"]

            contours, hierarchy = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

            mask = cv2.drawContours(mask , contours, contourIdx = -1, color=255, thickness = 5)
            mask = cv2.bitwise_or(mask, tag_img)
            mask_inv = cv2.bitwise_not(mask)

            width = tag_img.shape[1]
            height = tag_img.shape[0]

            x_offset = random.randrange(background_img.shape[1] - width + 1)
            y_offset = random.randrange(background_img.shape[0] - height + 1)

            #Merge with the image
            background_img_view = background_img[y_offset:y_offset + height , x_offset:x_offset + width]
            img_masked          = cv2.bitwise_and(background_img_view, background_img_view, mask=mask_inv)

            tag_img             = cv2.cvtColor(tag_img, cv2.COLOR_GRAY2BGR)
            tag_img_masked      = cv2.bitwise_and(tag_img, tag_img, mask = mask)


            #Find light
            background_img_view_lab = cv2.cvtColor(background_img_view, cv2.COLOR_BGR2LAB)
            tag_img_view_lab = cv2.cvtColor(tag_img_masked, cv2.COLOR_BGR2LAB)

            light_background = background_img_view_lab[:,:,0].mean()
            light_tag = tag_img_view_lab[:,:,0].sum() / np.count_nonzero(mask)

            w_light =  (( light_background/light_tag));
            tag_img_view_lab[:, :, 0] = np.clip((tag_img_view_lab[:, :,0] * w_light), 0, 255);

            tag_img_masked= cv2.cvtColor(tag_img_view_lab, cv2.COLOR_LAB2BGR)
            background_img_view =  cv2.add(img_masked, tag_img_masked)

            #make sure no overlaps
            out_response_view      = out_response[y_offset:y_offset + height, x_offset:x_offset + width]
            real_out_response_view = real_out_response[y_offset:y_offset + height , x_offset:x_offset + width]

            if not cv2.bitwise_and(out_response_view, mask).any():


                blurred_background_img_view = cv2.GaussianBlur(background_img_view, (5, 5), 0)

                contours, hierarchy = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

                tmp_mask = np.zeros(background_img_view.shape, dtype = np.uint8)
                cv2.drawContours(tmp_mask, contours, -1, (255,255, 255),5)
                background_img_view = np.where(tmp_mask==np.array([255, 255, 255]), blurred_background_img_view, background_img_view)

                background_img[y_offset:y_offset + height , x_offset:x_offset + width]     = background_img_view
                out_response[y_offset:y_offset + height , x_offset:x_offset + width]       = cv2.bitwise_or(out_response_view, mask)
                real_out_response[y_offset:y_offset + height , x_offset:x_offset + width] += 4- np.array(response.argmax(axis = 2), dtype = np.uint8)


        # background_img[:,:,0] = cv2.equalizeHist(background_img[:,:,0]);
        # background_img[:,:,1] = cv2.equalizeHist(background_img[:,:,1]);
        # background_img[:,:,2] = cv2.equalizeHist(background_img[:,:,2]);
        background_img = add_shadow(background_img, random.randrange(5))
        background_img = add_noise(background_img, "gauss")
        return background_img, out_response, real_out_response
