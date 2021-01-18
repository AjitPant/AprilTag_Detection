
import cv2
import numpy as np
import random
import os
from helper import add_noise, add_shadow, apply_motion_blur, add_spot_light, add_parallel_light

class backgroundOverlayer(object):
    """
    Overlay's april tag on the background image
    """
    def __init__(self, apriltag_generator , mx_tags):

        self.generator = apriltag_generator
        self.mx_tags = mx_tags

    def __call__(self, background_img):


        corners_collection = []
        bytecode_collection = []
        familycode_collection = []
        tags_to_overlay = 1
        out_response = np.zeros(background_img.shape[:2], dtype = np.uint8)
        real_out_response = np.full((background_img.shape[0],background_img.shape[1], 5),0, dtype = np.uint8)
        real_out_response[:,:,-1] = 255


        really_real_out_response = np.full((background_img.shape[0],background_img.shape[1], 5),0, dtype = np.uint8)
        really_real_out_response[:,:,-1] = 255

        id_real_out_response = np.full((background_img.shape[0],background_img.shape[1], 2),0, dtype = np.uint8)

        #It attemps to generate as many tags as possible till the upper_limit tags_to_overlay, but sometimes two might overlap it will just remove the later one

        for tag in range(tags_to_overlay):
            index = random.randrange(min(1,len(self.generator)))

            result = self.generator[index]

            response = result["response"]
            response_in_use = result["response_in_use"]
            mask     = result["mask"]
            tag_img  = result["image"]


            corners_coords = result["corners_uv"]
            bytecode = result["bytecode"]
            familycode = result["familycode"]



            # mask = np.maximum(mask, tag_img)
            _, mask = cv2.threshold(mask, 254, 255, cv2.THRESH_BINARY)
            mask_inv = cv2.bitwise_not(mask)


            width = tag_img.shape[1]
            height = tag_img.shape[0]
            if background_img.shape[1] < width or background_img.shape[0] < height:
                continue

            x_offset = random.randrange(background_img.shape[1] - width + 1)
            y_offset = random.randrange(background_img.shape[0] - height + 1)

            out_response_view      = out_response[y_offset:y_offset + height, x_offset:x_offset + width]
            real_out_response_view = real_out_response[y_offset:y_offset + height , x_offset:x_offset + width]
            really_real_out_response_view = real_out_response[y_offset:y_offset + height , x_offset:x_offset + width]

            if  cv2.bitwise_and(out_response_view, mask).any():
                continue


            #Merge with the image
            background_img_view = background_img[y_offset:y_offset + height , x_offset:x_offset + width]
            img_masked          = cv2.bitwise_and(background_img_view, background_img_view, mask=mask_inv)

            tag_img             = cv2.cvtColor(tag_img, cv2.COLOR_GRAY2BGR)
            tag_img = np.clip(tag_img, random.randint(0,10)*10, 255)
            tag_img_masked      = cv2.bitwise_and(tag_img, tag_img, mask = mask)


            #Find light
            if np.random.uniform(0, 1, 1)[0] > 0.1:

                background_img_view_lab = cv2.cvtColor(background_img_view, cv2.COLOR_BGR2LAB)
                tag_img_view_lab = cv2.cvtColor(tag_img_masked, cv2.COLOR_BGR2LAB)

                light_background = background_img_view_lab[:, :,0].mean()
                light_tag = tag_img_view_lab[:,:,0].sum()/ np.count_nonzero(mask)


                w_light =  (( light_background/(light_tag + 0.0001)))
                tag_img_view_lab[:, :, 0] = np.clip(np.multiply(tag_img_view_lab[:,:,0]  ,w_light), 0, 255)

                if np.random.uniform(0, 1, 1)[0] > 1.7:
                    tag_img_view_lab[:, :,0] = add_spot_light(tag_img_view_lab[:,:,0][..., np.newaxis])
                    tag_img_view_lab[:, :,0] = add_parallel_light(tag_img_view_lab[:,:,0][..., np.newaxis])

                tag_img_masked= cv2.cvtColor(tag_img_view_lab, cv2.COLOR_LAB2BGR)


            tag_img_masked      = cv2.bitwise_and(tag_img_masked, tag_img, mask = mask)
            background_img_view =  cv2.add(img_masked, tag_img_masked)

            #make sure no overlaps
            out_response_view      = out_response[y_offset:y_offset + height, x_offset:x_offset + width]
            real_out_response_view = real_out_response[y_offset:y_offset + height , x_offset:x_offset + width]
            id_real_out_response_view = id_real_out_response[y_offset:y_offset + height , x_offset:x_offset + width, 0]
            really_real_out_response_view = really_real_out_response[y_offset:y_offset + height , x_offset:x_offset + width]

            if  not cv2.bitwise_and(out_response_view, mask).any():


                if np.random.uniform(0, 1, 1)[0] > 1.8:
                    blurred_background_img_view = cv2.GaussianBlur(background_img_view, (5, 5), 0)
                    contours, hierarchy = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                    tmp_mask = np.zeros(background_img_view.shape, dtype = np.uint8)
                    cv2.drawContours(tmp_mask, contours, -1, (255,255, 255),5)
                    background_img_view = np.where(tmp_mask==np.array([255, 255, 255]), blurred_background_img_view, background_img_view)

                background_img[y_offset:y_offset + height , x_offset:x_offset + width]     = background_img_view
                out_response[y_offset:y_offset + height , x_offset:x_offset + width]       = cv2.bitwise_or(out_response_view, mask)
                real_out_response[y_offset:y_offset + height , x_offset:x_offset + width, :-1]  = np.maximum(response[:,:,:-1], real_out_response_view[:,:,:-1])
                real_out_response[y_offset:y_offset + height , x_offset:x_offset + width, -1]  = np.minimum(response[:,:,-1], real_out_response_view[:,:,-1])

                id_real_out_response[y_offset:y_offset + height , x_offset:x_offset + width, 0]  = np.maximum(id_real_out_response_view, mask/255*index)

                really_real_out_response[y_offset:y_offset + height , x_offset:x_offset + width, :-1]  = np.maximum(response_in_use[:,:,:-1], really_real_out_response_view[:,:,:-1])
                really_real_out_response[y_offset:y_offset + height , x_offset:x_offset + width, -1]  = np.minimum(response_in_use[:,:,-1], really_real_out_response_view[:,:,-1])

                corners_collection.append([np.array([[x_offset, y_offset]])+corners_coords ])
                bytecode_collection.append(bytecode)
                familycode_collection.append(familycode)


        if np.random.uniform(0, 1, 1)[0] > 1.5:
            background_img = add_noise(background_img, "gauss")

        if np.random.uniform(0, 1, 1)[0] > 1.8:
            background_img = add_noise(background_img, "s&p")

        if np.random.uniform(0, 1, 1)[0] > 1.8:
            background_img = add_noise(background_img, "speckle")

        # Motion blur
        if np.random.uniform(0, 1, 1)[0] > 1.8 :
            size = np.random.randint(3, 7)
            deg = np.random.randint(-180, 180)
            background_img = apply_motion_blur(background_img, size, deg)




        return background_img, out_response, np.clip(real_out_response,0,255),np.clip(really_real_out_response,0,255), id_real_out_response, corners_collection, bytecode_collection, familycode_collection
