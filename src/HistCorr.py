import numpy as np
import cv2
import matplotlib.pyplot as plt
import pandas as pd


class HistCorr:
    
    def __init__(self, img, img_list, present = False):
        
        self.img = img
        self.img_list = img_list
        
        corr_list = []
        
        gray_sample = cv2.cvtColor(self.img, cv2.COLOR_BGR2GRAY)
        hist_sample = cv2.calcHist([gray_sample], [0], None, [256], [0, 256])
        hist_sample = cv2.normalize(hist_sample, hist_sample, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX)
        
        for img in img_list:
            
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            hist = cv2.calcHist([gray], [0], None, [256], [0, 256])
            hist = cv2.normalize(hist, hist, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX)
            
            corr = cv2.compareHist(hist_sample, hist, cv2.HISTCMP_CORREL)
            
            corr_list.append(corr) 
        
        self.corr_list = corr_list
        
        if present == True:

            ind = np.argmax(corr_list)

            fig, axs = plt.subplots(1, 2, figsize = (10,10))

            fig.tight_layout()

            axs[0].margins(0.02)
            axs[0].imshow(self.img)
            axs[0].axis('off')

            axs[1].margins(0.02)
            axs[1].imshow(img_list[ind])
            axs[1].axis('off')

            plt.show()
            
            print('Histogram correlation:', corr_list[ind])
    
    def get_corr(self):

        return max(self.corr_list)

    