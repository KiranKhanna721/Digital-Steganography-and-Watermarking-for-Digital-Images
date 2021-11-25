import streamlit as st
import numpy as np
import cv2
import pywt
import random
import math
import cmath
from PIL import Image
def app():
    st.title("Watermarking")
    classifiers =   st.sidebar.selectbox("Select Classifier",("DCT","DWT"))
    st.write(classifiers)            
    def DWT(coverImage, watermarkImage):
        st.write('Cover Image')
        st.image(coverImage)
        coverImage = Image.open(coverImage)
        coverImage = np.asarray(coverImage)
        coverImage = cv2.cvtColor(coverImage, cv2.COLOR_BGR2GRAY)
        coverImage = cv2.resize(coverImage,(300,300))
        st.write('Watermark Image')
        st.image(watermarkImage)
        watermarkImage = Image.open(watermarkImage)
        watermarkImage = np.asarray(watermarkImage)
        watermarkImage = cv2.cvtColor(watermarkImage,cv2.COLOR_BGR2GRAY)
        watermarkImage = cv2.resize(watermarkImage,(150,150))

        #DWT on cover image
        coverImage =  np.float32(coverImage)   
        coverImage /= 255;
        coeffC = pywt.dwt2(coverImage, 'haar')
        cA, (cH, cV, cD) = coeffC
    
        watermarkImage = np.float32(watermarkImage)
        watermarkImage /= 255;

        #Embedding
        coeffW = (0.4*cA + 0.1*watermarkImage, (cH, cV, cD))
        watermarkedImage = pywt.idwt2(coeffW, 'haar')
        st.write('Watermarked Image')
        #st.image(watermarkedImage)

        #Extraction
        coeffWM = pywt.dwt2(watermarkedImage, 'haar')
        hA, (hH, hV, hD) = coeffWM

        extracted = (hA-0.4*cA)/0.1
        extracted *= 255
        extracted = np.uint8(extracted)
        st.write('Extracted Image')
        st.image(extracted)

    def DCT(coverImage, watermarkImage):
        st.write('Cover Image')
        st.image(coverImage)
        coverImage = Image.open(coverImage)
        coverImage = np.asarray(coverImage)
        coverImage = cv2.cvtColor(coverImage, cv2.COLOR_BGR2GRAY)
        coverImage = cv2.resize(coverImage,(512,512))
        st.write('Watermark Image')
        st.image(watermarkImage)
        watermarkImage = Image.open(watermarkImage)
        watermarkImage = np.asarray(watermarkImage)
        watermarkImage = cv2.cvtColor(watermarkImage,cv2.COLOR_BGR2GRAY)
        watermarkImage = cv2.resize(watermarkImage,(64,64))
        coverImage =  np.float32(coverImage)   
        watermarkImage = np.float32(watermarkImage)
        watermarkImage /= 255

        blockSize = 8
        c1 = np.size(coverImage, 0)
        c2 = np.size(coverImage, 1)
        max_message = int((c1*c2)/(blockSize*blockSize))

        w1 = np.size(watermarkImage, 0)
        w2 = np.size(watermarkImage, 1)

        watermarkImage = np.round(np.reshape(watermarkImage,(w1*w2, 1)),0)

        if w1*w2 > max_message:
            st.write('Message too large to fit')

        message_pad = np.ones((max_message,1), np.float32)
        message_pad[0:w1*w2] = watermarkImage

        watermarkedImage = np.ones((c1,c2), np.float32)

        k=50
        a=0
        b=0

        for kk in range(int(max_message)):
            dct_block = cv2.dct(coverImage[b:b+blockSize, a:a+blockSize])
            if message_pad[kk] == 0:
                if dct_block[4,1]<dct_block[3,2]:
                    temp=dct_block[3,2]
                    dct_block[3,2]=dct_block[4,1]
                    dct_block[4,1]=temp
            else:
                if dct_block[4,1]>=dct_block[3,2]:
                    temp=dct_block[3,2]
                    dct_block[3,2]=dct_block[4,1]
                    dct_block[4,1]=temp

            if dct_block[4,1]>dct_block[3,2]:
                if dct_block[4,1] - dct_block[3,2] <k:
                    dct_block[4,1] = dct_block[4,1]+k/2
                    dct_block[3,2] = dct_block[3,2]-k/2
            else:
                if dct_block[3,2] - dct_block[4,1]<k:
                    dct_block[3,2] = dct_block[3,2]+k/2
                    dct_block[4,1] = dct_block[4,1]-k/2
            
            watermarkedImage[b:b+blockSize, a:a+blockSize]=cv2.idct(dct_block)
            if a+blockSize>=c1-1:
                a=0
                b=b+blockSize
            else:
                a=a+blockSize

        watermarkedImage_8 = np.uint8(watermarkedImage)
        st.image(watermarkedImage_8)
    
    st.markdown("Upload an Cover image ")
    c_image = st.file_uploader("Choose an Cover image...", type="jpg")
    
    if c_image is not None:
        st.markdown("Upload an image ")
        s_image = st.file_uploader("Choose an image...", type="jpg")
        if s_image is not None:
            
            if classifiers =='DWT':          
                DWT(c_image, s_image)
            elif classifiers =='DCT':              
                DCT(c_image, s_image)

