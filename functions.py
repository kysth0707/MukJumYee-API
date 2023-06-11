import numpy as np
import cv2 as cv
import pytesseract

from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.patches as patches
pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

if __name__ == "__main__":
    print("This is function file.")
    print("Please import this, not start this file!")

def imshow(img : np.ndarray):
	plt.imshow(cv.cvtColor(img, cv.COLOR_BGR2RGB))
	return True

def getClearImage_1(img : np.ndarray) -> np.ndarray:
	try:
		img = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
	except:
		pass
	b = cv.adaptiveThreshold(img,255,cv.ADAPTIVE_THRESH_GAUSSIAN_C,\
	cv.THRESH_BINARY,15,2)
	k = cv.getStructuringElement(cv.MORPH_RECT, (1,1))
	b = cv.dilate(b, k)
	return b


def getMaskImage_2(clearImg : np.ndarray) -> np.ndarray:
	try:
		gray = cv.cvtColor(clearImg, cv.COLOR_BGR2GRAY)
	except:
		gray = clearImg
	# (H, W) = gray.shape
	
	rectKernel = cv.getStructuringElement(cv.MORPH_RECT, (30, 20))
	# sqKernel = cv.getStructuringElement(cv.MORPH_RECT, (50, 21))
	
	gray = cv.GaussianBlur(gray, (11, 11), 0)
	blackhat = cv.morphologyEx(gray, cv.MORPH_BLACKHAT, rectKernel)
	
	grad = cv.Sobel(blackhat, ddepth=cv.CV_32F, dx=1, dy=0, ksize=-1)
	grad = np.absolute(grad)
	(minVal, maxVal) = (np.min(grad), np.max(grad))
	grad = (grad - minVal) / (maxVal - minVal)
	grad = (grad * 255).astype("uint8")
	
	grad = cv.morphologyEx(grad, cv.MORPH_CLOSE, rectKernel)
	thresh = cv.threshold(grad, 0, 255, cv.THRESH_BINARY | cv.THRESH_OTSU)[1]
	maskImage = addIt(subtractIt(thresh, (3, 3)), (10, 10))
	return maskImage

def getMaskPlusClearImage_3(clearImg : np.ndarray, maskImg : np.ndarray) -> np.ndarray:
	dst = np.full((clearImg.shape), 255, dtype=np.uint16)
	mask = maskImg
	src = clearImg
	idx = (mask!=0)
	dst[idx] = src[idx]
	return dst

def getBoundedImages_4(targetImg : np.ndarray):
	img = cv.cvtColor(targetImg.astype(np.uint8), cv.COLOR_GRAY2BGR)
	gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
	
	cutImage = []
	contours, hierarchy = cv.findContours(gray, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
	for contour in contours:
		x, y, w, h = cv.boundingRect(contour)
		cutImage.append(targetImg[y:y+h,x:x+w])
	cutImage = np.array(cutImage)
	return cutImage

def getClearedMultipleStackImage_5(targetImg : np.ndarray):
	heights = []
	for x in targetImg:
		h = x.shape[0]
		heights.append(h)
	cutHeights = heights - np.full((len(heights)), np.min(heights), dtype=np.uint16)
	realCutImage = []
	meanValue = np.mean(cutHeights)
	minValue = np.min(heights)
	for i, x in enumerate(cutHeights):
		if x > meanValue:
			print(i, x)
			imageCount = heights[i]/(meanValue + minValue)
			h = targetImg[i].shape[0]
			# print(int(h/imageCount))
			for loopNum in range(round(imageCount)):
				splitedImage = targetImg[i][loopNum*int(h/imageCount):(loopNum+1)*int(h/imageCount),:]
				splitedImage = getBoundedImages_4(splitedImage, splitedImage)[0]
				realCutImage.append(
					splitedImage
				)
		else:
			# pass
			realCutImage.append(targetImg[i])
	realCutImage = np.array(realCutImage)
	return realCutImage

def showTesseractedImage(img : np.ndarray, config:str):
	pointDatas = pytesseract.image_to_boxes(img, config=config)
	
	fig, ax = plt.subplots(figsize=(10, 7))
	ax.imshow(cv.cvtColor(img, cv.COLOR_BGR2RGB))
	h, w = img.shape[:2]
	outputText = ""
	for data in pointDatas.split(' 0\n'):
		try:
			a,b,c,d,e=data.split(' ')
		except:
			break
		txt, x1, y1, x2, y2 = data.split(' ')
		x1 = int(x1)
		y1 = h - int(y1)
		x2 = int(x2)
		y2 = h - int(y2)
		
		outputText += txt
		rect = patches.Rectangle((x1, y1), x2-x1, y2-y1, linewidth=1, edgecolor='r', facecolor='none')
		ax.add_patch(rect)
		ax.text(x1, y1, txt, {"family" : "Gulim", "color" : "red", "size":16})
	return outputText

def addIt(arr : np.ndarray, size : tuple) -> np.ndarray:
	k = cv.getStructuringElement(cv.MORPH_RECT, size)
	return cv.dilate(arr, k)

def subtractIt(arr : np.ndarray, size : tuple) -> np.ndarray:
	k = cv.getStructuringElement(cv.MORPH_RECT, size)
	return cv.erode(arr, k)