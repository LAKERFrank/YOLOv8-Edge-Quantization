import cv2
import csv
import os
import sys
import shutil
import glob

try:
	videoFolder = sys.argv[1]
	outputPath = sys.argv[2]
	if (not videoFolder) or (not outputPath):
		raise ''
except:
	print('usage: python3 Frame_Generator.py <videoFolder> <outputFolder>')
	exit(1)

if outputPath[-1] != '/':
	outputPath += '/'
	
if os.path.exists(outputPath):
	shutil.rmtree(outputPath)

os.makedirs(outputPath)

video_list = glob.glob(os.path.join(videoFolder, '*'))

#Segment the video into frames
for videoName in video_list:
	print(videoName)
	frameFolder = videoName.replace("video", "frame")
	frameFolder = os.path.splitext(frameFolder)[0]
	os.makedirs(frameFolder)

	cap = cv2.VideoCapture(videoName)
	success, count = True, 0
	success, image = cap.read()
	while success:
		cv2.imwrite(f'{frameFolder}/{count}.png', image)
		count += 1
		success, image = cap.read()
