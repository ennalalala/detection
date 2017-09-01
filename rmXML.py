import os
from os import listdir
from os.path import isfile, join
path = '/workspace/py-faster-rcnn/data/imagenet/'
filenames = listdir(path+'Annotations/n03126707')
count =0
for name in filenames:
	imagename = name.split('.')[0]+'.JPEG'
	if not os.path.exists(path+'Images/'+imagename):
		print(path+'Images'+imagename)
		count = count + 1
		os.remove(path+'Annotations/n03126707/'+name)
print(count)
