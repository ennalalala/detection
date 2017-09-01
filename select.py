import os
from os import listdir
from os.path import isfile, join
import xml.dom.minidom
def getText(node):
	return node.firstChild.nodeValue
path = '/workspace/py-faster-rcnn/data/imagenet/'
filenames = listdir(path+'Annotations/n03126707')
count =0
for name in filenames:
	dom = xml.dom.minidom.parse(path+'Annotations/n03126707/'+name)
	root = dom.documentElement
	size = root.getElementsByTagName("size")[0]
	width = float(getText(size.getElementsByTagName("width")[0]))
	height = float(getText(size.getElementsByTagName("height")[0]))
	ratio = width/height
	if(ratio<0.462 or ratio>6.828):
		print(path+'Annotations/n03126707/'+name)
		print(ratio)
		count = count + 1	
print(count)

