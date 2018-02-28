from os import listdir,rename
from cv2 import *
from os.path import isfile, join
path="D:/ML Papers HRS/LowtoHIGH/LOW_105x70"
paths="D:/ML Papers HRS/LowtoHIGH/LOW_525x350"
pathb="D:/ML Papers HRS/LowtoHIGH/BICUBIC"

onlyfiles = [f for f in listdir(path) if isfile(join(path, f))]
i=1;
pct=0.2
new_size=(int(525*pct),int(350*pct))
for f in onlyfiles:
    #rename(path+"/"+f, path+"/"+str(i)+".jpg")
    #imwrite(f,resize(imread(path+"/"+f),(525,350),interpolation=INTER_AREA))
	imwrite(paths+"/"+f,resize(imread(path+"/"+f),(525,350)))
	#resize(imread(path+"/"+f),(pathb+"/"+str(i)+".jpg"),[525,350],0,0,INTER_CUBIC)
	i=i+1
	
    