from os import listdir,rename
from cv2 import *
from os.path import isfile, join
path="D:/ML Papers HRS/LowtoHIGH/LOW_102x102"
paths="D:/ML Papers HRS/LowtoHIGH/HIGH_525x350"
onlyfiles = [f for f in listdir(paths) if isfile(join(paths, f))]
i=1;
pct=0.2
new_size=(int(525*pct),int(350*pct))
for f in onlyfiles:
    #rename(path+"/"+f, path+"/"+str(i)+".jpg")
    #i=i+1;
    #imwrite(f,resize(imread(path+"/"+f),(525,350),interpolation=INTER_AREA))
	imwrite(path+"/"+f,resize(imread(paths+"/"+f),new_size))
	
    