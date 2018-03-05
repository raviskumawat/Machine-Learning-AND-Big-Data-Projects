import math
p=[15  ,12,  8 ,  8  , 7,   7  , 7  , 6 ,  5   ,3]
h=[10  ,25 , 17 , 11  ,13  ,17 , 20 , 13 , 9 ,  15]
#p = [int(x) for x in input().split()]
#h = [int(x) for x in input().split()]
#print(p)
s_x2=sum([x**2 for x in p])
s_y2=sum([x**2 for x in h])
s_x=sum(p)
s_y=sum(h)
s_xy=sum([x*y for x,y in zip(p,h)])
r=(10*(s_xy)-s_x*s_y)/math.sqrt((10*s_x2-s_x**2)*(10*s_y2-s_y**2))
print(format(r,'.3f'))