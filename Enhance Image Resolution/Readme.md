# TO DO
To make the model able to resolute any resolution level, by first identifying some kind of "Resolution index" and then passing to model accordingly for 5-10 different resolution levels

## Enhance Image resolution
Input vs output vs original image
![](https://github.com/raviskumawat/Machine-Learning-AND-Big-Data-Projects/blob/master/Enhance%20Image%20Resolution/ResoluteIT.png)


Keep feeding the resoluted image back-to-back to make all images upto a final best resolution


## Use GAN's (Recovers finer texture)? and Sub-pixel Convolution(Learns upscaling features)??
<br/>
Try:<br/>
i) Sigmoid+log as GAN loss function<br/>
ii) Perceptual loss/Adversarial loss/content loss<br/>
iii) total variation loss/discriminator loss<br/>
iv) subpixel reshuffle layer<br/>
v) Replacing per-pixel loss by perceptual loss gives visually pleasing results<br/>
vi) use Y channel from YCrCb format<br/>
vii) normalize image 0-255 to 0-1<br/>
<br/>
<br/>

## what papers say??
i) Non-linear dimentionality reduction: Isometric feature mapping(Isomap) , locally linear embedding(LLE), laplacian eigen map<br/>


