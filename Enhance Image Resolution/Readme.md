# TO DO
To make the model able to resolute any resolution level, by first identifying some kind of "Resolution index" and then passing to model accordingly for 5-10 different resolution levels

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
i) 
