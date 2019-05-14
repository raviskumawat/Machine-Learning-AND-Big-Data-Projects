# TO DO
To make the model able to resolute any resolution level, by first identifying some kind of "Resolution index" and then passing to model accordingly for 5-10 different resolution levels

Keep feeding the resoluted image back-to-back to make all images upto a final best resolution


## Use GAN's (Recovers finer texture)? and Sub-pixel Convolution(Learns upscaling features)??

Try:
i) Sigmoid+log as GAN loss function
ii) Perceptual loss/Adversarial loss/content loss
iii) total variation loss/discriminator loss
iv) subpixel reshuffle layer
v) Replacing per-pixel loss by perceptual loss gives visually pleasing results
vi) use Y channel from YCrCb format
vii) normalize image 0-255 to 0-1


## what papers say??
i) 
