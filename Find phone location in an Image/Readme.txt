To train:
python train_phone_finder.py ~/find_phone

To test:
python phone_predictor.py ~/test.jpg


Things Tried to increase the accuracy:
1) Use pretrained Mobilenet to extract image features
2) Use Euclidian distance as loss and/or metric parameter
3) Use R_squared as metric 
4) Compared linear vs sigmoid activation functions for the last layer
5) Tried various DNN node configurations



Things that may increase accuracy:
1) Use Image augmentation to get more data
