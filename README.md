# Image Captioning using Neural Networks and Visual Attention

This project aims at generating novel, natural language captions of input images, using a Seq2Seq architecture that uses a CNN as image encoder and an LSTM RNN as a feature decoder. The project is implemented using Tensorflow, and is roughly based on [Show, Attend and Tell: Neural Image Caption Generation with Visual Attention](https://arxiv.org/abs/1502.03044) paper by Kelvin Xu et al. All the code has been thoroughly commented on for ease of understanding. Schematic below roughly shows the structure of the training graph ('Z' denotes the 'Context Vector'):

![picture alt](https://github.com/aayushARM/image-captioning/blob/master/assets/training_graph.jpg)


## Overview: ##

* Uses a multi-threaded input pipeline implemented using Tensorflow Queues, producing a fast, steady stream of inputs from multiple TFRecord files(shards), with rigorous shuffling of data for data augmentation.   
* Uses Google AutoML Project's [NASNet](https://github.com/tensorflow/models/tree/master/research/slim/nets/nasnet) architecture to extract visual features from images. NASNet currently has highest recorded accuracy on ImageNetLSVRC2012 data set.
* Trained on MS COCO 2017 Training data set. The final pre-processed data consists of 5,87,605 image-caption pairs. The vocabulary used consists of 10,204 words(words occuring >=5 times in MSCOCO captions data).
* The LSTM RNN is combined with Soft Attention Mechanism, which computes attention weights, applies them to image features, and produces a "context vector", which is fed into LSTM as an additional input along with hidden states. This gives our LSTM more contextual information at every time-step for producing higher quality captions while decoding.
* Added L2 regularization to all fully-connected layers and applied drop-out ratio of 0.5 on LSTM and FC Layers to prevent over-fitting.
* The image-caption pairs from data set were added to SequenceExample protocol buffers and then written to files with TFRecord file format. This helps in faster, asynchronous reading of data by input pipeline.

To Do: Use Beam Search decoder with LSTM instead of Greedy decoder for greater quality captions.

## Results: ##

The model achieves following scores on COCO 2014 Validation and Test data:
* BLUE1: __70.07__
* BLUE2: __52.88__
* BLUE3: __37.95__
* BLUE4: __26.68__
* Meteor: __23.61__
* Rogue-L: __51.29__
* CIDEr: __84.81__

The captions generated for the COCO test and validation images are usually relevant to the images. An Android application was also developed that sends the images to the online Flask server, which runs the frozen inference graph and returns the captions to the client. Below are the examples of captions generated using the Android application on some random images that weren't a part of the MS COCO data set:

![picture alt](https://github.com/aayushARM/image-captioning/blob/master/assets/1.jpeg)
![picture alt](https://github.com/aayushARM/image-captioning/blob/master/assets/3.jpeg)
![picture alt](https://github.com/aayushARM/image-captioning/blob/master/assets/4.jpeg)


![picture alt](https://github.com/aayushARM/image-captioning/blob/master/assets/5.jpeg)
![picture alt](https://github.com/aayushARM/image-captioning/blob/master/assets/6.jpeg)
![picture alt](https://github.com/aayushARM/image-captioning/blob/master/assets/7.jpeg)

<p align="center">
<img src="https://github.com/aayushARM/image-captioning/blob/master/assets/8.jpeg">
</p>

### References: ###

* [Show, Attend and Tell: Neural Image Caption Generation with Visual Attention](https://arxiv.org/abs/1502.03044). Kelvin Xu, Jimmy Ba, Ryan Kiros, Kyunghyun Cho, Aaron Courville, Ruslan Salakhutdinov, Richard Zemel, Yoshua Bengio.
* [Show, Attend and Tell slides](http://www.slideshare.net/eunjileee/show-attend-and-tell-neural-image-caption-generation-with-visual-attention)
* [Attention Mechanism Blog Post](https://blog.heuritech.com/2016/01/20/attention-mechanism/)
* [Interpreting, Training, and Distilling Seq2Seq Models, Alexander Rush (@harvardnlp)](http://www.github.com/harvardnlp/seq2seq-talk/)
* [The original paper implementation in Theano](https://github.com/kelvinxu/arctic-captions)
* [Another Tensorflow implementation](https://github.com/DeepRNN/image_captioning)
* [NASNet Architecture and pre-trained checkpoints](https://github.com/tensorflow/models/tree/master/research/slim/nets/nasnet)
* [Google's implementation of "Show and Tell: A Neural Image Caption Generator"](https://github.com/tensorflow/models/tree/master/research/im2txt)
