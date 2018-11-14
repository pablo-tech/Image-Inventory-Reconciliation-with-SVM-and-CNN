## Project Title: Amazon Inventory Reconciliation using AI
## Project Category: Computer Vision
## Team Members: Sravan Sripada 06318830, Nutchapol 005976992 , Pablo Bertorello 04530473

Amazon Fulfillment Centers are bustling hubs of innovation that allow Amazon to deliver millions of products to over 100 countries worldwide with the help of robotic and computer vision technologies. Occasionally, items are misplaced while being handled, resulting in a mismatch between recorded bin inventory and contents of some bin images. The project consists of using a bin image dataset to count the number of items in each bin, to detect variance from recorded inventory. 

Currently, Amazon uses a random storage scheme where items are placed into accessible bins with available space, so the contents of each bin are random--rather than organized by specific product types. To find the best solution to the bin inventory mismatch, Amazon has made public the Bin Image Dataset.  It contains images and metadata from bins of a pod in an operating Amazon Fulfillment Center. The bin images in this dataset are captured as robot units carry pods as part of normal operations. Bin Image dataset provides the metadata for each image from where number of items in bin can be derived.  More details at:
https://docs.opendata.aws/aft-vbi-pds/readme.html

Our hypothesis is that the problem can be treated as a multiclass classification problem: predicting if each bin consists of 1, 2 , 3, 4, >4 items.  We anticipate using various techniques to extract features: Fourier transforms, Histogram of gradients SIFT, SURF, BRIEF on images.  These features may then be use to train a support vector machines and multi-class logistic regression algorithms. Various versions Bag of visual words technique may be helpful, adjusting for parameters in K means clustering, to extract histogram features which can be used for training.  Lastly, convolutional Neural networks may be trained on the images to predict the number of objects. We plan to explore the possibility of using Mask RCNN algorithm to detect objects in photos. This model generates bounding boxes and segmentation masks for each instance of an object in the image (it is based on Feature Pyramid Network, FPN, and a ResNet101 backbone).  More at: https://github.com/matterport/Mask_RCNN

To pick the best method, the classification error for each class may be used as a key metric

Time permitting, we will work on the following additional interesting problems:
Whether a particular item is present in the bin. Given a question of whether an item is present in the bin, the output should be a yes or no
How many units of a given item are present in the bin

