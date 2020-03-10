# YOLO-Augmentator

YOLO Augmentator is the project that augmentation methods for object detection dataset are implemented.


In this repositoty, I implemented 6 augmentation methods.

You can easily check and understand the results of each method through the visualization.

### Horizontal Flip

[Example 움짤 첨부]()

### Translation

[Example 움짤 첨부]()

### Scale

[Example 움짤 첨부]()

### Color and Brightness Distortion

[Example 움짤 첨부]()

### Shearing

[Example 움짤 첨부]()

### Rotation

[Example 움짤 첨부]()

### Augmentation Policy

Sometimes the objects in image are severely cropped or occluded by augmentation. In that case, the augmented data may harm the stability of training and performance of the network.

[Example 움짤 첨부]()

To overcome this problem, the program check the IoU between non-augmented object and augmented object after augmentation, and then if there are objects that hardly loss their information, the program ignore that augmentation.

[Example 움짤 첨부]()

[Code Line 첨부]
