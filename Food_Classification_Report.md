## <center>Food Classification using TensorFlow (CNN Transfer learning)</center>
<center><img src= "https://timesofindia.indiatimes.com/thumb/msid-77563051,width-1200,height-900,resizemode-4/.jpg" alt ="Titanic" style='width:500px;'></center><br>

- <h4> Image classification is the task of identifying images and categorizing them in one of several predefined distinct classes.</h4><br>
- <h4> Being one of the computer vision (CV) tasks, Image classification serves as the foundation for solving different CV problems such as object detection which further gets divided into semantic and instance segmentation.</h4><br>
- <h4> The leading architecture used for image recognition and detection tasks is Convolutional Neural Networks (CNNs). Convolutional neural networks consist of several layers with small neuron collections, each of them perceiving small parts of an image. </h4>


```python
import tensorflow as tf
import tensorflow.keras.backend as K
from tensorflow.keras.preprocessing import image
from tensorflow.keras import models
from tensorflow.keras.models import load_model
from tensorflow.keras import models
from PIL import Image 
from skimage.io import imread
import cv2

K.clear_session()
```

    WARNING:tensorflow:From D:\Data Engineering Project\ML Project\aiwork\lib\site-packages\keras\src\backend\common\global_state.py:82: The name tf.reset_default_graph is deprecated. Please use tf.compat.v1.reset_default_graph instead.
    


<h2> <span style="color:red"><b> CHALLENGES </b></span></h2><hr> <h3>1. Cleaning the dataset due to following :</h3>
    
- A lot of misspelled labels.<br>
- Overwhelming number of raw images not related to any class.<br>

<h3> 2. Training model without bias : </h3>

- Similarly looking items with common attributes (Fried samosa and pakode looks familiar.)<br>
- Multiple labels clashing together (Butter Naan and Dal Makhni in one picture, Samosa and Pakode, etc.)<br>
- Unwanted feature selection because of big architectures in transfer learning (Ketchup and green chutney is most common thing alongside pakode, samosa, kathi rolls and what not.)<br>

<hr>

<h2> <span style="color:green"> Finally after 4-5 hours of data cleaning and correcting labels, it was time for fun! </span> </h2>

# <b>IMAGE PROCESSSING</b>

<h2>Reshaping dimensions so we can start processing arrays. </h2>


```python
img = plt.imread('Dataset/train/pizza/006.jpg')
dims = np.shape(img)
matrix = np.reshape(img, (dims[0] * dims[1], dims[2]))
print(np.shape(matrix))
```

    (1080000, 3)



```python
plt.imshow(img)
print("Image shape -> ",dims[:2])
print("Color channels -> ",dims[2])
print("Min color depth : {}, Max color depth {}".format(np.min(img),np.max(img)))
```

    Image shape ->  (900, 1200)
    Color channels ->  3
    Min color depth : 0, Max color depth 255



    
![png](output_8_1.png)


<h2>Plot for visualizing pixel intensities for RGB in color space</h2>

```python
sns.distplot(matrix[:,0], bins=20,color="red",hist_kws=dict(alpha=0.3))
sns.distplot(matrix[:,1], bins=20,color="green",hist_kws=dict(alpha=0.35))
sns.distplot(matrix[:,2], bins=20,color="blue",hist_kws=dict(alpha=0.2))
plt.show()
```


    
![png](output_10_0.png)
    


<h2>Plot for visualizing histogram between 2 color channel</h2>


```python
_ = plt.hist2d(matrix[:,1], matrix[:,2], bins=(50,50))
plt.xlabel('Green channel')
plt.ylabel('Blue channel')
plt.show()
```


    
![png](output_12_0.png)
    


- <span style="font-size:18px;color:white"><b>The pixels between green and blue bands are correlated (as evident from overlapping on above graph), and typically has visible imagery.</b></span>

- <span style="font-size:18px;color:white"><b>Raw band differences will need to be scaled or thresholded </b></span>

<span style="font-size:15px;"><b>Image data consists of variations due to resolution differences between scenes, pixel intensities of an image and the environment around which the image was taken. This area of image processing is critical in today's time with the rise of Artificial intelligence.From motion detection to complex circuits in self driving car, the research requires tremendous amount of work and can be seen as widely growing areas of computer vision.</b><span>


```python
from sklearn import cluster
n_vals=[2,4,6,8]
plt.figure(1, figsize=(12, 8))

for subplot,n in enumerate(n_vals):
    kmeans=cluster.KMeans(n)
    clustered = kmeans.fit_predict(matrix)
    dims = np.shape(img)
    clustered_img = np.reshape(clustered, (dims[0], dims[1]))
    plt.subplot(2,2, subplot+1)
    plt.title("n = {}".format(n), pad = 10,size=18)
    plt.imshow(clustered_img)
    
plt.tight_layout()
```


    
![png](output_15_0.png)
    


<h2> Let's visualize the channel intensity for every cluster we just generated.</h2>


```python
from mpl_toolkits.mplot3d import Axes3D

fig=plt.figure(figsize=(14,10))

ax = [fig.add_subplot(221, projection='3d'),
      fig.add_subplot(222, projection='3d'),
      fig.add_subplot(223, projection='3d'),
      fig.add_subplot(224, projection='3d')]

for plot_number,n in enumerate(n_vals):
    
    kmeans=cluster.KMeans(n)
    clustered = kmeans.fit_predict(matrix)
    x1, y1, z1 = [np.where(clustered == x)[0] for x in [0, 1, 2]]

    plot_vals = [('r', x1),
                 ('b', y1),
                 ('g', z1),
                 ]
    
    for c, channel in plot_vals:
        x = matrix[channel, 0]
        y = matrix[channel, 1]
        z = matrix[channel, 2]
        ax[plot_number].scatter(x, y, z, c=c,s=10)
    
    ax[plot_number].set_xlabel('Blue channel')
    ax[plot_number].set_ylabel('Green channel')
    ax[plot_number].set_zlabel('Red channel')

plt.tight_layout()
```


    
![png](output_17_0.png)
    


<center><span style="font-size:18px;color:antiquewhite"><b>It is evident from above graph, intensity is reduced in the color space as the number of clusters are increased.</b></span></center>

<h4> Brightness normalization is a process that changes the range of pixel intensity values. Applications include photographs with poor contrast due to glare, for example. Normalization is sometimes called contrast stretching or histogram stretching. </h4>


```python
bnorm = np.zeros_like(matrix, dtype=np.float32)
max_range = np.max(matrix, axis=1)
bnorm = matrix / np.vstack((max_range, max_range, max_range)).T
bnorm_img = np.reshape(bnorm, (dims[0],dims[1],dims[2]))
plt.figure(figsize=(8,10))
plt.imshow(bnorm_img)
plt.show()
```


    
![png](output_20_0.png)
    


<h4> Sobel filter is a basic way to get an edge magnitude/gradient image. </h4>
    
**It works by calculating the gradient of image intensity at each pixel within the image. It finds the direction of the largest increase from light to dark and the rate of change in that direction.**


```python
import skimage
# from skimage.feature import greycomatrix, greycoprops
import matplotlib.pyplot as plt
import numpy as np
from skimage import io, color
from skimage.filters import sobel
from skimage.filters import sobel_h

plt.figure(1,figsize=(20,15))
cmap="YlGnBu"
plt.subplot(3,1,1)
plt.imshow(img)

plt.subplot(3,1,2)
plt.imshow(sobel(img[:,:,2]),cmap=cmap)

plt.subplot(3,1,3)
plt.imshow(sobel_h(img[:,:,1]), cmap=cmap)

plt.tight_layout()
```


    
![png](output_22_0.png)
    


<center><span style="font-size:18px;color:blue"><b>Clearly the results are fascinating. We are able to isolate object and detect the edge. </b></span></center>

<h5> Let's apply Principal Component Analysis. It is a linear dimensionality reduction technique that can be utilized for extracting information from a high-dimensional space by projecting it into a lower-dimensional sub-space.</h5>


```python
from sklearn.decomposition import PCA

pca = PCA(3)
pca.fit(matrix)
img_pca = pca.transform(matrix)
img_pca = np.reshape(img_pca, (dims[0], dims[1], dims[2]))

fig = plt.figure(figsize=(8, 8))
plt.imshow(img_pca[:,:,1], cmap=cmap)
```

    <matplotlib.image.AxesImage at 0x1f403291d00>

![png](output_25_1.png)
    


# <b>Exploratory Data Analysis (EDA)</b>

## Let's visualize number of training examples for each food item


```python
main='Dataset/train/'

data=dict()

for i in os.listdir(main):
    sub_dir=os.path.join(main,i)
    count=len(os.listdir(sub_dir))
    data[i]=count
    
  
keys = data.keys()
values = data.values()

colors=["red" if x<= 150 else "green" for x in values]

fig, ax = plt.subplots(figsize=(12,8))
y_pos=np.arange(len(values))
plt.barh(y_pos,values,align='center',color=colors)
for i, v in enumerate(values):
    ax.text(v+1.4, i-0.25, str(v), color=colors[i])
ax.set_yticks(y_pos)
ax.set_yticklabels(keys)
ax.set_xlabel('Images',fontsize=16)
plt.xticks(color='black',fontsize=13)
plt.yticks(fontsize=13)
plt.show()
```


    
![png](output_28_0.png)

<h4>We notice that 2 classes (Pani puri and Kulfi) lacks behind with training data.</h4>

> - <span style="font-size:14px;color:white">Data augmentation helps with classes not having enough training examples by increasing the amount of relevant data in the dataset.</span>
<br><br>
> - <span style="font-size:14px;color:white"> We would be doing what is known as **offline augmentation**. It works on relatively smaller datasets, by increasing the size of the dataset by a factor equal to the number of transformations you perform. (For example, by flipping all my images, it would increase the size of the dataset by a factor of 2).</span>
<br><br>
> - <span style="font-size:14px;color:white"><u>You will get more clarity in the coming section</u></span>


<hr>

## We discussed Data Augmentation before. Let's see how it works:
> 1. Accepting a batch of images used for training.</span>

> 2. Taking this batch and applying a series of random transformations to each image in the batch. (including random rotation, resizing, shearing, etc.)</span>

> 3. Replacing the original batch with the new, randomly transformed batch.</span>

> 4. Training the CNN on this randomly transformed batch. (i.e, the original data itself is not used for training)</span>

<hr>

# <b> MODEL TRAINING </b>


```python
from tensorflow.keras.preprocessing.image import ImageDataGenerator

n_classes = 20
batch_size = 32
img_width, img_height = 299, 299

train_data_dir = 'Dataset/train'

# Data Augmentation with ImageDataGenerator
train_datagen = ImageDataGenerator(
    rescale=1. / 255,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True)

train_generator = train_datagen.flow_from_directory(
    train_data_dir,
    target_size=(img_height, img_width),
    batch_size=batch_size,
    class_mode='categorical')

val_data_dir = 'Dataset/val'

val_datagen = ImageDataGenerator(rescale=1. / 255)

val_generator = val_datagen.flow_from_directory(
    val_data_dir,
    target_size=(img_height, img_width),
    batch_size=batch_size,
    class_mode='categorical')
```

    Found 3996 images belonging to 20 classes.
    Found 1250 images belonging to 20 classes.



```python
class_map = train_generator.class_indices
class_map
```
    {'burger': 0,
     'butter_naan': 1,
     'chai': 2,
     'chapati': 3,
     'chole_bhature': 4,
     'dal_makhani': 5,
     'dhokla': 6,
     'fried_rice': 7,
     'idli': 8,
     'jalebi': 9,
     'kaathi_rolls': 10,
     'kadai_paneer': 11,
     'kulfi': 12,
     'masala_dosa': 13,
     'momos': 14,
     'paani_puri': 15,
     'pakode': 16,
     'pav_bhaji': 17,
     'pizza': 18,
     'samosa': 19}




```python
# print(tf.__version__)
# print(tf.test.gpu_device_name())
```

## Training the model


```python

# Set up the number of training and validation samples
nb_train_samples = 3583
nb_validation_samples = 1089
batch_size = 32  # Define your batch size
n_classes = 20  # Set the number of output classes

# InceptionV3 model setup without the top layer
inception = InceptionV3(weights='imagenet', include_top=False)
x = inception.output
x = GlobalAveragePooling2D()(x)
x = Dense(128, activation='relu')(x)
x = Dropout(0.2)(x)

# Train the model using `fit` instead of `fit_generator`
history = model.fit(
    train_generator,
    steps_per_epoch=nb_train_samples // batch_size,
    validation_data=val_generator,
    validation_steps=nb_validation_samples // batch_size,
    epochs=20,
    verbose=1
)

```
    
    Epoch 20/20
    [1m111/111 0m 32m━━━━━━━━━━━━━━━━━━━━[0m[37m[0m [1m50s[0m 419ms/step - accuracy: 0.8189 - loss: 0.9891 - val_accuracy: 0.9012 - val_loss: 0.7882
## Saving the model


```python
model.save('Model/model_v1_inceptionV3_1.h5')
```


## Accuracy and Loss curves
![png](data/Loss_v1_InceptionV3.png)


# <b> PREDICTIONS </b>

## Load the model


```python
K.clear_session()
path_to_model='Model/model_v1_inceptionV3.h5'
print("Loading the model..")
model = load_model(path_to_model)
print("Done!")




```

    Found 585 images belonging to 20 classes.



```python
# Compile the model with correct metrics format
model.compile(optimizer=SGD(learning_rate=0.0001, momentum=0.9),
              loss='categorical_crossentropy',
              metrics=['accuracy'])  # Correct metrics format
```


```python
# Evaluate the model using the test generator
nb_test_samples=585
scores = model.evaluate(
    test_generator,  # Generator that yields batches of test data
    steps=nb_test_samples // batch_size,  # Number of steps to run the test
    verbose=1
)

# Print the test accuracy
print("Test Accuracy: {:.3f}".format(scores[1]))

```

    18/18 0m 32m━━━━━━━━━━━━━━━━━━━━ 37m 1m19s  838ms/step - accuracy: 0.8510 - loss: 0.7571
    Test Accuracy: 0.866


## Function to predict single image or predict all images from a directory


```python
category={
    0: ['burger','Burger'], 1: ['butter_naan','Butter Naan'], 2: ['chai','Chai'],
    3: ['chapati','Chapati'], 4: ['chole_bhature','Chole Bhature'], 5: ['dal_makhani','Dal Makhani'],
    6: ['dhokla','Dhokla'], 7: ['fried_rice','Fried Rice'], 8: ['idli','Idli'], 9: ['jalegi','Jalebi'],
    10: ['kathi_rolls','Kaathi Rolls'], 11: ['kadai_paneer','Kadai Paneer'], 12: ['kulfi','Kulfi'],
    13: ['masala_dosa','Masala Dosa'], 14: ['momos','Momos'], 15: ['paani_puri','Paani Puri'],
    16: ['pakode','Pakode'], 17: ['pav_bhaji','Pav Bhaji'], 18: ['pizza','Pizza'], 19: ['samosa','Samosa']
}
```

- <h3> Single image prediction </h3>


```python
predict_image('test/burger/images (16).jpg',model)
```

    1m1/1 0m 32m━━━━━━━━━━━━━━━━━━━━ 0m 37m 0m 1m2s 0m 2s/step


```python
predict_image('test/burger/images (16).jpg',model)
```
![png](output_57_0.png)

- <h3> Predicting category </h3>

![png](output_59_1.png)
    


## Let's plot a confusion matrix for all the food items


```python
    # Save and display the plot
    plt.savefig('confusion_matrix.png')
    plt.show()
```

    
![png](output_62_1.png)
    


<h1> <b> MODEL Innear Structure LEARNING VISUALIZATIONS </b></h1>

#### SOME HELPER FUNCTIONS WHICH WILL ENABLE US TO VISUALIZE HOW NEURAL NETWORK WORKS AND PERFORMS!

## MODEL LAYERS 


```python
print("Total layers in the model : ",len(model.layers),"\n")

# We start with index 1 instead of 0, as input layer is at index 0
layers = [layer.output for layer in model.layers[1:11]]
# We now initialize a model which takes an input and outputs the above chosen layers
activations_output = models.Model(inputs=model.input, outputs=layers)
# print(layers)

layer_names = []
for layer in model.layers[1:11]: 
    layer_names.append(layer.name)
    
print("First 10 layers which we can visualize are -> ", layer_names)

```

    Total layers in the model :  315 
    
    First 10 layers which we can visualize are ->  ['conv2d', 'batch_normalization', 'activation', 'conv2d_1', 'batch_normalization_1', 'activation_1', 'conv2d_2', 'batch_normalization_2', 'activation_2', 'max_pooling2d']


<span style="font-size:14px;color:white"><b>Our model has 315 layers with InceptionV3 architecture!</b></span>
<br><br>

### <b>LAYER WISE ACTIVATIONS</b>


```python
food = 'Dataset/val/pizza/155.jpg'
activations = get_activations(food,activations_output)
show_activations(activations, layer_names)
```



## Let's show the activation outputs of Conv2D layer (we have three of them in first 10 layers) to compare how layers get abstract with depth.


```python
activation_conv()
```

![png](output_72_0.png)

## This time let's visualize some other food item's layer.


```python
food = '../input/indian-food-classification/dataset/Dataset/val/idli/065.jpg'
activations = get_activations(food,activations_output)
show_activations(activations, layer_names)
```


    
![png](output_75_0.png)
    


<h2><span style="color:blue">NOW THIS IS WHERE THINGS WILL GET INTERESTING!</span></h2>

- <span style="font-size:17px;"> So far we were doing activation maps visualization which helped us understand how the input is transformed from one layer to another as it goes through several operations. </span>

- <span style="font-size:17px;"> At the end of training, we want the model to classify or detect objects based on features which are specific to the class.</span>

- <span style="font-size:17px;"> To validate how model attributes the features to class output, we can generate heat maps using gradients to find out which regions in the input images were instrumental in determining the class. </span>

# <b>GENERATING HEATMAPS</b>

```python
pred3=get_attribution('test/chai/images (3).jpg')
```

![png](output_80_0.png)
    



```python
pred4=get_attribution('test/jalebi/images (4).jpg')
```


    
![png](output_81_0.png)
    



```python
pred5=get_attribution('test/chole_bhature/images (10).jpg')
```
![png](output_82_0.png)
    


<ol>
    <h5>
        <li>In the above plot, we see on left the input image passed to the model, heat map in the middle and the class activation map on right</li><br>
        <li>Heat map gives a visual of what regions in the image were used in determining the class of the image</li><br>
        <li>Now it's clearly visible what a model looks for in an image if it has to be classified as an idli!</li>
    </h5>
</ol>

## Downloading random image from net to predict and generate heatmap


```python
!wget -O download.jpg https://www.cookwithmanali.com/wp-content/uploads/2015/01/Restaurant-Style-Dal-Makhani-Recipe.jpg
    
model_load = load_model('Model/model_v1_inceptionV3.h5')
```
    
    download.jpg        100%[===================>] 160.20K  --.-KB/s    in 0.02s   
    
    2021-06-02 17:45:32 (9.52 MB/s) - ‘download.jpg’ saved [164040/164040]
    


<span style="font-size:16px;"><b>This image was given to confuse the model as both classes <u>Butter naan</u> and <u>Dal makhni</u> are present. But it predicted dal makhni and why was that?<br><br>Because the output layer inside the model computed high activations for dal makhni object. It does happen sometimes due to centralized nature of model's focus area. </b></span>


```python
pred = get_attribution('download.jpg')
```

![png](output_87_0.png)
    

