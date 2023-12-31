# DCGAN-Flowers-Generation
Using DCGAN to generate flowers that has never been sean before in data.

![](https://github.com/OlaPietka/DCGAN-Flowers-Generation/blob/main/data/garden.png)

## Data
This project leverages the [102 Flowers Dataset](https://www.robots.ox.ac.uk/~vgg/data/flowers/102/). The dataset was prepared by manually selecting images that contain only one flower facing upwards. Images were cropped to a square shape, centered around the flower so that the center of the flower aligns with the center of the image, and resized to a resolution of 128x128 pixels. This approach ensured better results as the model could focus on the flower's shape and characteristics rather than background distractions or multiple flowers in a frame. Finding the right images did limit the final dataset size. To counteract this, some data augmentation techniques were applied.

## Model
The model used is DCGAN, with generator and discriminator architectures depicted below.

### Generator
![](https://github.com/OlaPietka/DCGAN-Flowers-Generation/blob/main/data/generator_architecture.png)

### Discriminator
![](https://github.com/OlaPietka/DCGAN-Flowers-Generation/blob/main/data/discriminator_architecture.png)

## Learning
The criterion for training is BCELoss because it is effective for binary classification tasks inherent to the discriminator in GANs. The model was trained for 1000 epochs with a batch size of 128. Both the generator and the discriminator had a learning rate set to 0.0005. The latent space has a size of 300. To optimize performance, a learning rate scheduler was employed, reducing the learning rate by 0.9 every 50 epochs.

![](https://github.com/OlaPietka/DCGAN-Flowers-Generation/blob/main/data/learning.gif)
