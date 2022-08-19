### University of Southampton
## Computer Vision Group Coursework - Scene Recognition

This project was used as an introduction to image recognition by beginning with scene recognition. 
At first we used very simple methods of scene recognition. We then moved on to more complex techniques
such as 'bags of quantized local features'. We used a very common benchmarking technique in which we 
were provided with a set of labelled images from which we developed and tuned our classifiers. These classifiers
were then tested against a set of unlabelled images to give a score of accuracy.

### Authors
- Charles Gilbert
- Isaac Whale
- Roberto Martinez Camacho
- Bastin Punneliparambil

### Scenes
The training data was labelled with 15 different scene descriptions which were as follows
![Scenes](https://i.imgur.com/jB6Gbke.png?raw=true)

### Run 1
Accuracy: 23% (Random Approach 6.6%)
<details closed>
<summary>Description</summary>
<br>
For the implementation of a K nearest neighbours classifier, we used the KNNAnnotator class provided by OpenImaj. The constructor for this class required a feature extractor, and a distance comparator. The comparators with the best accuracy were CITY_BLOCK, SUM_SQUARE and COSINE_DIST so we opted to use CITY_BLOCK as we found that this gave the best accuracy on average. For the feature extractor we implemented a class called TinyImageExtractor, which extended the FImage2FloatFV class in OpenImaj. This extractor class provides an override of the extractFeature method which takes an image and returns a FloatFV (OpenImaj float feature vector class). The method crops the image into a square, then resizes it to 16x16 pixels using the ResizeProcessor.zoomInplace() static method, the image is then normalised to ensure that all pixel values fall between the values 0 and 1, and finally the image is processed by a MeanCentre processor which subtracts the mean from all pixel values, making the mean pixel value become 0. This processed image is then passed to the extract feature method of the FImage2FloatFV class by using super.extractFeature() to return an image vector for the tiny image.

After constructing the KNNAnnotator, it is then trained on the provided training dataset. It’s k value is set to 9 as we found it provided the best results, though there was a lot of variability in the accuracy as we tested using a random split of the training set, so we can’t be sure. 

Testing was done by splitting the training set into two equally sized groups, using a GroupedRandomSplitter, and an adapted version of the ClassificationEvaluator code provided in chapter 12 of the OpenImaj tutorial was used, which calculated the accuracy of each run allowing us to compare different parameters.Overall we achieved an accuracy of roughly 23% with this approach, which is significantly better than a random approach (6.6%) but it is still not very good when compared to the other methods in this coursework.
</details>

### Run 2
Accuracy: (Random Approach 6.6%)
<details closed>
<summary>Description</summary>
<br>
For this run, after importing the training and testing datasets, we created a method called trainQuantiser(training_dataset) that creates an assigner which will be used later to create the feature extractor. This method creates an empty list of patches; then, for every image in the dataset, the patches from the image are extracted and added to the list of patches using a class called PatchExtractor and its extractFeatureVector method. The PatchExtractor splits an image into 8x8 patches sampled every 4 pixels, each patch is put through a MeanCenter processor, normalised and converted into a DoubleFV. The complete list of DoubleFVs of the patches is returned. Then the trainQuantiser() method caps the number of patches to 10,000 patches, to avoid using too many computational resources. After this, the method uses K-means to divide the patches into 500 clusters. Finally, this method returns an assigner.

Once the assigner is created, the POWExtractor method takes this assigner as a parameter to create an extractor. The extractor takes an image, extracts all its patches and uses the HardAssigner trainQuantiser() to assign each patch to a cluster. It then represents all these labelled patches using BagOfVisualWords and aggregates these values to create an Integer Feature Vector representing how many patches which belong to each cluster are shown in the image.

This extractor is then used with the LiblinearAnnotator class to create 15 One Vs All classifiers which we then use on the testing dataset and write the results to a text file.
</details>

### Run 3
Accuracy 74% (Random Approach 6.6%)
<details closed>
<summary>Description</summary>
<br>
The feature extraction technique we used for Run 3 is Dense SIFT with spatial pooling, and the classifier we chose to use is a non-linear SVM, using a non-linear classifier with a Homogenous Kernel Map. After importing the training and testing datasets we created a DenseSIFT object with step size 3, and then a PyramidDenseSIFT object that uses this DenseSIFT object, and we set its window size to 7 pixels.

The assigner is created using the trainQuatiser(dataset, pyramidDenseSIFT) method, which creates an empty list of key features, iterates through every image in the dataset, analyses the images using the pyramid dense SIFT object and adds the features obtained to the list of key features. This method then uses K-means to group the features into 300 clusters.

The feature extractor is created using a PHOWExtractor(pyramidDenseSIFT, assigner) method that is very similar to the one used in the OpenImaj tutorial. The main difference is that we tried using the PyramidSpatialAggregator and found out that performance improves compared to the BlockSpatialAggregator, however we couldn’t do many tests using different block sizes due to the program taking a long time to display results. Once the extractor was created, we wrapped it in a homogeneous kernel map, which drastically improved the accuracy obtained compared with not wrapping it with this kernel map.

This extractor is then used on the linear classifier to train the neural network and test it against the testing dataset, and finally it displays the accuracy results, which were around 74%.
</details>

### Contributions
We split into two pairs to complete run 1 and run 2, and then in the same pairs we each attempted an approach to run 3 and settled with the method with highest accuracy. All team members contributed equally.
