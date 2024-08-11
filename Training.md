# Creating a dataset and training a classification model

## 1. Navigate to the Label/Train page in CBASv2.
<p align="left">
    <img src=".//assets/labeltrain.png" alt="CBAS Labeling" style="width: 250px; height: auto;">
</p>
<p align="left"> 

## 2. Hit the plus sign at the bottom of the page and fill out the dataset form.
1. The dataset name should be one word with underscores instead of spaces (e.g. `dataset_1`)
2. Separate behaviors of interest with a semicolon (`;`) with or without a space
   1. Example: `eating; drinking; rearing`
3. Select the camera directories to be included in the dataset. New videos added to these directories will be automatically added to the dataset.
<p align="left">
    <img src=".//assets/createdataset.png" alt="CBAS Dataset" style="width: 250px; height: auto;">
</p>
<p align="left"> 
4. Hit the 'Create' button to finalize the dataset.
<p align="left">
    <img src=".//assets/finalizedataset.png" alt="CBAS Dataset Final" style="width: 250px; height: auto;">
</p>
<p align="left"> 

## 3. Hit the 'Label' button on the dataset to begin labeling videos in the dataset.
<p align="left">
    <img src=".//assets/labelingvids.png" alt="CBAS Labeling" style="width: 525px; height: auto;">
</p>
<p align="left"> 

1. Video surfing:
   1. If you hit the left and right arrows on your keyboard, the video frame will change in those directions (forward and back).
   2. If you hit the up arrow on your keyboard, the 'surf speed' will double (each left and right arrow will go forward or back two frames).
   3. This may still be too slow. Try hitting the up arrow 2-3 times to make the surf speed even faster and move through the video with the left and right arrows.
   4. Hitting the down arrow on your keyboard will halve the surf speed.
   5. If you hold the **Ctrl** key and hit the left or right arrows, CBAS will direct you to another video in the dataset.
2. Labeling a behavior
   1. Your behaviors of interest will be listed on the right side of the screen.
   2. `Code` refers to the keyboard key binding of a particular behavior
   3. `Count` refers to the number of instances of a behavior that you have labeled.
   4. Surf through the video to find the start of an instance of a behavior you want to label.
   5. Hit the Code of that behavior on your keyboard **once**.
      1. Example: if your first behavior is `eating`, the Code will be `1`. Press 1 on your keyboard.
   6. Surf to the end of the behavior instance (e.g. when the behavior ends).
   7. Hit the Code of that behavior on your keyboard **once** again.
      1. The frames corresponding to that behavior will be colored at the bottom of the video image.
   8. You have successfully labeled an example of a behavior!
   9. If you make an error, press the **Backspace** key.
      1.  This will delete the entire instance of the behavior.
      2.  The colors corresponding to that behavior at the bottom of the video image will disappear.

## 4. When you have labeled an adequate number of behaviors:
1. A good number would be >100 instances per behavior across multiple animals, but CBAS can also perform well on a smaller dataset.
2. Navigate back to the Label/Train page by clicking the link at the top of CBASv2.
3. Hit the 'Train' button and wait for the Precision, Recall, and F1 score values to update.
<p align="left">
    <img src=".//assets/f1.png" alt="CBAS F1" style="width: 250px; height: auto;">
</p>
<p align="left"> 

## 5. Hit the 'Infer' button and select directories for the classification model to infer.
1. Any video with a `_cls.h5` file in the same directory will be classified using the trained model.
2. The `_outputs.csv` file created will contain the model's prediction probabilities for each behavior at each frame in the classified video.
<p align="left">
    <img src=".//assets/outputs.png" alt="CBAS outputs" style="width: 610px; height: auto;">
</p>
<p align="left"> 
