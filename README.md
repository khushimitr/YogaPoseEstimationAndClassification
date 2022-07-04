# Yoga Pose Estimation and Classification

This project takes on the problem of identifying the keypoints of the person
doing yoga pose and classifying it. Trained on 48 classes of yoga pose, our model
achieves 91% accuracy and can be used in real-time as well.



## Demo

<p align="center">
  <img width="460" height="300" src="https://github.com/khushimitr/YogaPoseEstimationAndClassification/blob/main/Inferences/Bhujangasana.gif">
</p>



## Dependencies
To run code in this repo properly, you will need:
* openCV
* Tensorflow
* Keras
* csv
* numpy
* pandas
* wget

## Classes
List of classes used can be viewed [here](https://github.com/khushimitr/YogaPoseEstimationAndClassification/blob/main/Models/pose_labels.txt).

## Prepare your dataset
```bash
    datasets
    └── Train 
        ├── class1
        ├── class2
        └── class3
    └── Test
        ├── class1
        ├── class2
        └── class3

```

Split the data into training and testing, each having n number of classes that you
want to classify.

## Generate Keypoints per image
Use class Movenet and Preprocessing provided and for each image run the 
movenet thunder model to identify keypoints and store them in a csv file.

```
Note: Movenet thunder detect 17 keypoints and give their coordinates and score.
```

Now, you should get two csv files with names train.csv and test.csv.

## Weights

Weights for the presented model can be found [here](https://github.com/khushimitr/YogaPoseEstimationAndClassification/blob/main/Models/weights_yoga_dataset.best2.hdf5).

## Load in the model

* Create the model architecture as described in the ipynb file given.
* Download the weights provided.
``` bash
    model.load_weights(weights_path)
```
give the path to where you have downloaded teh weights `weights_path`

## Inference with pretrained weights

### For a image
* Use the get_inference_img method
```
    get_inference_img(img_path,y_true)
```

`img_path` is the path of the image that is to be predicted and `y_true` is
the actual label of the image.


### for a real-time webcam
```
class_to_match = 'class_name'

cap = cv2.VideoCapture(0)

cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1500)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1500)


while cap.isOpened():
    ret,frame = cap.read()
    
    img = frame.copy()
        
    rgb_tensor = tf.convert_to_tensor(img, dtype=tf.float32)

    person = detect(rgb_tensor)
    detection_threshold = 0.1
    
    pose_landmarks = np.array([[keypoint.coordinate.x, keypoint.coordinate.y, keypoint.score] for keypoint in person.keypoints], dtype=np.float32)

    test_coord = pose_landmarks.flatten().tolist()

    keypoints = []
    for kp in person.keypoints:
        keypoints.append([kp.coordinate.x,kp.coordinate.y,kp.score])

    embedding = landmarks_to_embedding(tf.reshape(tf.convert_to_tensor(test_coord),(1,51)))
    embed = tf.reshape(embedding, (34))
    test_pro = tf.reshape(tf.convert_to_tensor(embed),(1,34))

    pred_probs = model.predict(test_pro)
    pred_classes = pred_probs.argmax(axis = 1)
    predicted = class_names[pred_classes[0]]
    # rendering
    score = 100*tf.reduce_max(pred_probs)
    str_score = str(int(score))

    match = False
    if(class_to_match == predicted):
        match = True
        
    draw_connections(img,keypoints,EDGES,0.1,score,match)
    draw_keypoints(img,keypoints,0.1,score,match)
    
    cv2.putText(img,str_score,(20,700),cv2.FONT_HERSHEY_SIMPLEX,1,(255,0,0),2)
    cv2.putText(img,class_to_match,(1000,50),cv2.FONT_HERSHEY_SIMPLEX,1,(255,0,0),2)
    
    cv2.imshow('Detections',img)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
    
cap.release()
cv2.destroyAllWindows()

```

`class_to_match` takes the class that you want your real time feed to match.

## Confusion Matrix

Since Yoga poses can have small pysical differentiations, 
so it is interesting to see where our model gets confused:

![App ScreenShot](https://github.com/khushimitr/YogaPoseEstimationAndClassification/blob/main/Inferences/confusion_matrix.png)

