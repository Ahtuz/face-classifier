import cv2
import os
import math
import numpy as np

def get_path_list(root_path):
    '''
        To get a list of path directories from root path

        Parameters
        ----------
        root_path : str
            Location of root directory
        
        Returns
        -------
        list
            List containing the names of the sub-directories in the
            root directory
    '''

    root_path = 'dataset/train'
    train_names = []

    for name in os.listdir(root_path):
        train_names.append(name)
    
    return train_names
    # return Aaron_Eckhart, Aaron_Guiel, Aaron_Patterson, ...

def get_class_names(root_path, train_names):
    '''
        To get a list of train images path and a list of image classes id

        Parameters
        ----------
        root_path : str
            Location of images root directory
        train_names : list
            List containing the names of the train sub-directories
        
        Returns
        -------
        list
            List containing all image paths in the train directories
        list
            List containing all image classes id
    '''

    image_path_list = []
    image_classes_list = []

    for class_id, name in enumerate(os.listdir(root_path)):
        image_class_path = root_path + '/' + name
        # dataset/train/Aaron_Eckhart
        image_classes_list.append(class_id)
        # Aaron_Eckhart

        for image in os.listdir(image_class_path):
            image_path_list.append(root_path + '/' + name + '/' + image)
            # dataset/train/Aaron_Eckhart/Aaron_Eckhart_0001.jpg

        return image_path_list, image_classes_list

def get_train_images_data(image_path_list):
    '''
        To load a list of train images from given path list

        Parameters
        ----------
        image_path_list : list
            List containing all image paths in the train directories
        
        Returns
        -------
        list
            List containing all loaded train images
    '''

    image_list = []

    for image in image_path_list:
        image_list.append(cv2.imread(image, 0)) 

    return image_list

def detect_faces_and_filter(image_list, image_classes_list=None):
    '''
        To detect a face from given image list and filter it if the face on
        the given image is more or less than one

        Parameters
        ----------
        image_list : list
            List containing all loaded images
        image_classes_list : list, optional
            List containing all image classes id
        
        Returns
        -------
        list
            List containing all filtered and cropped face images in grayscale
        list
            List containing all filtered faces location saved in rectangle
        list
            List containing all filtered image classes id
    '''

    train_face_grays = []
    test_faces_rects = []
    train_img_id = []

    face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

    for i, image in enumerate(image_list):
        detected_faces = face_cascade.detectMultiScale(image, scaleFactor=1.2, minNeighbors=5)

        if len(detected_faces) < 1:
            continue
            
        for face in detected_faces:
            x, y, w, h = face
            test_faces_rects.append(face)
            train_face_grays.append(image[y:y+h , x:x+w])
            train_img_id.append(image_classes_list[i])

    return train_face_grays, test_faces_rects, train_img_id

def train(train_face_grays, image_classes_list):
    '''
        To create and train classifier object

        Parameters
        ----------
        train_face_grays : list
            List containing all filtered and cropped face images in grayscale
        image_classes_list : list
            List containing all filtered image classes id
        
        Returns
        -------
        object
            Classifier object after being trained with cropped face images
    '''
    classifier = cv2.face.LBPHFaceRecognizer_create()

    classifier.train(train_face_grays, np.array(image_classes_list))

    return classifier

def get_test_images_data(test_root_path, image_path_list):
    '''
        To load a list of test images from given path list

        Parameters
        ----------
        test_root_path : str
            Location of images root directory
        image_path_list : list
            List containing all image paths in the test directories
        
        Returns
        -------
        list
            List containing all loaded test images
    '''
    image_list = []

    for image in os.listdir(test_root_path):
        
        image_list.append(cv2.imread(image, 0))

    return image_list

def predict(classifier, test_faces_gray):
    '''
        To predict the test image with classifier

        Parameters
        ----------
        classifier : object
            Classifier object after being trained with cropped face images
        train_face_grays : list
            List containing all filtered and cropped face images in grayscale

        Returns
        -------
        list
            List containing all prediction results from given test faces
    '''
    predict_results = []

    for image in test_faces_gray:
        result, confidence = classifier.predict(image)

        confidence = math.floor(confidence * 100) / 100

        predict_results.append(str(result + " " + confidence + "% "))

    return predict_results


def draw_prediction_results(predict_results, test_image_list, test_faces_rects, train_names):
    '''
        To draw prediction results on the given test images

        Parameters
        ----------
        predict_results : list
            List containing all prediction results from given test faces
        test_image_list : list
            List containing all loaded test images
        test_faces_rects : list
            List containing all filtered faces location saved in rectangle
        train_names : list
            List containing the names of the train sub-directories

        Returns
        -------
        list
            List containing all test images after being drawn with
            prediction result
    ''' 

    predicted_test_image_list = []

    for i,face in enumerate(test_faces_rects):

        x, y, w, h = face

        cv2.rectangle(test_image_list[i], (x,y), (x+w,y+h), (255,0,255), 2)

        predicted_test_image = cv2.putText(test_image_list[i], predict_results[i], (x,y-10), 2, 1, (255,255,255))

        predicted_test_image_list.append(predicted_test_image)
    
    return predicted_test_image_list


def combine_results(predicted_test_image_list):
    '''
        To combine all predicted test image result into one image

        Parameters
        ----------
        predicted_test_image_list : list
            List containing all test images after being drawn with
            prediction result

        Returns
        -------
        ndarray
            Array containing image data after being combined
    '''
    ndarray = []

    for image in predicted_test_image_list:
        ndarray = np.hstack((img, image))
        
    return ndarray

def show_result(image):
    '''
        To show the given image

        Parameters
        ----------
        image : ndarray
            Array containing image data
    '''

    cv2.imshow("results",image)
    cv2.waitKey(0)

'''
You may modify the code below if it's marked between

-------------------
Modifiable
-------------------

and

-------------------
End of modifiable
-------------------
'''
if __name__ == "__main__":
    '''
        Please modify train_root_path value according to the location of
        your data train root directory

        -------------------
        Modifiable
        -------------------
    '''

    train_root_path = "dataset/train"

    '''
        -------------------
        End of modifiable
        -------------------
    '''
    
    train_names = get_path_list(train_root_path)
    image_path_list, image_classes_list = get_class_names(train_root_path, train_names)
    train_image_list = get_train_images_data(image_path_list)
    train_face_grays, _, filtered_classes_list = detect_faces_and_filter(train_image_list, image_classes_list)
    classifier = train(train_face_grays, filtered_classes_list)

    '''
        Please modify test_image_path value according to the location of
        your data test root directory

        -------------------
        Modifiable
        -------------------
    '''

    test_root_path = "dataset/test"

    '''
        -------------------
        End of modifiable
        -------------------
    '''

    test_names = get_path_list(test_root_path)
    test_image_list = get_test_images_data(test_root_path, test_names)
    test_faces_gray, test_faces_rects, _ = detect_faces_and_filter(test_image_list)
    predict_results = predict(classifier, test_faces_gray)
    predicted_test_image_list = draw_prediction_results(predict_results, test_image_list, test_faces_rects, train_names)
    final_image_result = combine_results(predicted_test_image_list)
    show_result(final_image_result)