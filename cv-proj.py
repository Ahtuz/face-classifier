import cv2
import os
import math
import numpy as np

def get_path_list(root_path):
    train_names = []

    for name in os.listdir(root_path):
        train_names.append(name)
    
    return train_names

def get_class_names(root_path, train_names):
    image_path_list = []
    image_classes_list = []

    for class_id, name in enumerate(os.listdir(root_path)):
        image_class_path = root_path + '/' + name

        for image in os.listdir(image_class_path):
            image_path_list.append(root_path + '/' + name + '/' + image)
            image_classes_list.append(class_id)

    return image_path_list, image_classes_list

def get_train_images_data(image_path_list):
    train_image_list = []

    for imagepath in image_path_list:
        train_image_list.append(cv2.imread(imagepath)) 

    return train_image_list

def detect_faces_and_filter(image_list, image_classes_list=None):
    train_face_grays = []
    test_faces_rects = []
    filtered_classes_list = []

    face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

    for i, image in enumerate(image_list):
        temp = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        detected_faces = face_cascade.detectMultiScale(temp, scaleFactor=1.2, minNeighbors=5)

        if (len(detected_faces) != 1) :
            continue
            
        for face in detected_faces:
            x, y, w, h = face

            face_gray_rect = temp[y:y+h , x:x+w]
            train_face_grays.append(face_gray_rect)

            test_faces_rects.append(face)

            if image_classes_list is not None:
                
                filtered_classes_list.append(image_classes_list[i])
        
            else:
                filtered_classes_list.append([''])


    return train_face_grays, test_faces_rects, filtered_classes_list

def train(train_face_grays, image_classes_list):
    face_detect_object = cv2.face.LBPHFaceRecognizer_create()

    face_detect_object.train(train_face_grays, np.array(image_classes_list))

    return face_detect_object

def get_test_images_data(test_root_path, image_path_list):
    test_image_list = []

    for image in os.listdir(test_root_path):
        
        test_image_list.append(cv2.imread(test_root_path+'/'+image))

    return test_image_list

def predict(classifier, test_faces_gray):
    result_list = []

    for image in test_faces_gray:
        result, _ = classifier.predict(image)

        result_list.append(result)

    return result_list


def draw_prediction_results(predict_results, test_image_list, test_faces_rects, train_names):
    drawn_list = []

    for i,face in enumerate(test_faces_rects):

        x, y, w, h = face

        cv2.rectangle(test_image_list[i], (x,y), (x+w,y+h), (0,255,0), 2)

        newimg = cv2.putText(test_image_list[i], train_names[predict_results[i]], (x,y-10), 1, 1, (0,255,0))

        drawn_list.append(newimg)
    
    return drawn_list


def combine_results(predicted_test_image_list):
    images = predicted_test_image_list
    
    final_image_result = images[0]

    for i in range(1, len(images)):
        final_image_result = np.hstack((final_image_result, images[i]))

    return final_image_result

def show_result(image):
    cv2.imshow("results",image)
    cv2.waitKey(0)

if __name__ == "__main__":

    train_root_path = "dataset/train"
    
    train_names = get_path_list(train_root_path)
    image_path_list, image_classes_list = get_class_names(train_root_path, train_names)
    train_image_list = get_train_images_data(image_path_list)
    train_face_grays, _, filtered_classes_list = detect_faces_and_filter(train_image_list, image_classes_list)
    classifier = train(train_face_grays, filtered_classes_list)

    test_root_path = "dataset/test"

    test_names = get_path_list(test_root_path)
    test_image_list = get_test_images_data(test_root_path, test_names)
    test_faces_gray, test_faces_rects, _ = detect_faces_and_filter(test_image_list)
    predict_results = predict(classifier, test_faces_gray)
    predicted_test_image_list = draw_prediction_results(predict_results, test_image_list, test_faces_rects, train_names)
    final_image_result = combine_results(predicted_test_image_list)
    show_result(final_image_result)