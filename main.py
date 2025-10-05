# -*- coding: utf-8 -*-
"""
Created on Thu Jan 28 00:44:25 2021

@author: chakati
"""
import csv
import cv2
import numpy as np
import pandas as pd
import os
import tensorflow as tf
from frameextractor import frameExtractor

from sklearn.metrics.pairwise import cosine_similarity


## import the handfeature extractor class
from handshape_feature_extractor import HandShapeFeatureExtractor

# =============================================================================
# Get the penultimate layer for trainig data
# =============================================================================
# your code goes here
# Extract the middle frame of each gesture video
def get_test_video_paths(test_data_dir="TestData"):
    """
    Get all video paths from the TestData directory.

    Args:
        test_data_dir (str): Path to the test data directory (default: "TestData")

    Returns:
        list: List of full paths to all video files in TestData directory
    """
    test_video_paths = []

    if os.path.exists(test_data_dir):
        for filename in os.listdir(test_data_dir):
            if filename.endswith('.mp4'):
                video_path = os.path.join(test_data_dir, filename)
                test_video_paths.append(video_path)
    else:
        print(f"TestData directory '{test_data_dir}' not found!")

    return sorted(test_video_paths)


def get_label_from_filename(name):
    if '-0' in name:
        return '0'
    elif '-1' in name:
        return '1'
    elif '-2' in name:
        return '2'
    elif '-3' in name:
        return '3'
    elif '-4' in name:
        return '4'
    elif '-5' in name:
        return '5'
    elif '-6' in name:
        return '6'
    elif '-7' in name:
        return '7'
    elif '-8' in name:
        return '8'
    elif '-9' in name:
        return '9'
    elif 'FanOn' in name:
        return 'FanOn'
    elif 'FanOff' in name:
        return 'FanOff'
    elif 'LightOn' in name:
        return 'LightOn'
    elif 'LightOff' in name:
        return 'LightOff'
    elif 'SetThermo' in name:
        return 'SetThermo'
    elif 'IncreaseFanSpeed' in name:
        return 'IncreaseFanSpeed'
    elif 'DecreaseFanSpeed' in name or 'DecereaseFanSpeed' in name:
        return 'DecreaseFanSpeed'
    else:
        print(name)
        return 'unknown'


def extract_middle_frames(gestureVideos="TestData", frames_path="framePath"):
    videopaths = get_test_video_paths(gestureVideos)
    handShapeFeatureExtractor = HandShapeFeatureExtractor.get_instance()
    gesture_labels = []
    features = []
    images = []
    for i in range(len(videopaths)):
        gesture_label = get_label_from_filename(videopaths[i])
        gesture_labels.append(gesture_label)
        frameExtractor(videopaths[i],frames_path,i)
        image = cv2.imread(frames_path + "/%#05d.png" % (i+1), cv2.IMREAD_GRAYSCALE)
        images.append(image)
        feature = handShapeFeatureExtractor.extract_feature(image)
        features.append(feature)
    return (gesture_labels, images, features)


def extractMiddleFrameFromTrainingData():
    return extract_middle_frames("traindata", "framePathTrain")


# =============================================================================
# Get the penultimate layer for test data
# =============================================================================
# your code goes here 
# Extract the middle frame of each gesture video

def extractMiddleFrameFromTestData():
    return extract_middle_frames("TestData", "framePathTest")


# =============================================================================
# Recognize the gesture (use cosine similarity for comparing the vectors)
# =============================================================================
def recognize_gesture_cosine(test_feature, training_features, gesture_labels):
    """
    Recognize gesture using cosine similarity with proper array flattening

    Args:
        test_feature: Feature vector from test image (can be multi-dimensional)
        training_features: Feature vectors from training data (list or array)
        gesture_labels: Labels corresponding to training features

    Returns:
        predicted_label: The recognized gesture label
        similarity_score: The highest similarity score
    """
    # Flatten test feature to 1D and reshape to 2D for sklearn
    test_feature_flat = test_feature.flatten().reshape(1, -1)

    # Flatten training features and convert to 2D array
    training_features_flat = []
    for feature in training_features:
        training_features_flat.append(feature.flatten())

    training_features_flat = np.array(training_features_flat)

    # Debug prints (remove these after fixing)
    print(f"Test feature shape: {test_feature_flat.shape}")
    print(f"Training features shape: {training_features_flat.shape}")

    # Calculate cosine similarity
    similarities = cosine_similarity(test_feature_flat, training_features_flat)

    # Find the index of highest similarity
    best_match_index = np.argmax(similarities)

    # Get the predicted label and similarity score
    predicted_label = gesture_labels[best_match_index]
    similarity_score = similarities[0][best_match_index]

    return predicted_label, similarity_score

def save_results(predicted_labels, filename="results.csv"):
    """
    Save predicted labels in a 51x1 matrix to CSV file (no header).

    Args:
        predicted_labels (list or array): List of predicted gesture labels
        filename (str): Output CSV filename (default: "results.csv")
    """
    # Ensure we have exactly 51 predictions
    if len(predicted_labels) != 51:
        print(f"Warning: Expected 51 predictions, but got {len(predicted_labels)}")
        # Pad with 'Unknown' if fewer than 51, truncate if more
        if len(predicted_labels) < 51:
            predicted_labels = list(predicted_labels) + ['Unknown'] * (51 - len(predicted_labels))
        else:
            predicted_labels = predicted_labels[:51]

    # Write to CSV using built-in csv module (no header)
    with open(filename, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)

        for label in predicted_labels:
            writer.writerow([label])  # Just write the label, no header

    print(f"Results saved to {filename}")


def main():
    trainingData = extractMiddleFrameFromTrainingData()
    testData = extractMiddleFrameFromTestData()

    gesture_labels = np.array(trainingData[0])
    trainingFeatures = np.array(trainingData[2])
    testFeatures = np.array(testData[2])
    results = []
    gestureMap = {}
    gestureMap['0'] = '0'
    gestureMap['1'] = '1'
    gestureMap['2'] = '2'
    gestureMap['3'] = '3'
    gestureMap['4'] = '4'
    gestureMap['5'] = '5'
    gestureMap['6'] = '6'
    gestureMap['7'] = '7'
    gestureMap['8'] = '8'
    gestureMap['9'] = '9'
    gestureMap['DecreaseFanSpeed'] = '10'
    gestureMap['FanOff'] = '11'
    gestureMap['FanOn'] = '12'
    gestureMap['IncreaseFanSpeed'] = '13'
    gestureMap['LightOff'] = '14'
    gestureMap['LightOn'] = '15'
    gestureMap['SetThermo'] = '16'

    for testFeature in testFeatures:
        result = recognize_gesture_cosine(testFeature, trainingFeatures, gesture_labels)
        results.append(gestureMap[result[0]])
    print(results)
    save_results(results)


if __name__ == "__main__":
    main()