#ifndef INFERENCE_H
#define INFERENCE_H

#include <fstream>
#include <vector>
#include <string>
#include <random>

#include <opencv2/imgproc.hpp>
#include <opencv2/opencv.hpp>
#include <opencv2/dnn.hpp>

//std::vector<std::string> classes{"person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck", "boat", "traffic light", "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat", "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella", "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball", "kite", "baseball bat", "baseball glove", "skateboard", "surfboard", "tennis racket", "bottle", "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair", "couch", "potted plant", "bed", "dining table", "toilet", "tv", "laptop", "mouse", "remote", "keyboard", "cell phone", "microwave", "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase", "scissors", "teddy bear", "hair drier", "toothbrush"};


struct Detection
{
    int class_id{0};
    std::string className{};
    float confidence{0};
    cv::Rect box{};
    cv::Scalar color{};
    std::vector<cv::Point2f> keypoint;
};


class Inference
{
    public:
    Inference(const std::string &onnxmodelpath,const cv::Size modelInputShape,const int keypoints_Num);
    std::vector<Detection> runinference(const cv::Mat &input);

    private:
    float modelConfidenceThreshold = 0.25;
    float modelScoreThreshold = 0.45;
    float modelNMSThreshold = 0.50;
    float modelkeypointThreshold = 0.65;
    int modelkeypoint_Num;
    std::string modelPath{};
    cv::Size2f modelShape{};
    cv::dnn::Net net;
    std::vector<std::string> classes{"person"};
    void loadOnnxNetwork();
    cv::Mat formatToSquare(const cv::Mat &source);


};

#endif