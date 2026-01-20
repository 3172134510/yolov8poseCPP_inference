#include "inference.h"


Inference::Inference(const std::string &onnxmodelpath,const cv::Size modelInputShape,const int keypoints_Num)
{
    modelPath = onnxmodelpath;
    modelShape = modelInputShape;
    modelkeypoint_Num = keypoints_Num;
    loadOnnxNetwork();
}

void Inference::loadOnnxNetwork()
{
    net = cv::dnn::readNetFromONNX(modelPath);
    net.setPreferableBackend(cv::dnn::DNN_BACKEND_OPENCV);
    net.setPreferableTarget(cv::dnn::DNN_TARGET_CPU);
    std::cout<<"Running on CPU"<<std::endl;
}

cv::Mat Inference::formatToSquare(const cv::Mat &source)
{
    int col = source.cols;
    int row = source.rows;
    int _max = MAX(col, row);
    cv::Mat result = cv::Mat::zeros(_max, _max, CV_8UC3);
    source.copyTo(result(cv::Rect(0, 0, col, row)));
    return result;
}

std::vector<Detection> Inference::runinference(const cv::Mat &input)
{
    
    cv::Mat modelInput = input;
    modelInput = formatToSquare(modelInput);
    cv::Mat blob;
    cv::dnn::blobFromImage(modelInput, blob, 1.0/255.0, modelShape, cv::Scalar(), true, false);
    net.setInput(blob);

    std::vector<cv::Mat> outputs;
    
    net.forward(outputs,net.getUnconnectedOutLayersNames());
    
    // yolov5 has an output of shape (batchSize, 25200, 85) (Num classes + box[x,y,w,h] + confidence[c])
    // yolov8 has an output of shape (batchSize, 84,  8400) (Num classes + box[x,y,w,h])
    // yolov8pose has an output of shape (batchSize, 56,8400) (box[x,y,w,h,conf] + keypoint[x,y,visible]*17)
    int rows = outputs[0].size[2];
    int dimensions = outputs[0].size[1];
    outputs[0] = outputs[0].reshape(1, dimensions);
    cv::transpose(outputs[0], outputs[0]);//tunrn to yolov5formate (1,8400,56)

    float *data = (float *)outputs[0].data;
    float x_factor = modelInput.cols / modelShape.width;
    float y_factor = modelInput.rows / modelShape.height;

    std::vector<cv::Rect> boxes;
    std::vector<float> confidences;
    std::vector<int> class_ids;
    std::vector<std::vector<cv::Point2f>> all_Key_Points;

    for( int i = 0; i<rows; ++i)
    {
        
        if(data[4] > modelScoreThreshold)
        {
            confidences.push_back(data[4]);
            class_ids.push_back(0);
            float x = data[0];
            float y = data[1];
            float w = data[2];
            float h = data[3];

            int left = int((x - 0.5 * w) * x_factor);
            int top = int((y - 0.5 * h) * y_factor);
            int width = int(w * x_factor);
            int height = int(h * y_factor);

            boxes.push_back(cv::Rect(left, top, width, height));

            std::vector<cv::Point2f> frame_keypoints;
            for(int j = 0; j < modelkeypoint_Num; ++j)
            {
                float kp_x = data[5 + j * 3] * x_factor;      // 直接计算偏移，易于调试
                float kp_y = data[6 + j * 3] * y_factor;
                float visibility = data[7 + j * 3]; // 可见性
                if(visibility > modelkeypointThreshold && kp_x >0 && kp_y>0 && kp_x < input.cols && kp_y < input.rows)
                {
                frame_keypoints.push_back(cv::Point2f(kp_x, kp_y));
                }
            }

            all_Key_Points.push_back(frame_keypoints);
        }

        data += dimensions;
    }


    std::vector<int> nms_result;
    cv::dnn::NMSBoxes(boxes, confidences, modelScoreThreshold, modelNMSThreshold, nms_result);

    std::vector<Detection> detections{};
    for (unsigned long i = 0; i < nms_result.size(); ++i)
    {
        int idx = nms_result[i];

        Detection result;
        result.class_id = class_ids[idx];
        result.confidence = confidences[idx];

        result.className = classes[result.class_id];
        result.box = boxes[idx];
        result.keypoint = all_Key_Points[idx];
        detections.push_back(result);
    }

    return detections;

}