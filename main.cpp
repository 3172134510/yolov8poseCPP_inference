#include "inference.h"



using namespace std;
using namespace cv;

int main()
{  
    Mat img = imread("/home/ljj/文档/yolov8/examples/Yolov8pose-CPP/people.jpg");
    Size input_size(640,640);
    int Keypoints_Num = 17;
    string modelpath = "/home/ljj/文档/yolov8/yolov8n-pose.onnx";
    Inference inf(modelpath,input_size,Keypoints_Num);
    

    //inference here
    TickMeter tm;
    tm.start(); 
    vector<Detection> output = inf.runinference(img);
    tm.stop();

    int detections = output.size();
    cout << "Number of detections:" << detections << std::endl;
    
    for(int i=0;i<detections;i++)
    {
        rectangle(img,output[i].box,Scalar(0,255,0),2);
        cout<<"keypoints number :"<<output[i].keypoint.size()<<endl;
        for(int j=0; j<output[i].keypoint.size(); j++)  // 改为 output[i]
        {
            
            circle(img,output[i].keypoint[j],3,Scalar(0,0,255),-1);
        }
        
    }
    
    cout<<"Time: "<<tm.getTimeMilli()<<"ms"<<endl;

  

    imshow("result",img);
    waitKey(0);
}