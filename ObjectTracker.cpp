#include <iostream>
#include <time.h>
#include "opencv2/opencv.hpp"

using namespace cv;
using namespace std;

int main( int argc, char** argv ) {
  //cout << getNumThreads() << endl;
  
  //capture the video from webcam
  VideoCapture cap("http://192.168.43.1:8080/video");
  
  int ret;
  ret = cap.set(3, 320);
  ret = cap.set(4, 240);
  
  // if not success, exit program
  if ( !cap.isOpened() )
  {
       cout << "Cannot open the web cam" << endl;
       return -1;
  }
  
  //create a window called "Control"
  namedWindow("Control", WINDOW_NORMAL);
  
  int iLowH = 170;
  int iHighH = 179;
  int iLowS = 150; 
  int iHighS = 255;
  int iLowV = 60;
  int iHighV = 255;
  
  //Create trackbars in "Control" window
  createTrackbar("LowH", "Control", &iLowH, 179); //Hue (0 - 179)
  createTrackbar("HighH", "Control", &iHighH, 179);
  
  createTrackbar("LowS", "Control", &iLowS, 255); //Saturation (0 - 255)
  createTrackbar("HighS", "Control", &iHighS, 255);
  
  createTrackbar("LowV", "Control", &iLowV, 255);//Value (0 - 255)
  createTrackbar("HighV", "Control", &iHighV, 255);
  
  int iLastX = -1; 
  int iLastY = -1;
  
  // Capture a temporary image from the camera
  Mat imgTmp;
  cap.read(imgTmp);
   
  // Create a black image with the size as the camera output
  Mat imgLines = Mat::zeros( imgTmp.size(), CV_8UC3 );
  time_t start,end;
  time (&start);
  int frames = 0;
  
    while (true) {
        Mat imgOriginal;
        // read a new frame
        bool bSuccess = cap.read(imgOriginal);
        
        //if not success, break loop
        if (!bSuccess) {      
             cout << "Cannot read a frame from video stream" << endl;
             break;
        }
        
  //Convert the captured frame from BGR to HSV
  Mat imgHSV;
  cvtColor(imgOriginal, imgHSV, COLOR_BGR2HSV);
  
  //Threshold the image
  Mat imgThresholded;
  inRange(imgHSV, Scalar(iLowH, iLowS, iLowV), Scalar(iHighH, iHighS, iHighV), imgThresholded);
  
  //morphological opening (removes small objects from the foreground)
  erode(imgThresholded, imgThresholded, getStructuringElement(MORPH_ELLIPSE, Size(5, 5)) );
  dilate( imgThresholded, imgThresholded, getStructuringElement(MORPH_ELLIPSE, Size(5, 5)) ); 
  
  //morphological closing (removes small holes from the foreground)
  dilate( imgThresholded, imgThresholded, getStructuringElement(MORPH_ELLIPSE, Size(5, 5)) ); 
  erode(imgThresholded, imgThresholded, getStructuringElement(MORPH_ELLIPSE, Size(5, 5)) );
  
  //Calculate the moments of the thresholded image
  Moments oMoments = moments(imgThresholded);
  double dM01 = oMoments.m01;
  double dM10 = oMoments.m10;
  double dArea = oMoments.m00;
  
  // if the area <= 10000, I consider that the there are no object in the image and it's because of the noise, the area is not zero 
  if (dArea > 10000)
  {
   //calculate the position of the object
   int posX = dM10 / dArea;
   int posY = dM01 / dArea;        
   if (iLastX >= 0 && iLastY >= 0 && posX >= 0 && posY >= 0)
   {
    //Draw a red line from the previous point to the current point
    //line(imgLines, Point(posX, posY), Point(iLastX, iLastY), Scalar(0,0,255), 2);
    rectangle(imgLines, Point(posX-50, posY-120), Point(iLastX+50, iLastY+100), Scalar(0,0,255), 3);
   }
   iLastX = posX;
   iLastY = posY;
  }
  
    //show the thresholded image
    imshow("Thresholded Image", imgThresholded);
    imgOriginal = imgOriginal + imgLines;
    //show the original image
    imshow("Original", imgOriginal);
    
    //wait for 'esc' key press for 30ms. If 'esc' key is pressed, break loop
    if (waitKey(30) == 27)
    {
          cout << "esc key is pressed by user" << endl;
          break; 
    }
    frames++;
  }
  
  time (&end);
  double dif = difftime (end,start);
  printf("FPS %.2lf seconds.\r\n", (frames / dif));
  return 0;
}
