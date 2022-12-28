#include <opencv2/opencv.hpp>
#include <iostream>

using namespace cv;
using namespace std;

int main(int argc, char* argv[])
{
 //Open the default video camera
 VideoCapture cap(0);

 // if not success, exit program
 if (cap.isOpened() == false)  
 {
  cout << "Cannot open the video camera" << endl;
  cin.get(); //wait for any key press
  return -1;
 } 

 double dWidth = cap.get(CAP_PROP_FRAME_WIDTH); //get the width of frames of the video
 double dHeight = cap.get(CAP_PROP_FRAME_HEIGHT); //get the height of frames of the video

 Size frame_size(dWidth, dHeight);
 int frames_per_second = 10;     
 //Create and initialize the VideoWriter object 
 VideoWriter oVideoWriter("MyVideo.avi", VideoWriter::fourcc('M', 'J', 'P', 'G'), 
                                                           frames_per_second, frame_size, true); 

 //If the VideoWriter object is not initialized successfully, exit the program
 if (oVideoWriter.isOpened() == false) 
 {
     cout << "Cannot save the video to a file" << endl;
     cin.get(); //wait for any key press
     return -1;
 }

 string window_name = "My Camera Feed";
 namedWindow(window_name); //create a window called "My Camera Feed"
 while (true)
 {
  Mat frame;
  bool bSuccess = cap.read(frame); // read a new frame from video 

  //Breaking the while loop if the frames cannot be captured
  if (bSuccess == false) 
  {
   cout << "Video camera is disconnected" << endl;
   cin.get(); //Wait for any key press
   break;
  }

  //write the video frame to the file
  oVideoWriter.write(frame); 


  //show the frame in the created window
  imshow(window_name, frame);

  //wait for for 10 ms until any key is pressed.  
  //If the 'Esc' key is pressed, break the while loop.
  //If the any other key is pressed, continue the loop 
  //If any key is not pressed withing 10 ms, continue the loop 
  if (waitKey(10) == 27)
  {
   cout << "Esc key is pressed by user. Stoppig the video" << endl;
   break;
  }
 }

 return 0;

}