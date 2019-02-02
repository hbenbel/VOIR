#include "tracker.hpp"


 bool confirm_box = false;
 bool draw_rectangle = false;
 cv::Rect rect;
 

void selectTarget(int event, int x, int y, int flag, void *param){
    cv::Mat *img = (cv::Mat*) param;
    switch(event){
        case cv::EVENT_MOUSEMOVE:
            if(draw_rectangle){
                rect.width = x - rect.x;
                rect.height = y - rect.y;
            }
            break;
        case cv::EVENT_LBUTTONDOWN:
            draw_rectangle = true;
            rect = cv::Rect(x, y, 0, 0);
            break;
        case cv::EVENT_LBUTTONUP:
            draw_rectangle = false;
            if(rect.width < 0){
                rect.x += rect.width;
                rect.width *= 1;
            }
            if(rect.height < 0){
                rect.y += rect.height;
                rect.height *= -1;
            }
            cv::rectangle(*img, rect, cv::Scalar(0, 255, 0), 2);
            confirm_box = true;
            break;
    }
}


int main(int argc, char **argv){
    if(argc != 2){
        std::cerr << "Usage: ./VOIR_tracker [video_path]" << std::endl;
        return 1;
    }

    cv::VideoCapture cap(argv[1]);
    cv::Mat frame;

    cap.read(frame);
    cv::namedWindow("Tracker");

    cv::Mat tmp = frame.clone();
    cv::setMouseCallback("Tracker", selectTarget, (void *) &tmp);
    while(!confirm_box){
        cv::Mat tmp2 = tmp.clone();
        if(draw_rectangle)
            cv::rectangle(tmp2, rect, cv::Scalar(0, 255, 0), 2);
        cv::imshow("Tracker", tmp2);
        if(cv::waitKey(30) == 'c')
            break;
    }

    //MeanShift tracker;
    //tracker.TargetModelInitialisation(frame, rect, 10, 16);
    
    while(cap.read(frame)){
        //cv::Rect new_rect = tracker.Track(frame, 1);
        cv::rectangle(frame, new_rect, cv::Scalar(0,255,0), 2);
        cv::imshow("Tracker", frame);
        if(cv::waitKey(30) == 'q')
            break;
    }
    cap.release();
    cv::destroyAllWindows();
    return 0;
}