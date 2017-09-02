#include <GSLAM/core/GSLAM.h>
#include <opencv2/opencv.hpp>
#include "GMSMatcher.h"

class GMSTracker : public GSLAM::SLAM
{
public:
    virtual std::string type()const{return "GMSTracker";}
    virtual bool valid()const{return true;}
    virtual bool isDrawable()const{return false;}

    virtual bool    setCallback(GSLAM::GObjectHandle* cbk){handle=cbk;return true;}

    virtual bool    track(GSLAM::FramePtr& frame){
        curFrame=frame;
        curImage=frame->getImage();
        if(curImage.empty()) return false;
        cv::Mat img;
        if(curImage.cols>640)
        {
            cv::resize((cv::Mat)curImage,img,cv::Size(640,640*curImage.rows/curImage.cols));
        }
        else img=curImage;
        if(img.type()==CV_8UC3) cv::cvtColor(img,img,CV_RGB2GRAY);
        else if(img.type()==CV_8UC4) cv::cvtColor(img,img,CV_RGBA2GRAY);
        if(img.type()!=CV_8UC1) return false;
        curSmallImage=img;

        if(!lastSmallImage.empty())
        {
            match();
        }
        lastImage=curImage;
        lastSmallImage=curSmallImage;
        return true;
    }

    void matchORB()
    {
        using namespace cv;
        cv::Ptr<cv::Feature2D>     feature2d;
        Ptr<cv::DescriptorMatcher> matcher;
#if CV_VERSION_EPOCH==2
        feature2d=cv::Ptr<cv::Feature2D>(new ORB(svar.GetInt("ORB.nFeature",10000),
                                                 1.2,8,1,0,2,ORB::HARRIS_SCORE,15));
//        feature2d=cv::Ptr<cv::Feature2D>(new ORBextractor(svar.GetInt("ORB.nFeature",10000)));
#else
        Ptr<ORB> orb=ORB::create(svar.GetInt("ORB.nFeature",10000));
        orb->setFastThreshold(0);
        feature2d=orb;
#endif
        matcher=Ptr<cv::DescriptorMatcher>(new cv::BFMatcher(NORM_HAMMING));
        std::vector<cv::KeyPoint> kpts1,kpts2;
        cv::Mat                   des1,des2;
        cv::Mat                   img1=lastSmallImage,img2=curSmallImage;
    #if CV_VERSION_EPOCH==2
        (*feature2d)(img1,cv::Mat(),kpts1,des1);
        (*feature2d)(img2,cv::Mat(),kpts2,des2);
    #else
        feature2d->detectAndCompute(img1,cv::Mat(),kpts1,des1);
        feature2d->detectAndCompute(img2,cv::Mat(),kpts2,des2);
    #endif
        std::vector<cv::DMatch> matches;
        matcher->match(des1,des2,matches);

        std::vector<char> mask;
        std::vector<bool> maskTmp;
        gms_matcher gms(kpts1,img1.size(), kpts2,img2.size(), matches);
        int num_inliers = gms.GetInlierMask(maskTmp, false, false);
        mask.resize(maskTmp.size(),0);
        for(int i=0;i<maskTmp.size();i++)
            if(maskTmp[i]) mask[i]=1;
        cv::Mat outImg=img1.clone();
        cv::drawKeypoints(img1,kpts1,outImg);
        for(int i=0;i<matches.size();i++)
        if(mask[i])
        {
            cv::line(outImg,kpts1[matches[i].queryIdx].pt,kpts2[matches[i].trainIdx].pt,cv::Scalar(255,0,0),2);
        }

//        cv::drawMatches(img1,kpts1,img2,kpts2,matches,outImg,Scalar::all(-1),Scalar::all(-1),mask);
        cv::imwrite(to_string(curFrame->id())+".jpg",outImg);


    }

    GSLAM::GImage         lastImage,lastSmallImage,curImage,curSmallImage;
    GSLAM::GObjectHandle* handle;
    GSLAM::FramePtr       curFrame;
};

USE_GSLAM_PLUGIN(GMSTracker);
