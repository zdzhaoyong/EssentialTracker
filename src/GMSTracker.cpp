#include <GSLAM/core/GSLAM.h>
#include <GSLAM/core/Timer.h>
#include <GSLAM/core/HashMap.h>
#include <GSLAM/core/Optimizer.h>

#include <opencv2/opencv.hpp>
#include <opencv2/nonfree/nonfree.hpp>

#include "GMSMatcher.h"
#include "GPUSIFT.h"
#if __linux
    #ifndef INCLUDED_GL_H
    #define INCLUDED_GL_H
    #include <GL/glew.h>
    //#include <GL/gl.h>
    #include <GL/glut.h>
    #endif
#else
#include <GL/glew.h>
    //#include <GL/gl.h>
    #include "gui/GL_headers/glext.h"
#endif

inline void glVertex(const pi::Point3d& pt)
{
    glVertex3d(pt.x,pt.y,pt.z);
}

inline void glVertex(const pi::Point3f& pt)
{
    glVertex3f(pt.x,pt.y,pt.z);
}

inline void glColor(const pi::Point3ub& color)
{
    glColor3ub(color.x,color.y,color.z);
}

class ScopedTimer
{
public:
    ScopedTimer(const char* func_name):_func_name(func_name){timer.enter(func_name);}
    ~ScopedTimer(){timer.leave(_func_name);}
    const char* _func_name;
};

class GMSTracker : public GSLAM::SLAM
{
public:
    GMSTracker(){
        _curMap=GSLAM::MapPtr(new GSLAM::HashMap());
        optimizer=GSLAM::Optimizer::create();
    }
    virtual std::string type()const{return "GMSTracker";}
    virtual bool valid()const{return true;}
    virtual bool isDrawable()const{return true;}

    virtual bool    setCallback(GSLAM::GObjectHandle* cbk){handle=cbk;return true;}

    virtual bool    track(GSLAM::FramePtr& frame){
        curFrame=frame;
        curImage=frame->getImage();
        if(curImage.empty()) return false;
        cv::Mat img;
        int maxWidth=640;
        if(svar.GetString("Feature2D","SIFT")=="SIFT")
            maxWidth=19200;
        if(curImage.cols>maxWidth)
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
            matchORB();
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
        if(svar.GetString("Feature2D","SIFT")=="SIFT")
        {
            svar.GetInt("GPU_SIFT.nOctaveLayers", 3)=8;
//            svar.GetDouble("GPU_SIFT.contrastThreshold", 0.04)=0.01;
//            svar.GetDouble("GPU_SIFT.edgeThreshold", 5)=2;
            feature2d=cv::Ptr<cv::Feature2D>(new GPUSIFT());
            matcher=Ptr<cv::DescriptorMatcher>(new cv::FlannBasedMatcher());;
        }
        else
        {
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
        }

        std::vector<cv::KeyPoint> kpts1,kpts2;
        cv::Mat                   des1,des2;
        cv::Mat                   img1=lastSmallImage,img2=curSmallImage;

        {
            ScopedTimer detectionTimer("Detection");
#if CV_VERSION_EPOCH==2
            (*feature2d)(img1,cv::Mat(),kpts1,des1);
            (*feature2d)(img2,cv::Mat(),kpts2,des2);
#else
            feature2d->detectAndCompute(img1,cv::Mat(),kpts1,des1);
            feature2d->detectAndCompute(img2,cv::Mat(),kpts2,des2);
#endif
        }
        std::vector<cv::DMatch> matches;
        std::vector<cv::DMatch> matches21;
        {
            ScopedTimer matchTimer("Matching");
            matcher->match(des1,des2,matches);
            matcher->match(des2,des1,matches21);
        }
        //block filter
        {
            ScopedTimer blockFilter("BlockFilter");
            std::vector<cv::DMatch> matchesTMP;
            for(auto m:matches)
            {
                assert(m.queryIdx<des1.rows&&m.trainIdx<des2.rows);
                if(matches21[m.trainIdx].trainIdx==m.queryIdx)
                    matchesTMP.push_back(m);
            }
            matches=matchesTMP;
        }

        std::vector<char> mask;
        if(0)
        {
            ScopedTimer GMSFilter("GMSFilter");
            std::vector<bool> maskTmp;
            ScopedTimer gmsTimer("GMSFilter");
            gms_matcher gms(kpts1,img1.size(), kpts2,img2.size(), matches);
            int num_inliers = gms.GetInlierMask(maskTmp, false, false);
            mask.resize(maskTmp.size(),0);
            for(int i=0;i<maskTmp.size();i++)
                if(maskTmp[i]) mask[i]=1;
        }
        else
        {
            ScopedTimer FundamentalFilter("FundamentalFilter");
            std::vector<uchar> maskTmp;
            vector<cv::Point2f> kps1,kps2;
            kps1.reserve(des1.rows);
            kps2.reserve(des2.rows);
            for(auto m:matches)
            {
                assert(m.queryIdx<des1.rows&&m.trainIdx<des2.rows);
                kps1.push_back(kpts1[m.queryIdx].pt);
                kps2.push_back(kpts2[m.trainIdx].pt);
            }
            if(cv::findFundamentalMat(kps1,kps2,maskTmp).empty()) return;
//            cv::findHomography(kps1,kps2,maskTmp,cv::RANSAC,img1.cols/20);
            mask.resize(maskTmp.size(),0);
            for(int i=0;i<maskTmp.size();i++)
                if(maskTmp[i]) mask[i]=1;
        }

        cv::Mat outImg=img1.clone();
        cv::drawKeypoints(img1,kpts1,outImg);
        int count=0;
        for(int i=0;i<matches.size();i++)
            if(mask[i])
            {
                count++;
                cv::line(outImg,kpts1[matches[i].queryIdx].pt,kpts2[matches[i].trainIdx].pt,cv::Scalar(255,0,0),2);
            }
        cout<<curFrame->id()<<":"<<des1.rows<<"-"<<des2.rows<<"-"<<count<<endl;

        if(svar.GetInt("Verbose"))
        {
            cv::imwrite(to_string(curFrame->id())+".jpg",outImg);
            cv::drawMatches(img1,kpts1,img2,kpts2,matches,outImg,Scalar::all(-1),Scalar::all(-1),mask);
            cv::imwrite(to_string(curFrame->id())+"_match.jpg",outImg);
        }

        GSLAM::SE3    relativePose;
        {
            GSLAM::Camera camera=curFrame->getCamera();
            std::vector<std::pair<GSLAM::CameraAnchor,GSLAM::CameraAnchor> > anchors;
            std::vector<GSLAM::IdepthEstimation> firstIDepth;

            for(int i=0;i<matches.size();i++)
            {
                if(!mask[i]) continue;
                cv::Point2f pt1=kpts1[matches[i].queryIdx].pt;
                cv::Point2f pt2=kpts2[matches[i].trainIdx].pt;
                anchors.push_back(make_pair(camera.UnProject(pt1.x,pt1.y),
                                  camera.UnProject(pt2.x,pt2.y)));
            }

            firstIDepth.resize(anchors.size(),GSLAM::Point2d(1,-1));

            if(!optimizer->optimizePose(anchors,firstIDepth,relativePose))
            {
                LOG(ERROR)<<"Failed to optimize pose.";
                return;
            }
        }


        if(lastFrame)
        {
            curFrame->setPose(lastFrame->getPose()*relativePose);

            LOG(INFO)<<"CurPose:"<<curFrame->getPose();
            _curMap->insertMapFrame(curFrame);
        }
        lastFrame=curFrame;

    }

    void draw()
    {
        GSLAM::FrameArray frames;
        if(!_curMap->getFrames(frames)) return;

        for(GSLAM::FramePtr frame:frames)
        {
            GSLAM::SIM3 sim3=frame->getPoseScale();
            GSLAM::SE3  pose =sim3.get_se3();
            double      depth=sim3.get_scale()*0.1;
            GSLAM::Point3Type t=pose.get_translation();
            double r[9];
            pose.get_rotation().getMatrixUnsafe(r);
            //            glMultMatrix(pose);
            GSLAM::Camera camera=frame->getCamera();

            // Draw camera rect
            {
                pi::Point3d tl=camera.UnProject(pi::Point2d(0,0));
                pi::Point3d tr=camera.UnProject(pi::Point2d(camera.width(),0));
                pi::Point3d bl=camera.UnProject(pi::Point2d(0,camera.height()));
                pi::Point3d br=camera.UnProject(pi::Point2d(camera.width(),camera.height()));
                //pi::Point2d ct=cam_out->UnProject(pi::Point2d(cam_out->Cx(),cam_out->Cy()));

                GSLAM::Point3Type  W_tl=pose*(pi::Point3d(tl.x,tl.y,1)*depth);
                GSLAM::Point3Type  W_tr=pose*(pi::Point3d(tr.x,tr.y,1)*depth);
                GSLAM::Point3Type  W_bl=pose*(pi::Point3d(bl.x,bl.y,1)*depth);
                GSLAM::Point3Type  W_br=pose*(pi::Point3d(br.x,br.y,1)*depth);

                //        Point3Type  W_ct=pose*(pi::Point3d(ct.x,ct.y,1)*depth);
                glBegin(GL_LINES);
                glLineWidth(2.5);
                glColor3f(0, 0, 1);
                glVertex(t);        glVertex(W_tl);
                glVertex(t);        glVertex(W_tr);
                glVertex(t);        glVertex(W_bl);
                glVertex(t);        glVertex(W_br);
                glVertex(W_tl);     glVertex(W_tr);
                glVertex(W_tr);     glVertex(W_br);
                glVertex(W_br);     glVertex(W_bl);
                glVertex(W_bl);     glVertex(W_tl);
                glEnd();
            }
        }
    }

    SPtr<GSLAM::Optimizer> optimizer;
    GSLAM::GImage         lastImage,lastSmallImage,curImage,curSmallImage;
    GSLAM::GObjectHandle* handle;
    GSLAM::FramePtr       curFrame,lastFrame;
};

USE_GSLAM_PLUGIN(GMSTracker);
