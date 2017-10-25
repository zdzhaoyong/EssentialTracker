#include "GPUSIFT.h"

#include "SiftGPU/SiftGPU.h"
#include <GSLAM/core/Svar.h>
#include <GSLAM/core/SharedLibrary.h>
#include <pil/gui/gl/OpenGL.h>

#include <opencv2/imgproc/imgproc.hpp>

using namespace std;
using namespace GSLAM;

class GPUSIFTImpl
{
  public:
    GPUSIFTImpl()
    {
        sift = NULL;
        siftMatcher = NULL;
        init();
    }

    ~GPUSIFTImpl()
    {
        if( sift )
        {
            delete sift;
            sift = NULL;
        }

        if( siftMatcher )
        {
            delete siftMatcher;
            siftMatcher = NULL;
        }
    }

    int init(Svar* sv=NULL)
    {
        // parameters
        int nFeature = 0;
        int nOctaveLayers = 3;
        double contrastThreshold = 0.04;
        double edgeThreshold = 10;
        double sigma = 1.6;
        int verbose = 0;

        string libPath = "./libs/libsiftgpu.so";

        if(sv==NULL) sv=&svar;
//        if( sv != NULL )
        {
            nFeature = sv->GetInt("GPU_SIFT.nFeature", 2000);
            nOctaveLayers = sv->GetInt("GPU_SIFT.nOctaveLayers", 3);
            contrastThreshold = sv->GetDouble("GPU_SIFT.contrastThreshold", 0.04);
            edgeThreshold = sv->GetDouble("GPU_SIFT.edgeThreshold", 10);
            sigma = sv->GetDouble("GPU_SIFT.sigma", 1.6);
            verbose = sv->GetInt("GPU_SIFT.verbose", 0);

            libPath = sv->GetString("GPU_SIFT.lib", std::string(libPath));
        }


        sift = CreateNewSiftGPU(0);

        //Create a context for computation, and SiftGPU will be initialized automatically
        //The same context can be used by SiftMatchGPU
        if(sift->CreateContextGL() != SiftGPU::SIFTGPU_FULL_SUPPORTED) return 0;

        // prepare arguments
        vector<string> args;

        args.push_back("-tc");
        args.push_back(to_string(nFeature));
        args.push_back("-v");
        args.push_back(to_string(verbose));
        args.push_back("-e");
        args.push_back(to_string(edgeThreshold));
        args.push_back("-d");
        args.push_back(to_string(nOctaveLayers));
        args.push_back("-t");
        args.push_back(to_string(contrastThreshold));
//        args.push_back("-maxd");
//        args.push_back(to_string(svar.GetInt("GPU_SIFT.MaxImageSize",4000)));

//        for(int i=0; i<args.size(); i++)
//            printf("siftgpu_args[%2d] = %s\n", i, args[i].c_str());

        int argc = args.size();
        char** argv = (char**) malloc(sizeof(char*)*argc);
        for(int i=0; i<argc; i++)
            argv[i] = (char*) args[i].c_str();

        sift->ParseParam(argc, argv);

        free(argv);

        desc_size = 128;

        // create siftMatcher
        maxSiftNum = 8096;
        {
            siftMatcher = CreateNewSiftMatchGPU(maxSiftNum);
            siftMatcher->VerifyContextGL();
        }

        return 0;
    }


    virtual int detect(const cv::Mat& image, std::vector<cv::KeyPoint>& keypoints,
                       const cv::Mat& mask=cv::Mat()) const
    {
        if( !sift ) return -1;

        cv::Mat imgGray;
        if( image.type() == CV_8UC3 )
            cv::cvtColor(image, imgGray, cv::COLOR_BGR2GRAY);
        else if(image.type()==CV_8UC1)
            imgGray = image;
        else if(image.type()==CV_8UC4)
            cv::cvtColor(image,imgGray,CV_RGBA2GRAY);
        if(imgGray.empty()) return -1;

        sift->RunSIFT(imgGray.cols, imgGray.rows, imgGray.data,
                      GL_LUMINANCE, GL_UNSIGNED_BYTE);

        int nFea = sift->GetFeatureNum();

        vector<SiftGPU::SiftKeypoint>   kps;

        kps.resize(nFea);
        keypoints.resize(nFea);

        sift->GetFeatureVector(&kps[0], NULL);

        for(int i=0; i<nFea; i++)
        {
            keypoints[i].pt.x   = kps[i].x;
            keypoints[i].pt.y   = kps[i].y;
            keypoints[i].size   = kps[i].s;
            keypoints[i].angle  = kps[i].o*180.0/M_PI;
        }

        return 0;
    }

    virtual int compute(const cv::Mat& image, std::vector<cv::KeyPoint>& keypoints,
                        cv::Mat& descriptors) const
    {
        if( !sift ) return -1;

        cv::Mat imgGray;
        if( image.channels() == 3 )
            cv::cvtColor(image, imgGray, cv::COLOR_BGR2GRAY);
        else
            imgGray = image.clone();

        sift->RunSIFT(imgGray.cols, imgGray.rows, imgGray.data,
                      GL_LUMINANCE, GL_UNSIGNED_BYTE);

        int nFea = sift->GetFeatureNum();

        vector<SiftGPU::SiftKeypoint> kps(nFea);
        keypoints.resize(nFea);
        descriptors.create(nFea, desc_size, CV_32F);

        sift->GetFeatureVector(&kps[0], descriptors.ptr<float>(0));

        for(int i=0; i<nFea; i++)
        {
            keypoints[i].pt.x   = kps[i].x;
            keypoints[i].pt.y   = kps[i].y;
            keypoints[i].size   = kps[i].s;
            keypoints[i].angle  = kps[i].o*180.0/M_PI;
        }

        return 0;
    }

    virtual int detectAndCompute(cv::InputArray imageIn, cv::InputArray mask,
                                 CV_OUT std::vector<cv::KeyPoint>& keypoints,
                                 cv::OutputArray descriptorsDest,
                                 bool useProvidedKeypoints=false) const
    {
        if( !sift ) return -1;
        cv::Mat image=imageIn.getMat();

        cv::Mat imgGray;
        if( image.type() == CV_8UC3 )
            cv::cvtColor(image, imgGray, cv::COLOR_BGR2GRAY);
        else if(image.type()==CV_8UC1)
            imgGray = image;
        else if(image.type()==CV_8UC4)
            cv::cvtColor(image,imgGray,CV_RGBA2GRAY);
        if(imgGray.empty()) return -1;

        sift->RunSIFT(image.cols, imgGray.rows, imgGray.data,
                      GL_LUMINANCE, GL_UNSIGNED_BYTE);

        int nFea = sift->GetFeatureNum();

        vector<SiftGPU::SiftKeypoint> kps(nFea);
        keypoints.resize(nFea);
        descriptorsDest.create(nFea,desc_size,CV_32F);
        cv::Mat descriptors=descriptorsDest.getMat();

        sift->GetFeatureVector(&kps[0], descriptors.ptr<float>(0));

        for(int i=0; i<nFea; i++)
        {
            keypoints[i].pt.x   = kps[i].x;
            keypoints[i].pt.y   = kps[i].y;
            keypoints[i].size   = kps[i].s;
            keypoints[i].angle  = kps[i].o*180.0/M_PI;
        }

        return 0;
    }

    int                                 desc_size;
    cv::L2<float>                       fd_dis;
    int                                 maxSiftNum;

    SharedLibrary                   sl;
    SiftGPU                             *sift;
    SiftMatchGPU                        *siftMatcher;
};

GPUSIFT::GPUSIFT()
    :_impl(new GPUSIFTImpl())
{
}

void GPUSIFT::operator()( cv::InputArray image, cv::InputArray mask,
                 CV_OUT std::vector<cv::KeyPoint>& keypoints,
                 cv::OutputArray descriptors,
                 bool useProvidedKeypoints) const
{
    _impl->detectAndCompute(image,mask,keypoints,descriptors,useProvidedKeypoints);
}

void GPUSIFT::computeImpl(const cv::Mat& image,std::vector<cv::KeyPoint>& keypoints,cv::Mat& des)const
{
    _impl->compute(image,keypoints,des);
}

void GPUSIFT::detectImpl(const cv::Mat &image, vector<cv::KeyPoint> &keypoints, const cv::Mat &mask) const
{
    _impl->detect(image,keypoints,mask);
}


