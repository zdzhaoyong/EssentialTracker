#ifndef GPUSIFT_H
#define GPUSIFT_H
#include <opencv2/features2d/features2d.hpp>


class GPUSIFTImpl;
class GPUSIFT : public cv::Feature2D
{
public:
    GPUSIFT();
    virtual void operator()( cv::InputArray image, cv::InputArray mask,
                             CV_OUT std::vector<cv::KeyPoint>& keypoints,
                             cv::OutputArray descriptors,
                             bool useProvidedKeypoints=false ) const;


private:
    virtual int descriptorSize()const{return 512;}//4*128
    virtual int descriptorType()const{return 0;}  // FIXME: really don't know

    virtual void computeImpl(const cv::Mat& image,std::vector<cv::KeyPoint>& keypoints,cv::Mat& des)const;
    virtual void detectImpl(const cv::Mat &image, std::vector<cv::KeyPoint> &keypoints, const cv::Mat &mask) const;

    cv::Ptr<GPUSIFTImpl>  _impl;
};

#endif // GPUSIFT_H
