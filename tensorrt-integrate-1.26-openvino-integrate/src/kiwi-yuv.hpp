#ifndef KIWI_YUV_HPP
#define KIWI_YUV_HPP

#include <opencv2/opencv.hpp>

namespace kiwi{
    
    class YUVImage{
    public:
        YUVImage(bool usebgr=true, int std_width=320, int std_height=256){
            usebgr_     = usebgr;

            assert(std_width % 2 == 0 && std_height % 2 == 0);
            yuv_merge_.create(std_height * 1.5, std_width, CV_8U);
            image_.create(std_height, std_width, CV_8U);
            y_ = cv::Mat(std_height, std_width, CV_8U, yuv_merge_.data);
            u_ = cv::Mat(std_height * 0.5, std_width * 0.5, CV_8U, yuv_merge_.ptr<uchar>(std_height));
            v_ = cv::Mat(std_height * 0.5, std_width * 0.5, CV_8U, yuv_merge_.ptr<uchar>(std_height * 1.25));
        }

        bool update(const char* file){
            auto im = cv::imread(file);
            if(im.empty()) return false;

            cv::resize(im, image_, image_.size());
            has_update_ = true;
            return true;
        }

        void update(const unsigned char* y, const unsigned char* u, const unsigned char* v, int width, int height){

            has_update_ = true;
            cv::Mat my(height, width, CV_8U, (unsigned char*)y);
            cv::Mat mu(height * 0.5, width * 0.5, CV_8U, (unsigned char*)u);
            cv::Mat mv(height * 0.5, width * 0.5, CV_8U, (unsigned char*)v);

            cv::resize(my, y_, y_.size());
            cv::resize(mu, u_, u_.size());
            cv::resize(mv, v_, v_.size());

            if(usebgr_){
                cv::cvtColor(yuv_merge_, image_, cv::COLOR_YUV2BGR_I420);
            }else{
                cv::cvtColor(yuv_merge_, image_, cv::COLOR_YUV2RGB_I420);
            }
        }

        cv::Mat get_image(){
            return image_;
        }

        bool empty(){
            return !has_update_;
        }

    private:
        bool usebgr_;
        bool has_update_ = false;
        cv::Mat image_;
        cv::Mat y_, u_, v_;
        cv::Mat yuv_merge_;
    };
};

#endif // KIWI_YUV_HPP