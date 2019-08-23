/*-----------------------------------------------------------
--------------------------------------------------------------
if have question ,please contact 韩世豪  1336295654@qq.com---
-------------------------------------------------------------
-----------------------------------------------------------*/

#include <opencv2/highgui/highgui.hpp>
#include <opencv2/opencv.hpp>
#include <iostream>
#include <sstream>
#include <fstream>
#include <vector>
#include <glob.h>
#include <unistd.h>
#include <dirent.h>
#include <stdlib.h>
#include <string>
#include <stdio.h>
#include <map>
#include <Eigen/Dense>
#include "msckf_vio/image_processor.h"
#include "msckf_vio/msckf_vio.h"
#include <thread>
#include <mutex>
#include <condition_variable>

using namespace std;

queue<pair<double,vector<Eigen::Vector3d>> > IMU;
queue<pair<double,vector<cv::Mat>> > IMAGE;
std::mutex m_buf;
double current_time = -1;
std::condition_variable con;

msckf_vio::ImageProcessor vio_msckf_frontend_;
msckf_vio::MsckfVio vio_msckf_backend_;


//get num in file name in sort from small to big
vector<double> getFiles(char* dirc){
    vector<string> files;
    struct dirent *ptr;
    DIR *dir;
    dir = opendir(dirc);

    if(dir == NULL)
    {
        perror("open dir error ...");
        exit(1);
    }

    while((ptr = readdir(dir)) != NULL){
        if(strcmp(ptr->d_name,".")==0 || strcmp(ptr->d_name,"..")==0)    ///current dir OR parrent dir
            continue;
        if(ptr->d_type == 8)//it;s file
        {
            files.push_back(ptr->d_name);
        }

        else if(ptr->d_type == 10)//link file
            continue;
        else if(ptr->d_type == 4) //dir
        {
            files.push_back(ptr->d_name);
        }
    }
    closedir(dir);

    vector<double> result;
    for(int i=0;i < files.size();i++)
    {
        result.push_back(atof(files[i].c_str()));
    }
    sort(result.begin(),result.end());

   // for(size_t i = 0; i < result.size();++i){
        //std::cout.setf(std::ios::fixed, std::ios::floatfield);
        //std::cout.precision(20);
        //cout << result[i] << endl;
    //}
    return result;
}



Eigen::Vector3d PreprocessAccelerator(const Eigen::Vector3d &inacc)
{
    static Eigen::Vector3d last_output = Eigen::Vector3d::Zero();
    static int num[3] = {0,0,0};
    static double k[3] = {0.03,0.03,0.03};
    static bool old_flag[3] = {false,false,false};
    bool new_flag[3];
    Eigen::Vector3d new_output;
    if(last_output.norm() < 1)
    {
        last_output = inacc;
        return last_output;
    }
    for(int i=0;i<2;i++)
    {
        new_flag[i] = inacc[i] > last_output[i];
        if(new_flag[i] == old_flag[i])
        {
            if(fabs(inacc[i] - last_output[i]) > 2)
                num[i]++;
            if(num[i] > 5)
                k[i] += 0.1;
        }
        else
        {
            num[i] = 0;
            k[i] = 0.03;
        }
        if(k[i] > 0.5) k[i] = 0.5;
        new_output[i] = k[i] * inacc[i] + (1 - k[i]) * last_output[i];
        old_flag[i] = new_flag[i];
    }
    for(int i=2;i<3;i++)
    {
        new_flag[i] = inacc[i] > last_output[i];
        if(new_flag[i] == old_flag[i])
        {
            if(fabs(inacc[i] - last_output[i]) > 2)
                num[i]++;
            if(num[i] > 5)
                k[i] += 0.01;
        }
        else
        {
            num[i] = 0;
            k[i] = 0.01;
        }
        if(k[i] > 0.1) k[i] = 0.1;
        new_output[i] = k[i] * inacc[i] + (1 - k[i]) * last_output[i];
        old_flag[i] = new_flag[i];
    }

    // new_output = inacc;
    return new_output;
}
void InputIMU( const double timestamp, const Eigen::Vector3d& accl, const Eigen::Vector3d& gyro)
{
    m_buf.lock();
    vector<Eigen::Vector3d> IMUTEMP;
    IMUTEMP.push_back(accl);
    IMUTEMP.push_back(gyro);
    IMU.push(make_pair(timestamp,IMUTEMP));
    m_buf.unlock();
    con.notify_one();
}

void InputIMAGE(const cv::Mat& cam0_img,
                const cv::Mat& cam1_img,
                double time)
{
    m_buf.lock();
    vector<cv::Mat> IMAGETEMP;
    IMAGETEMP.push_back(cam0_img);
    IMAGETEMP.push_back(cam1_img);
    IMAGE.push(make_pair(time,IMAGETEMP));
    m_buf.unlock();
    con.notify_one();

}


std::vector<std::pair<vector<pair<double,vector<Eigen::Vector3d>> >, vector<pair<double,vector<cv::Mat>> > >>
getMeasurements()
{
    std::vector<std::pair<vector<pair<double,vector<Eigen::Vector3d>> >, vector<pair<double,vector<cv::Mat>> > >>  measurements;
    while (true)
    {

        if(IMAGE.empty()||IMU.empty())
        {
          //cout<<"wait for data"<<endl;
          return measurements;
        }

        if (!(IMU.back().first > IMAGE.front().first))
        {
            cout<<"wait for imu, only should happen at the beginning";
            return measurements;
        }
        if (!(IMU.front().first < IMAGE.front().first))
        {
            cout<<"throw img, only should happen at the beginning";
            IMAGE.pop();
            continue;
        }
        vector<pair<double,vector<Eigen::Vector3d>> > IMUs;
        while (IMU.front().first < IMAGE.front().first)
        {
            IMUs.emplace_back(IMU.front());
            IMU.pop();
        }
        IMUs.emplace_back(IMU.front());

        if (IMUs.empty())
           cout<<"no imu between two image";

        vector<pair<double,vector<cv::Mat>> > IMAGES;
        IMAGES.push_back(IMAGE.front());
        IMAGE.pop();
        measurements.push_back(make_pair(IMUs,IMAGES));

    }
    cout<<measurements.size()<<endl;
    return measurements;
}

void process()
{

    while (true)
       {

        std::vector<std::pair<vector<pair<double,vector<Eigen::Vector3d>> >, vector<pair<double,vector<cv::Mat>> > >> measurements;
        std::unique_lock<std::mutex> lk(m_buf);

        con.wait(lk, [&]
                 {
            return (measurements = getMeasurements()).size() != 0;
                 });
        lk.unlock();

        for (auto &measurement : measurements)
        {
           auto img = measurement.second.front();
           double dx = 0, dy = 0, dz = 0, rx = 0, ry = 0, rz = 0;
           for (auto &imu_msg : measurement.first)
           {
               double t = imu_msg.first;
               double img_t = img.first;
               if (t <= img_t)
               {
                   if (current_time < 0)
                       current_time = t;
                   double dt = t - current_time;
                   current_time = t;
                   dx = imu_msg.second.front()[0];
                   dy = imu_msg.second.front()[1];
                   dz = imu_msg.second.front()[2];
                   rx = imu_msg.second.back()[0];
                   ry = imu_msg.second.back()[1];
                   rz = imu_msg.second.back()[2];
                   vio_msckf_frontend_.imuCallback(current_time,Eigen::Vector3d(dx, dy, dz), Eigen::Vector3d(rx, ry, rz));
                   vio_msckf_backend_.imuCallback(current_time,Eigen::Vector3d(dx, dy, dz), Eigen::Vector3d(rx, ry, rz));
                  //std::cout.setf(std::ios::fixed, std::ios::floatfield);
                  //std::cout.precision(6);
                  //cout<<"imu:"<<current_time<<endl;

               }
               else
               {
                   double dt_1 = img_t - current_time;
                   double dt_2 = t - img_t;
                   current_time = img_t;
                   double w1 = dt_2 / (dt_1 + dt_2);
                   double w2 = dt_1 / (dt_1 + dt_2);
                   dx = w1 * dx + w2 * imu_msg.second.front()[0];
                   dy = w1 * dy + w2 * imu_msg.second.front()[1];
                   dz = w1 * dz + w2 * imu_msg.second.front()[2];
                   rx = w1 * rx + w2 * imu_msg.second.back()[0];
                   ry = w1 * ry + w2 * imu_msg.second.back()[1];
                   rz = w1 * rz + w2 * imu_msg.second.back()[2];
                   vio_msckf_frontend_.imuCallback(current_time,Eigen::Vector3d(dx, dy, dz), Eigen::Vector3d(rx, ry, rz));
                   vio_msckf_backend_.imuCallback(current_time,Eigen::Vector3d(dx, dy, dz), Eigen::Vector3d(rx, ry, rz));
                   //std::cout.setf(std::ios::fixed, std::ios::floatfield);
                   //std::cout.precision(6);
                   //cout<<"imu:"<<current_time<<endl;
               }
            }
           //cv::imshow("1",img.second.front());
          // cv::waitKey(1);
           //std::cout.precision(6);
           //cout<<"imagetime"<<img.first<<endl;
          vector<pair<double, std::vector<Eigen::Matrix<double, 5, 1>>> >  tempmsckf_feature;
          tempmsckf_feature=vio_msckf_frontend_.stereoCallback(img.second.front(),img.second.back(),img.first);

          vio_msckf_backend_.featureCallback(tempmsckf_feature);

        }
       }
}

int main(int argc, char** argv)
{

        std::thread measurement_process{process};

        char* left_image_path = "/media/teemos/Work/ubuntu/download/MH02/mav0/cam0/data";
        char* right_image_path = "/media/teemos/Work/ubuntu/download/MH02/mav0/cam1/data";
        char* imu_path = "/media/teemos/Work/ubuntu/download/MH02/mav0/imu0/data.txt";

        vector<double> left_image_index = getFiles(left_image_path);// ..
        vector<double> right_image_index = getFiles(right_image_path);
        FILE *fp;
        fp = fopen(imu_path,"r");

        double imu_time,last_imu_time;
        double acceleration[3],angular_v[3];

        double time_count_left,time_count_right;
        int imu_seq = 1;  //..
        std::map<int,int> imu_big_interval;
        int fscanf_return;
        fscanf_return = fscanf(fp,"%lf,%lf,%lf,%lf,%lf,%lf,%lf,",
                    &imu_time,angular_v,angular_v+1,angular_v+2,acceleration,acceleration+1,acceleration+2);
        imu_time=imu_time*0.000000001;
        last_imu_time = imu_time;

        if (fscanf_return != 7)
        {
            std::cout << "1fscanf error " << std::endl;
            fclose(fp);
            return -1;

        }

        for(size_t i = 10; (i < left_image_index.size()-10)&&(i < right_image_index.size()-10) ;++i)
        {

            if (feof(fp))
                break;

            ostringstream stringStream;
            //转换左图
            //cout<<left_image_index[i]<<endl;
            unsigned long long left_image_index_=left_image_index[i];


            std::string left_filename =  string(left_image_path)+"/"+ to_string(left_image_index_)  + string(".png");///构造文件名
            //cout<<left_filename<<endl;
            cv::Mat left_image = cv::imread(left_filename,CV_LOAD_IMAGE_GRAYSCALE);
            time_count_left = left_image_index[i]*0.000000001;


            //转换右图
            unsigned long long right_image_index_=right_image_index[i];
            std::string right_filename = string(right_image_path)+"/" + to_string(right_image_index_) + ".png";
            cv::Mat right_image = cv::imread(right_filename,CV_LOAD_IMAGE_GRAYSCALE);
            if(left_image_index.empty() || right_image_index.empty())
              {
                  std::cout << "right image empty\n";
                  fclose(fp);
                  return -1;
              }
            time_count_right = right_image_index[i]*0.000000001;

            if(time_count_left != time_count_right)
            {
               std::cout << "left image time != right image time .............." << std::endl;
               fclose(fp);
               return -1;
            }

            while (imu_time < left_image_index[i+1]*0.000000001)
            {
                if(last_imu_time > imu_time)
                {
                    std::cout << "imu time disorder!!!!!!!!!!!" << std::endl;
                    return -1;
                }
                if ( imu_time - last_imu_time > 8)
                {
                    imu_big_interval[imu_seq] = imu_time - last_imu_time;
                    std::cout<<"imu time to long!!!!!!!!!!!";
                    //return -1;
                }

                //线加速度
                 Eigen::Vector3d acc0(acceleration[0],acceleration[1],acceleration[2]);
                 Eigen::Vector3d acc1 = PreprocessAccelerator(acc0);
                 Eigen::Vector3d gyro(angular_v[0],angular_v[1],angular_v[2]);

                if(imu_time > last_imu_time)
                {
                    InputIMU(double(imu_time),acc0,gyro);
                    //std::cout.setf(std::ios::fixed, std::ios::floatfield);
                    //std::cout.precision(6);
                    //std::cout << "imu_time: " << imu_time << std::endl;
                }


                last_imu_time = imu_time;
                fscanf_return = fscanf(fp,"%lf,%lf,%lf,%lf,%lf,%lf,%lf,",
                    &imu_time,angular_v,angular_v+1,angular_v+2,acceleration,acceleration+1,acceleration+2);
                imu_time=imu_time*0.000000001;
                if (fscanf_return != 7)
                {
                    std::cout << "3fscanf error: " << fscanf_return << std::endl;
                    std::cout << "imu_time: " << imu_time << std::endl << "image_time: " << time_count_left << std::endl;
                    fclose(fp);
                    std::map<int,int>::iterator it1 ;
                    for(it1 = imu_big_interval.begin();it1 != imu_big_interval.end();it1++)
                    {
                        cout << "total: " << imu_big_interval.size() << "  imu_seq: " << it1->first << "  interval: " << it1->second << endl;
                    }
                    return -1;
                }
            }
            cv::Mat IMAGELEFT=left_image.clone();
            cv::Mat IMAGERIGHT=right_image.clone();
            InputIMAGE(IMAGELEFT, IMAGERIGHT, time_count_left);

            //std::cout.setf(std::ios::fixed, std::ios::floatfield);
            //std::cout.precision(6);
            //std::cout << "result image stamp: " << time_count_left << std::endl;
            usleep(2000);
        }

        fclose(fp);
        return 0;

}



