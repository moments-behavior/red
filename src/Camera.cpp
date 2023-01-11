#include "Camera.h"
#include "util.h"

#include <string>
#include <iostream>
#include <fstream>
#include <iterator>
#include <vector>
#include <algorithm>
#include <filesystem>

using namespace std;
using namespace util;
namespace fs = std::filesystem;

Camera::Camera(int index, int camNum, std::string rootDir, SkelEnum skelEnum)
{

    this->index = index;
    this->camNum = camNum;
    this->root_dir = rootDir;
    this->name = "Cam" + to_string(camNum);
    this->movie_dir = this->root_dir + "/movies";
    this->label_dir = this->root_dir + "/labeled_data/" + this->name;
    this->calibration_dir = this->root_dir + "/calibration";
    

    this->skelEnum = skelEnum;
    this->isImgHovered = false;
    this->isWindowHovered = false;
    this->LoadCameraParamsFromCSV();
    this->frameData = new FrameData2D(skelEnum);
    // this->get_circle_pts();
}


void Camera::get_circle_pts()
{
    std::vector<float> x;
    std::vector<float> y;
    std::vector<float> z;

    int n = 100;
    float radius = 1473.0f;

    std::vector<cv::Point3f> inPts;

    for (int i; i<=n; i++)
    {
        float angle = (3.14159265358979323846 * 2) * (float(i) / 100.0f);
        x.push_back(sin(angle) * radius);
        y.push_back(cos(angle) * radius);
        z.push_back(0.0f);
    }

    for (int i; i<x.size(); i++)
    {
        cv::Point3f p;
        p.x = x[i];
        p.y = y[i];
        p.z = z[i];
        inPts.push_back(p);
    }

    // cv::Mat imagePts;
    std::vector<cv::Point2f> imgPts;
    cv::projectPoints(inPts, this->cvp.rvec, this->cvp.tvec, this->cvp.K, this->cvp.distCoeffs, imgPts);

    for (int i=0; i<x.size(); i++)
    {
        this->arena_x.push_back( (double)imgPts.at(i).x );
        this->arena_y.push_back( (double)2200 - (double)imgPts.at(i).y );
    }

    // c-style array of points for fillConvexPolygon
    vector<cv::Point> mask_pts;
    for (int i=0; i<x.size(); i++)
    {
        int x_fill = (int)this->arena_x[i];
        int y_fill = (int)this->arena_y[i];
        mask_pts.push_back(cv::Point(x_fill, y_fill));
    }

    // create mask for ball finding
    this->arena_mask = cv::Mat::zeros(2200, 3208, CV_8U);
    cv::fillConvexPoly(this->arena_mask, mask_pts, cv::Scalar(255));
}



void Camera::PrintCameraParams(){
    std::cout << "K = " << std::endl << cv::format(cvp.K, cv::Formatter::FMT_PYTHON) << std::endl << std::endl;
    std::cout << " distCoeffs  = " << std::endl << cv::format(cvp.distCoeffs, cv::Formatter::FMT_PYTHON) << std::endl << std::endl;
    std::cout << "r = " << std::endl << cv::format(cvp.r, cv::Formatter::FMT_PYTHON) << std::endl << std::endl;
    std::cout << "tvec = " << std::endl << cv::format(cvp.tvec, cv::Formatter::FMT_PYTHON) << std::endl << std::endl;
    std::cout << "rvec = " << std::endl << cv::format(cvp.rvec, cv::Formatter::FMT_PYTHON) << std::endl << std::endl;
    std::cout << "projectionMat = " << std::endl << cv::format(cvp.projectionMat, cv::Formatter::FMT_PYTHON) << std::endl << std::endl;
}

bool Camera::LoadCameraParamsFromCSV()
{
    string csv_filename = this->calibration_dir + "/calibration.csv";
    std::cout << csv_filename << std::endl;

    ifstream fin;
    fin.open(csv_filename);
    if (fin.fail()) throw csv_filename;  // the exception being checked

    string line;
    string delimeter = ",";
    size_t pos = 0;
    string token;

    // read csv file with cam parameters and tokenize line for this camera
    int lineNum = 0;
    while(!fin.eof()){
        fin >> line;

        while ((pos = line.find(delimeter)) != string::npos)
        {
            token = line.substr(0, pos);
            if (lineNum == this->camNum)
            {
                csvCamValues.push_back(stof(token));
            }
            line.erase(0, pos + delimeter.length());
        }
        lineNum++;
    }

    // cout << "cam: " << this->camNum << endl;
    // cout << "n params read from csv: " << csvCamValues.size() << endl;

    for (int i=0; i<9; i++)
    {
        k.push_back(csvCamValues[i]);
    }
    for (int i=9; i<18; i++)
    {
        r_m.push_back(csvCamValues[i]);
    }
    for (int i=18; i<21; i++)
    {
        t.push_back(csvCamValues[i]);
    }
    for (int i=21; i<25; i++)
    {
        d.push_back(csvCamValues[i]);
    }

    cvp.K = cv::Mat_<float>(k).reshape(0, 3);
    // std::cout << "K = " << std::endl << cv::format(cvp.K, cv::Formatter::FMT_PYTHON) << std::endl << std::endl;

    cvp.distCoeffs = cv::Mat_<float>(d);
    // std::cout << " distCoeffs  = " << std::endl << cv::format(cvp.distCoeffs, cv::Formatter::FMT_PYTHON) << std::endl << std::endl;

    string ty =  type2str( cvp.K.type() );
    // printf("Matrix: %s %dx%d \n", ty.c_str(), cvp.K.cols, cvp.K.rows );

    cvp.r = cv::Mat_<float>(r_m).reshape(0, 3);
    // std::cout << "r = " << std::endl << cv::format(cvp.r, cv::Formatter::FMT_PYTHON) << std::endl << std::endl;

    cvp.tvec = cv::Mat_<float>(t);
    // std::cout << "tvec = " << std::endl << cv::format(cvp.tvec, cv::Formatter::FMT_PYTHON) << std::endl << std::endl;


    cv::Rodrigues(cvp.r, cvp.rvec);
    // std::cout << "rvec = " << std::endl << cv::format(cvp.rvec, cv::Formatter::FMT_PYTHON) << std::endl << std::endl;

    cv::sfm::projectionFromKRt(cvp.K, cvp.r, cvp.tvec, cvp.projectionMat);
    // std::cout << "projectionMat = " << std::endl << cv::format(cvp.projectionMat, cv::Formatter::FMT_PYTHON) << std::endl << std::endl;

    return true;
}

bool Camera::SaveSkelMap()
{
    map<uint, FrameData2D*>::iterator it = this->frameDataMap.begin();

    string now = current_date_time();
    string filename = this->label_dir + "/" + this->name + "_" + now;
    ofstream output_file(filename);
    
    // write SkelEnum to file to know which Skeleton configuration was used
    output_file << SkelEnumNames[this->skelEnum] << ",\n";

    while (it != this->frameDataMap.end())
    {
        uint frame = it->first;
        FrameData2D* fdata = it->second;

        // write frame number
        output_file << frame << ",";

        // fore each labeled keypoint, write idx, xpos, ypos
        for (uint i=0; i<fdata->nNodes; i++)
        {
            if (fdata->g.isLabeled[i])
            {
                output_file << i << "," << fdata->g.x[i] << "," << fdata->g.y[i] << ",";
            }
        }
        output_file << "\n";

        it++;
    }

    output_file.close();
    cout << this->name << ": " << filename << " created"  << endl;

    return true;
}

bool Camera::LoadSkelMap()
{
    cout << "Camera::LoadSkelMap() called" << endl;
    vector<string> filenames;
    string labeled_data_dir = this->root_dir + "/labeled_data/" + this->name;
    for (const auto & entry : fs::directory_iterator(labeled_data_dir))
    {
        filenames.push_back(entry.path());
    }

    if (filenames.size() == 0)
    {
        cout << "no files in labeled_data_dir for cam " << this->camNum << endl;
        return false;
    };

    for (int i=0; i<filenames.size(); i++)
    {
        cout << filenames.at(i) << endl;
    }

    sort(filenames.begin(), filenames.end());
    string mostRecentFile = filenames.back();

    cout << "mostRecentFile: " << mostRecentFile << endl;

    ifstream fin;
    fin.open(mostRecentFile);
    if (fin.fail()) throw mostRecentFile;  // the exception being checked

    string line;
    string delimeter = ",";
    size_t pos = 0;
    string token;

    // read csv file with cam parameters and tokenize line for this camera
    int lineNum = 0;
    while(!fin.eof()){
        fin >> line;

        while ((pos = line.find(delimeter)) != string::npos)
        {
            token = line.substr(0, pos);
            if (lineNum == 0)
            {
                int idx = getIndex(SkelEnumNames, token);
                this->skelEnum = (SkelEnum)idx;
                line.erase(0, pos + delimeter.length());
            }
            else
            {
                uint frame = stoul(token);
                
                FrameData2D* skel_ptr = new FrameData2D(this->skelEnum);
                std::map<uint, FrameData2D*>::iterator iter(this->frameDataMap.lower_bound(frame));  // get iterator to check if skel exists
                if (iter == this->frameDataMap.end() || frame < iter->first)
                {
                    cout << "no skeleton exists yet for frame " << frame << endl;
                }
                else
                {
                    delete iter->second;
                    this->frameDataMap.erase(iter);
                    cout << "erased pre-existing skeleton for frame " << frame << endl;
                }

                this->frameDataMap.insert(iter, std::make_pair(frame, skel_ptr));
                std::cout << "skeleton created for " << this->name << " on image " << frame << std::endl;
                line.erase(0, pos + delimeter.length());

                while ((pos = line.find(delimeter)) != string::npos)
                {
                    token = line.substr(0, pos);
                    int node = stoi(token);   // get the node index
                    line.erase(0, pos + delimeter.length());

                    pos = line.find(delimeter);
                    token = line.substr(0, pos);
                    double x = stod(token);
                    line.erase(0, pos + delimeter.length());

                    pos = line.find(delimeter);
                    token = line.substr(0, pos);
                    double y = stod(token);
                    line.erase(0, pos + delimeter.length());

                    skel_ptr->g.x[node] = x;
                    skel_ptr->g.y[node] = y; 
                    skel_ptr->g.isLabeled[node] = true;

                    cout << "frame: " << frame << "  node: " << node << "  x: " << skel_ptr->g.x[node] << "  y: " << skel_ptr->g.y[node] << endl;
                }
            }
        }
        lineNum++;
    }

    fin.close();
    return true;
}


string Camera::type2str(int type) {
    string r;

    uchar depth = type & CV_MAT_DEPTH_MASK;
    uchar chans = 1 + (type >> CV_CN_SHIFT);

    switch ( depth ) {
    case CV_8U:  r = "8U"; break;
    case CV_8S:  r = "8S"; break;
    case CV_16U: r = "16U"; break;
    case CV_16S: r = "16S"; break;
    case CV_32S: r = "32S"; break;
    case CV_32F: r = "32F"; break;
    case CV_64F: r = "64F"; break;
    default:     r = "User"; break;
    }

    r += "C";
    r += (chans+'0');

    return r;
}

