#ifndef RED_CAMERA
#define RED_CAMERA


struct Camera {
    cv::Mat K;
    cv::Mat distCoeffs;
    cv::Mat r;
    cv::Mat rvec;
    cv::Mat tvec;
    cv::Mat projectionMat;
};

// bool load_camera_params_from_csv(vector<Camera> cvp, string csv_filename)
// {

//     std::cout << csv_filename << std::endl;
//     ifstream fin;
//     fin.open(csv_filename);
//     if (fin.fail()) throw csv_filename;  // the exception being checked

//     string line;
//     string delimeter = ",";
//     size_t pos = 0;
//     string token;

//     // read csv file with cam parameters and tokenize line for this camera
//     int lineNum = 0;
//     while(!fin.eof()){
//         fin >> line;

//         while ((pos = line.find(delimeter)) != string::npos)
//         {
//             token = line.substr(0, pos);
//             if (lineNum == this->camNum)
//             {
//                 csvCamValues.push_back(stof(token));
//             }
//             line.erase(0, pos + delimeter.length());
//         }
//         lineNum++;
//     }


//     for (int i=0; i<9; i++)
//     {
//         k.push_back(csvCamValues[i]);
//     }
//     for (int i=9; i<18; i++)
//     {
//         r_m.push_back(csvCamValues[i]);
//     }
//     for (int i=18; i<21; i++)
//     {
//         t.push_back(csvCamValues[i]);
//     }
//     for (int i=21; i<25; i++)
//     {
//         d.push_back(csvCamValues[i]);
//     }

//     cvp.K = cv::Mat_<float>(k).reshape(0, 3);
//     // std::cout << "K = " << std::endl << cv::format(cvp.K, cv::Formatter::FMT_PYTHON) << std::endl << std::endl;

//     cvp.distCoeffs = cv::Mat_<float>(d);
//     // std::cout << " distCoeffs  = " << std::endl << cv::format(cvp.distCoeffs, cv::Formatter::FMT_PYTHON) << std::endl << std::endl;

//     cvp.r = cv::Mat_<float>(r_m).reshape(0, 3);
//     // std::cout << "r = " << std::endl << cv::format(cvp.r, cv::Formatter::FMT_PYTHON) << std::endl << std::endl;

//     cvp.tvec = cv::Mat_<float>(t);
//     // std::cout << "tvec = " << std::endl << cv::format(cvp.tvec, cv::Formatter::FMT_PYTHON) << std::endl << std::endl;


//     cv::Rodrigues(cvp.r, cvp.rvec);
//     // std::cout << "rvec = " << std::endl << cv::format(cvp.rvec, cv::Formatter::FMT_PYTHON) << std::endl << std::endl;

//     cv::sfm::projectionFromKRt(cvp.K, cvp.r, cvp.tvec, cvp.projectionMat);
//     // std::cout << "projectionMat = " << std::endl << cv::format(cvp.projectionMat, cv::Formatter::FMT_PYTHON) << std::endl << std::endl;

//     return true;
// }



#endif
