#include "LabelManager.h"
#include "util.h"

#include <string>
#include <iostream>
#include <fstream>
#include <iterator>
#include <vector>
#include <algorithm>
#include <filesystem>

namespace fs = std::filesystem;
using namespace util;

LabelManager::LabelManager(std::vector<Camera*> cameras)
{
    this->cameras = cameras;
    this->nCams = cameras.size();
    this->nNodes = cameras[0]->frameData->nNodes;
    this->nEdges = cameras[0]->frameData->nEdges;
    this->skelEnum = cameras[0]->skelEnum;
    this->numViewsLabeled = std::vector<int> (this->nNodes, 0);
    this->labeled_data_dir = cameras[0]->root_dir + "/labeled_data/worldKeyPoints";
    this->skelWorld = new SkelWorld(cameras[0]->frameData);

    for (int i=0; i<this->nCams; i++)
    {
        std::cout << this->cameras[i]->name << " was loaded in LabelManager" << std::endl;
        isLabeled.push_back(std::vector<bool> (nNodes, false));
    }

    if (fs::exists(this->labeled_data_dir)){
    cout << "cannot create labeled_data_dir... it already exists" << endl;
    }
    else{
        fs::create_directories(this->labeled_data_dir);
        cout << "created camKeyPointsDir: " << this->labeled_data_dir << endl;
    }
}

bool LabelManager::SaveWorldKeyPoints()
{

    map<uint, SkelWorld*>::iterator it = this->skelWorldMap.begin();

    string now = current_date_time();
    string filename = this->labeled_data_dir + "/" + "keypoints_" + now;
    ofstream output_file(filename);

    // write frameDataEnum to file to know which skelWorld configuration was used
    output_file << SkelEnumNames[this->cameras[0]->skelEnum] << ",\n";

    while (it != this->skelWorldMap.end())
    {
        uint frame = it->first;
        SkelWorld* skelWorld = it->second;

        // write frame number
        output_file << frame << ",";

        // fore each labeled keypoint, write idx, xpos, ypos, zpos
        for (uint i=0; i<skelWorld->nNodes; i++)
        {
            if (skelWorld->g.isLabeled[i])
            {
                output_file << i << "," << skelWorld->g.x[i] << "," << skelWorld->g.y[i] << "," << skelWorld->g.z[i] << ",";
            }
        }
        output_file << "\n";

        it++;
    }

    output_file.close();
    cout << filename << " created"  << endl;

    return true;
}

bool LabelManager::LoadWorldSkels()
{
    cout << "LabelManager::LoadWorldframeDatas() called" << endl;
    vector<string> filenames;
    for (const auto & entry : fs::directory_iterator(this->labeled_data_dir))
    {
        filenames.push_back(entry.path());
    }

    if (filenames.size() == 0)
    {
        cout << "no files in labeled_data_dir" << endl;
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
                
                SkelWorld* skel_ptr = new SkelWorld(this->skelEnum);
                std::map<uint, SkelWorld*>::iterator iter(this->skelWorldMap.lower_bound(frame));  // get iterator to check if frameData exists
                if (iter == this->skelWorldMap.end() || frame < iter->first)
                {
                    cout << "no frameDataWorld exists yet for frame " << frame << endl;
                }
                else
                {
                    delete iter->second;
                    this->skelWorldMap.erase(iter);
                    cout << "erased pre-existing skelWorld for frame " << frame << endl;
                }

                this->skelWorldMap.insert(iter, std::make_pair(frame, skel_ptr));
                std::cout << "SkelWorld created for image " << frame << std::endl;
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

                    pos = line.find(delimeter);
                    token = line.substr(0, pos);
                    double z = stod(token);
                    line.erase(0, pos + delimeter.length());

                    skel_ptr->g.x[node] = x;
                    skel_ptr->g.y[node] = y; 
                    skel_ptr->g.z[node] = z; 
                    skel_ptr->g.isLabeled[node] = true;

                    cout << "frame: " << frame << "  node: " << node << "  x: " << skel_ptr->g.x[node] \
                     << "  y: " << skel_ptr->g.y[node] << "  z: " << skel_ptr->g.z[node] << endl;
                }
            }
        }
        lineNum++;
    }

    fin.close();
    return true;
}

bool LabelManager::LoadCameraSkels()
{
    for (int i=0; i<this->cameras.size(); i++)
    {

        cout << "loading from: " << cameras.at(i)->name << endl;
        this->cameras.at(i)->LoadSkelMap();
    }

    return true;
}
