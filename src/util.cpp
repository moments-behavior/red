#include "util.h"
#include <string>
#include <algorithm>
#include <iterator>
using namespace std;

namespace util {

    std::string current_date_time() {
        time_t     now = time(0);
        struct tm  tstruct;
        char       buf[80];
        tstruct = *localtime(&now);
        strftime(buf, sizeof(buf), "%Y-%m-%d_%X", &tstruct);
        return buf;
    }

    vector<string> getMoviePaths(string movieDir, vector<string> camNames) {
        vector<string> out;

        for (int i=0; i<camNames.size(); i++)
        {
            out.push_back(movieDir + "/" + camNames.at(i) + ".mp4");
        }

        return out;
    }

    int getIndex(vector<string> arr, string elem)
    {
        auto it = find(arr.begin(), arr.end(), elem);
    
        // If element was found
        if (it != arr.end()) 
        {
            int index = it - arr.begin();
            return index;
        }
        else {
            return -1;
        }
    }
}