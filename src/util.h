#ifndef UTIL_H
#define UTIL_H

#include <iostream>
#include <stdlib.h>
#include <stdio.h>
#include <chrono>
#include <time.h>
#include <ctime>
#include <string>
#include <vector>

using namespace std;

namespace util {

    string current_date_time();
    vector<string> getMoviePaths(string movieDir, vector<string> camNames);

    enum SkelEnum { Rat13=0, Rat22=1, Rat22Target2=2, Rat23=3, Rat23Target2=4, Rat10AndTarget=5, TargetOnly=6, RobotBolts=7, Rat10Target2=8, CalibrationFourCorners=9};
    static const vector<string> SkelEnumNames {"Rat13", "Rat22", "Rat22Target2", "Rat23", "Rat23Target2", "Rat10AndTarget", "TargetOnly", "RobotBolts", "Rat10Target2", "CalibrationFourCorners"};
    int getIndex(vector<string> arr, string elem);

}

#endif
