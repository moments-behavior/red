// test_mujoco_units.cpp — Diagnose unit/scale issues with real project data
#ifdef RED_HAS_MUJOCO
#include "mujoco_context.h"
#include "mujoco_ik.h"
#include "skeleton.h"
#include "annotation.h"
#include <fstream>
#include <sstream>
#include <cmath>

int main() {
    SkeletonContext skeleton;
    skeleton_initialize("Rat24", &skeleton, Rat24);
    MujocoContext mj;
    mj.load("models/rodent/rodent_no_collision.xml", skeleton);

    std::string csv_path = "/Users/johnsonr/datasets/rat/mujoco_test/"
                           "labeled_data/2025_11_04_06_17_13/keypoints3d.csv";
    std::ifstream csv(csv_path);
    std::string header, line;
    std::getline(csv, header);
    std::getline(csv, line);

    // Parse first frame
    std::vector<Keypoint3D> kp3d(24);
    std::istringstream ss(line);
    std::string tok;
    std::getline(ss, tok, ',');
    int frame_num = std::stoi(tok);
    while (std::getline(ss, tok, ',')) {
        int idx = std::stoi(tok);
        std::string sx, sy, sz;
        std::getline(ss, sx, ','); std::getline(ss, sy, ','); std::getline(ss, sz, ',');
        if (idx < 0 || idx >= 24) continue;
        double x = std::stod(sx), y = std::stod(sy), z = std::stod(sz);
        if (std::abs(x) < 1e6) kp3d[idx] = {x, y, z, true, 1.0f};
    }

    // Centroids
    double tc[3]={0,0,0}, mc[3]={0,0,0}; int cnt=0;
    mj_forward(mj.model, mj.data);
    for (int n = 0; n < 24; n++) {
        if (!kp3d[n].triangulated) continue;
        int s = mj.skeleton_to_site[n]; if (s<0) continue;
        tc[0]+=kp3d[n].x; tc[1]+=kp3d[n].y; tc[2]+=kp3d[n].z;
        mc[0]+=mj.data->site_xpos[3*s]; mc[1]+=mj.data->site_xpos[3*s+1]; mc[2]+=mj.data->site_xpos[3*s+2];
        cnt++;
    }
    printf("Keypoint centroid (raw): (%.1f, %.1f, %.1f)\n", tc[0]/cnt, tc[1]/cnt, tc[2]/cnt);
    printf("Model centroid (m):      (%.4f, %.4f, %.4f)\n", mc[0]/cnt, mc[1]/cnt, mc[2]/cnt);

    // Compute per-axis spans
    double kmin[3]={1e9,1e9,1e9}, kmax[3]={-1e9,-1e9,-1e9};
    double mmin[3]={1e9,1e9,1e9}, mmax[3]={-1e9,-1e9,-1e9};
    for (int n = 0; n < 24; n++) {
        if (!kp3d[n].triangulated) continue;
        int s = mj.skeleton_to_site[n]; if (s<0) continue;
        double kv[3] = {kp3d[n].x, kp3d[n].y, kp3d[n].z};
        double mv[3] = {mj.data->site_xpos[3*s], mj.data->site_xpos[3*s+1], mj.data->site_xpos[3*s+2]};
        for (int c=0;c<3;c++) {
            if (kv[c]<kmin[c]) kmin[c]=kv[c]; if (kv[c]>kmax[c]) kmax[c]=kv[c];
            if (mv[c]<mmin[c]) mmin[c]=mv[c]; if (mv[c]>mmax[c]) mmax[c]=mv[c];
        }
    }
    printf("Keypoint spans: X=%.1f Y=%.1f Z=%.1f (max=%.1f)\n",
           kmax[0]-kmin[0], kmax[1]-kmin[1], kmax[2]-kmin[2],
           std::max({kmax[0]-kmin[0], kmax[1]-kmin[1], kmax[2]-kmin[2]}));
    printf("Model spans (m): X=%.4f Y=%.4f Z=%.4f (max=%.4f)\n",
           mmax[0]-mmin[0], mmax[1]-mmin[1], mmax[2]-mmin[2],
           std::max({mmax[0]-mmin[0], mmax[1]-mmin[1], mmax[2]-mmin[2]}));

    double kspan = std::max({kmax[0]-kmin[0], kmax[1]-kmin[1], kmax[2]-kmin[2]});
    double mspan = std::max({mmax[0]-mmin[0], mmax[1]-mmin[1], mmax[2]-mmin[2]});
    double auto_scale = mspan / kspan;
    printf("Auto scale factor: %.6f (model_span / kp_span)\n\n", auto_scale);

    // Test different scale factors
    float scales[] = {1.0f, 0.001f, (float)auto_scale, 0.0f};
    const char *names[] = {"1.0 (raw)", "0.001 (mm->m)", "manual span-match", "0.0 (auto-detect)"};

    for (int t = 0; t < 4; t++) {
        MujocoIKState ik;
        ik.max_iterations = 200;
        mj.scale_factor = scales[t];
        std::copy(mj.model->qpos0, mj.model->qpos0 + (int)mj.model->nq, mj.data->qpos);
        mj_forward(mj.model, mj.data);
        mujoco_ik_solve(mj, ik, kp3d.data(), 24, frame_num);
        printf("scale=%-20s residual=%.4f m (%.1f mm), %d iters, %.0f ms, %s\n",
               names[t], ik.final_residual, ik.final_residual*1000,
               ik.iterations_used, ik.solve_time_ms,
               ik.converged ? "CONVERGED" : "not converged");
    }

    // Run 20 consecutive frames with auto scale + warm-start
    printf("\n--- 20 consecutive frames (auto-detect scale, 50 iter, warm-start) ---\n");
    MujocoIKState ik2;
    ik2.max_iterations = 50;
    mj.scale_factor = 0.0f; // auto
    std::copy(mj.model->qpos0, mj.model->qpos0 + (int)mj.model->nq, mj.data->qpos);
    mj_forward(mj.model, mj.data);

    csv.clear(); csv.seekg(0);
    std::getline(csv, header);
    int nf = 0; double sum_res=0, sum_ms=0;
    while (std::getline(csv, line) && nf < 20) {
        std::istringstream ss2(line);
        std::string t2;
        std::getline(ss2, t2, ',');
        int fn = std::stoi(t2);
        std::vector<Keypoint3D> kp(24);
        while (std::getline(ss2, t2, ',')) {
            int idx = std::stoi(t2);
            std::string a,b,c;
            std::getline(ss2,a,','); std::getline(ss2,b,','); std::getline(ss2,c,',');
            if (idx<0||idx>=24) continue;
            double x=std::stod(a),y=std::stod(b),z=std::stod(c);
            if (std::abs(x)<1e6) kp[idx]={x,y,z,true,1.0f};
        }
        mujoco_ik_solve(mj, ik2, kp.data(), 24, fn);
        printf("  Frame %5d: %.1f mm, %d iters, %.0f ms\n",
               fn, ik2.final_residual*1000, ik2.iterations_used, ik2.solve_time_ms);
        sum_res+=ik2.final_residual; sum_ms+=ik2.solve_time_ms;
        nf++;
    }
    printf("  Avg: %.1f mm, %.0f ms\n", sum_res/nf*1000, sum_ms/nf);

    return 0;
}
#else
#include <stdio.h>
int main() { printf("MuJoCo not available\n"); return 0; }
#endif
