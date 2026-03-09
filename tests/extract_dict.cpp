// Extract ArUco dictionary bit patterns for embedding.
#include <opencv2/core.hpp>
#include <opencv2/objdetect/aruco_dictionary.hpp>
#include <cstdio>

int main() {
    auto dict = cv::aruco::getPredefinedDictionary(0); // DICT_4X4_50
    int nmarkers = dict.bytesList.rows;
    int marker_size = dict.markerSize;
    int nbytes = dict.bytesList.cols; // packed bytes per marker per rotation

    printf("// DICT_4X4_50: %d markers, %dx%d, maxCorrection=%d\n",
           nmarkers, marker_size, marker_size, dict.maxCorrectionBits);
    printf("// bytesList: rows=%d cols=%d channels=%d\n",
           dict.bytesList.rows, dict.bytesList.cols, dict.bytesList.channels());
    printf("// 4x4 = 16 bits = 2 bytes per marker. Channels = 4 rotations.\n\n");

    // Read rotation 0 for each marker: 2 bytes = 16-bit pattern
    printf("static constexpr uint16_t DICT_4X4_50[50] = {\n");
    for (int m = 0; m < nmarkers; m++) {
        uint16_t pattern = 0;
        for (int b = 0; b < nbytes; b++) {
            uint8_t byte_val = dict.bytesList.at<cv::Vec4b>(m, b)[0]; // rotation 0
            pattern |= (uint16_t(byte_val) << (8 * (nbytes - 1 - b)));
        }
        // Print with visual verification
        printf("    0x%04x, // id %2d: ", pattern, m);
        for (int r = 0; r < 4; r++) {
            for (int c = 0; c < 4; c++) {
                int bit_pos = 15 - (r * 4 + c);
                printf("%d", (pattern >> bit_pos) & 1);
            }
            if (r < 3) printf("|");
        }
        printf("\n");
    }
    printf("};\n");
    return 0;
}
