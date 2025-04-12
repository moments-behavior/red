#include "yolov8_pose.h"

YOLOv8_pose::YOLOv8_pose(const std::string& engine_file_path, int width, int height)
{
    img_width = width;
    img_height = height;

    std::ifstream file(engine_file_path, std::ios::binary);
    assert(file.good());
    file.seekg(0, std::ios::end);
    auto size = file.tellg();
    file.seekg(0, std::ios::beg);
    char* trtModelStream = new char[size];
    assert(trtModelStream);
    file.read(trtModelStream, size);
    file.close();
    initLibNvInferPlugins(&this->gLogger, "");
    this->runtime = nvinfer1::createInferRuntime(this->gLogger);
    assert(this->runtime != nullptr);

    this->engine = this->runtime->deserializeCudaEngine(trtModelStream, size);

    assert(this->engine != nullptr);
    delete[] trtModelStream;
    this->context = this->engine->createExecutionContext();

    assert(this->context != nullptr);
    cudaStreamCreate(&this->stream);
    this->num_bindings = this->engine->getNbIOTensors();
    for (int i = 0; i < this->num_bindings; ++i) {
        Binding            binding;
        nvinfer1::Dims     dims;
        const char*        name  = this->engine->getIOTensorName(i);
        nvinfer1::DataType dtype = this->engine->getTensorDataType(this->engine->getIOTensorName(i));
        binding.name             = name;
        binding.dsize            = type_to_size(dtype);

        nvinfer1::TensorIOMode ioMode = this->engine->getTensorIOMode(name);
        if (ioMode == nvinfer1::TensorIOMode::kINPUT) {
            this->num_inputs += 1;
            dims         = this->engine->getProfileShape(name, 0, nvinfer1::OptProfileSelector::kMAX);
            binding.size = get_size_by_dims(dims);
            binding.dims = dims;
            this->input_bindings.push_back(binding);
            // set max opt shape
            this->context->setInputShape(name, dims);
        }
        else {
            dims         = this->context->getTensorShape(name);            
            binding.size = get_size_by_dims(dims);
            binding.dims = dims;
            this->output_bindings.push_back(binding);
            this->num_outputs += 1;
        }
    }
}

YOLOv8_pose::~YOLOv8_pose()
{
    // Assuming context, engine, and runtime are all pointers:
    if (this->context) {
        delete this->context;  // Use delete to call the destructor for `context`.
    }
    if (this->engine) {
        delete this->engine;  // Use delete to call the destructor for `engine`.
    }
    if (this->runtime) {
        delete this->runtime;  // Use delete to call the destructor for `runtime`.
    }
    cudaStreamDestroy(this->stream);  // CUDA streams still need to be destroyed explicitly.
    for (auto& ptr : this->device_ptrs) {
        CHECK(cudaFree(ptr));
    }

    for (auto& ptr : this->host_ptrs) {
        CHECK(cudaFreeHost(ptr));
    }
}

void YOLOv8_pose::make_pipe(bool warmup)
{
    // allocate device resources for initialization
    cudaMalloc((void **)&d_temp, 640*439*3);
    cudaMalloc((void **)&d_boarder, 640*640*3);
    cudaMalloc((void **)&d_float, sizeof(float) * 640 * 640 *3);
    cudaMalloc((void **)&d_planar, sizeof(float) * 640 * 640 *3);

    for (auto& bindings : this->input_bindings) {
        void* d_ptr;
        CHECK(cudaMallocAsync(&d_ptr, bindings.size * bindings.dsize, this->stream));
        this->device_ptrs.push_back(d_ptr);
    }

    for (auto& bindings : this->output_bindings) {
        void * d_ptr, *h_ptr;
        size_t size = bindings.size * bindings.dsize;
        CHECK(cudaMallocAsync(&d_ptr, size, this->stream));
        CHECK(cudaHostAlloc(&h_ptr, size, 0));
        this->device_ptrs.push_back(d_ptr);
        this->host_ptrs.push_back(h_ptr);
    }

    if (warmup) {
        for (int i = 0; i < 10; i++) {
            for (auto& bindings : this->input_bindings) {
                size_t size  = bindings.size * bindings.dsize;
                void*  h_ptr = malloc(size);
                memset(h_ptr, 0, size);
                CHECK(cudaMemcpyAsync(this->device_ptrs[0], h_ptr, size, cudaMemcpyHostToDevice, this->stream));
                free(h_ptr);
            }
            this->infer();
        }
        printf("model warmup 10 times\n");
    }
}

void YOLOv8_pose::preprocess_gpu(unsigned char* d_rgb)
{
    const float inp_h  = (float)inp_h_int;
    const float inp_w  = (float)inp_w_int;
    float       width  = img_width;
    float       height = img_height;

    float r    = std::min(inp_h / height, inp_w / width);
    int   padw = std::round(width * r);
    int   padh = std::round(height * r);


    // npp resize, todo: check if resize needed
    NppiSize img_size;
    img_size.width = img_width;
    img_size.height = img_height;
    NppiRect roi;
    roi.x = 0;
    roi.y = 0;
    roi.width = img_width;
    roi.height = img_height;

    NppiSize output_resize_size;
    output_resize_size.width = padw;
    output_resize_size.height = padh;
    NppiRect output_roi;
    output_roi.x = 0;
    output_roi.y = 0;
    output_roi.width = padw;
    output_roi.height = padh;


    const NppStatus npp_result = nppiResize_8u_C3R(d_rgb, 
                                            img_width * sizeof(uchar3), 
                                            img_size, 
                                            roi, 
                                            d_temp, 
                                            inp_w_int * sizeof(uchar3), 
                                            output_resize_size, 
                                            output_roi, 
                                            NPPI_INTER_SUPER);
    if (npp_result != NPP_SUCCESS) {
        std::cerr << "Error executing Resize -- code: " << npp_result << std::endl;
    }

    // make boarder
    NppiSize boarder_size;
    boarder_size.width = inp_w_int;
    boarder_size.height = inp_h_int;

    float dw = inp_w - padw;
    float dh = inp_h - padh;

    dw /= 2.0f;
    dh /= 2.0f;
    int top    = int(std::round(dh - 0.1f));
    int left   = int(std::round(dw - 0.1f));


    Npp8u boarder_color[3] = {114, 114, 114};
    const NppStatus npp_result2 = nppiCopyConstBorder_8u_C3R(d_temp, 
                                            inp_w_int * sizeof(uchar3), 
                                            output_resize_size, 
                                            d_boarder, 
                                            inp_w_int * sizeof(uchar3), 
                                            boarder_size, 
                                            top, 
                                            left, 
                                            boarder_color);

    if (npp_result2 != NPP_SUCCESS) {
        std::cerr << "Error executing CopyConstBoarder -- code: " << npp_result2 << std::endl;
    }


    // blobImageNPP: 1. convert to float: nppiConvert_8u32f_C3R; 2. normalize, nppiDivC_32f_C3IR; 3. transpose: nppiCopy_32f_C3P3R
    const NppStatus npp_result3 = nppiConvert_8u32f_C3R(d_boarder, inp_w_int * sizeof(uchar3), d_float, inp_w_int * sizeof(float3), boarder_size);
    if (npp_result3 != NPP_SUCCESS) {
        std::cerr << "Error executing Convert to float -- code: " << npp_result3 << std::endl;
    }
    
    Npp32f scale_factor[3] = {255.0f, 255.0f, 255.0f};

    const NppStatus npp_result4 = nppiDivC_32f_C3IR(scale_factor, d_float, inp_w_int * sizeof(float3), boarder_size);
    if (npp_result4 != NPP_SUCCESS) {
        std::cerr << "Error executing Convert to float -- code: " << npp_result4 << std::endl;
    }

    float * const inputArr[3] {d_planar, d_planar + inp_w_int * inp_w_int, d_planar + (inp_w_int * inp_w_int * 2)};
    const NppStatus npp_result5 = nppiCopy_32f_C3P3R(d_float, inp_w_int * sizeof(float3), inputArr, inp_w_int * sizeof(float), boarder_size);
    if (npp_result5 != NPP_SUCCESS) {
        std::cerr << "Error executing convert to plannar -- code: " << npp_result5 << std::endl;
    }

    this->pparam.ratio  = 1 / r;
    this->pparam.dw     = dw;
    this->pparam.dh     = dh;
    this->pparam.height = height;
    this->pparam.width  = width;

    const char* name  = this->engine->getIOTensorName(0);
    this->context->setInputShape(name, nvinfer1::Dims{4, {1, 3, inp_w_int, inp_w_int}});
    CHECK(cudaMemcpyAsync(this->device_ptrs[0], d_planar, inp_w_int*inp_w_int*sizeof(float3), cudaMemcpyDeviceToDevice, this->stream));
}


void YOLOv8_pose::letterbox(const cv::Mat& image, cv::Mat& out, cv::Size& size)
{
    const float inp_h  = size.height;
    const float inp_w  = size.width;
    float       height = image.rows;
    float       width  = image.cols;

    float r    = std::min(inp_h / height, inp_w / width);
    int   padw = std::round(width * r);
    int   padh = std::round(height * r);

    cv::Mat tmp;
    if ((int)width != padw || (int)height != padh) {
        cv::resize(image, tmp, cv::Size(padw, padh));
    }
    else {
        tmp = image.clone();
    }

    float dw = inp_w - padw;
    float dh = inp_h - padh;

    dw /= 2.0f;
    dh /= 2.0f;
    int top    = int(std::round(dh - 0.1f));
    int bottom = int(std::round(dh + 0.1f));
    int left   = int(std::round(dw - 0.1f));
    int right  = int(std::round(dw + 0.1f));

    cv::copyMakeBorder(tmp, tmp, top, bottom, left, right, cv::BORDER_CONSTANT, {114, 114, 114});

    cv::dnn::blobFromImage(tmp, out, 1 / 255.f, cv::Size(), cv::Scalar(0, 0, 0), true, false, CV_32F);
    this->pparam.ratio  = 1 / r;
    this->pparam.dw     = dw;
    this->pparam.dh     = dh;
    this->pparam.height = height;
    this->pparam.width  = width;
}

void YOLOv8_pose::copy_from_Mat(const cv::Mat& image)
{
    cv::Mat  nchw;
    auto&    in_binding = this->input_bindings[0];
    auto     width      = in_binding.dims.d[3];
    auto     height     = in_binding.dims.d[2];
    cv::Size size{width, height};
    this->letterbox(image, nchw, size);

    const char* name  = this->engine->getIOTensorName(0);
    this->context->setInputShape(name, nvinfer1::Dims{4, {1, 3, height, width}});

    CHECK(cudaMemcpyAsync(
        this->device_ptrs[0], nchw.ptr<float>(), nchw.total() * nchw.elemSize(), cudaMemcpyHostToDevice, this->stream));
}

void YOLOv8_pose::copy_from_Mat(const cv::Mat& image, cv::Size& size)
{
    cv::Mat nchw;
    this->letterbox(image, nchw, size);
    
    const char* name  = this->engine->getIOTensorName(0);
    this->context->setInputShape(name, nvinfer1::Dims{4, {1, 3, size.height, size.width}});
    CHECK(cudaMemcpyAsync(
        this->device_ptrs[0], nchw.ptr<float>(), nchw.total() * nchw.elemSize(), cudaMemcpyHostToDevice, this->stream));
}
 
void YOLOv8_pose::infer()
{

    for (int32_t i = 0, e = this->engine->getNbIOTensors(); i < e; i++)
    {
        auto const name = this->engine->getIOTensorName(i);
        this->context->setTensorAddress(name, this->device_ptrs[i]);
    }

    this->context->enqueueV3(this->stream);
    for (int i = 0; i < this->num_outputs; i++) {
        size_t osize = this->output_bindings[i].size * this->output_bindings[i].dsize;
        CHECK(cudaMemcpyAsync(
            this->host_ptrs[i], this->device_ptrs[i + this->num_inputs], osize, cudaMemcpyDeviceToHost, this->stream));
    }
    cudaStreamSynchronize(this->stream);
}

void YOLOv8_pose::postprocess(std::vector<Object>& objs, float score_thres, float iou_thres, int topk)
{
    objs.clear();
    auto num_channels = this->output_bindings[0].dims.d[1];
    auto num_anchors  = this->output_bindings[0].dims.d[2];

    auto& dw     = this->pparam.dw;
    auto& dh     = this->pparam.dh;
    auto& width  = this->pparam.width;
    auto& height = this->pparam.height;
    auto& ratio  = this->pparam.ratio;

    std::vector<cv::Rect>           bboxes;
    std::vector<float>              scores;
    std::vector<int>                labels;
    std::vector<int>                indices;
    std::vector<std::vector<float>> kpss;

    cv::Mat output = cv::Mat(num_channels, num_anchors, CV_32F, static_cast<float*>(this->host_ptrs[0]));
    output         = output.t();
    for (int i = 0; i < num_anchors; i++) {
        auto row_ptr    = output.row(i).ptr<float>();
        auto bboxes_ptr = row_ptr;
        auto scores_ptr = row_ptr + 4;
        auto kps_ptr    = row_ptr + 5;

        float score = *scores_ptr;
        if (score > score_thres) {
            float x = *bboxes_ptr++ - dw;
            float y = *bboxes_ptr++ - dh;
            float w = *bboxes_ptr++;
            float h = *bboxes_ptr;

            float x0 = clamp((x - 0.5f * w) * ratio, 0.f, width);
            float y0 = clamp((y - 0.5f * h) * ratio, 0.f, height);
            float x1 = clamp((x + 0.5f * w) * ratio, 0.f, width);
            float y1 = clamp((y + 0.5f * h) * ratio, 0.f, height);

            cv::Rect_<float> bbox;
            bbox.x      = x0;
            bbox.y      = y0;
            bbox.width  = x1 - x0;
            bbox.height = y1 - y0;
            std::vector<float> kps;
            for (int k = 0; k < 4; k++) {
                float kps_x = (*(kps_ptr + 3 * k) - dw) * ratio;
                float kps_y = (*(kps_ptr + 3 * k + 1) - dh) * ratio;
                float kps_s = *(kps_ptr + 3 * k + 2);
                kps_x       = clamp(kps_x, 0.f, width);
                kps_y       = clamp(kps_y, 0.f, height);
                kps.push_back(kps_x);
                kps.push_back(kps_y);
                kps.push_back(kps_s);
            }

            bboxes.push_back(bbox);
            labels.push_back(0);
            scores.push_back(score);
            kpss.push_back(kps);
        }
    }

#ifdef BATCHED_NMS
    cv::dnn::NMSBoxesBatched(bboxes, scores, labels, score_thres, iou_thres, indices);
#else
    cv::dnn::NMSBoxes(bboxes, scores, score_thres, iou_thres, indices);
#endif

    int cnt = 0;
    for (auto& i : indices) {
        if (cnt >= topk) {
            break;
        }
        Object obj;
        obj.rect  = bboxes[i];
        obj.prob  = scores[i];
        obj.label = labels[i];
        obj.kps   = kpss[i];
        objs.push_back(obj);
        cnt += 1;
    }
}

void YOLOv8_pose::copy_keypoints_gpu(float* d_points, const std::vector<Object>& objs)
{
    const int num_point = 4;
    float points[8];
    // TODO: draw both the bbox and the keypoints 
    for (auto& obj : objs) {
        auto& kps = obj.kps;
        for (int k = 0; k < num_point ; k++) {
            points[k*2] = kps[k * 3];
            points[k*2+1] = kps[k * 3+1];
        }
    }
    CHECK(cudaMemcpy(d_points, points, sizeof(float) * 8, cudaMemcpyHostToDevice));
}