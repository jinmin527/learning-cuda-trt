

#include <opencv2/opencv.hpp>
#include <common/ilogger.hpp>
#include "builder/trt_builder.hpp"
#include "app_yolo/yolo.hpp"
#include "pybind11.hpp"

using namespace std;
namespace py = pybind11;

class YoloInfer { 
public:
	YoloInfer(
		string engine, Yolo::Type type, int device_id, float confidence_threshold, float nms_threshold,
		Yolo::NMSMethod nms_method, int max_objects, bool use_multi_preprocess_stream
	){
		instance_ = Yolo::create_infer(
			engine, 
			type,
			device_id,
			confidence_threshold,
			nms_threshold,
			nms_method, max_objects, use_multi_preprocess_stream
		);
	}

	bool valid(){
		return instance_ != nullptr;
	}

	shared_future<ObjectDetector::BoxArray> commit(const py::array& image){

		if(!valid())
			throw py::buffer_error("Invalid engine instance, please makesure your construct");

		if(!image.owndata())
			throw py::buffer_error("Image muse be owner, slice is unsupport, use image.copy() inside, image[1:-1, 1:-1] etc.");

		cv::Mat cvimage(image.shape(0), image.shape(1), CV_8UC3, (unsigned char*)image.data(0));
		return instance_->commit(cvimage);
	}

private:
	shared_ptr<Yolo::Infer> instance_;
}; 

bool compileTRT(
    int max_batch_size, string source, string output, bool fp16, int device_id, int max_workspace_size
){
    TRT::set_device(device_id);
    return TRT::compile(
        fp16 ? TRT::Mode::FP16 : TRT::Mode::FP32,
        max_batch_size, source, output, {}, nullptr, "", "", max_workspace_size
    );
}

PYBIND11_MODULE(yolo, m){

    py::class_<ObjectDetector::Box>(m, "ObjectBox")
		.def_property("left",        [](ObjectDetector::Box& self){return self.left;}, [](ObjectDetector::Box& self, float nv){self.left = nv;})
		.def_property("top",         [](ObjectDetector::Box& self){return self.top;}, [](ObjectDetector::Box& self, float nv){self.top = nv;})
		.def_property("right",       [](ObjectDetector::Box& self){return self.right;}, [](ObjectDetector::Box& self, float nv){self.right = nv;})
		.def_property("bottom",      [](ObjectDetector::Box& self){return self.bottom;}, [](ObjectDetector::Box& self, float nv){self.bottom = nv;})
		.def_property("confidence",  [](ObjectDetector::Box& self){return self.confidence;}, [](ObjectDetector::Box& self, float nv){self.confidence = nv;})
		.def_property("class_label", [](ObjectDetector::Box& self){return self.class_label;}, [](ObjectDetector::Box& self, int nv){self.class_label = nv;})
		.def_property_readonly("width", [](ObjectDetector::Box& self){return self.right - self.left;})
		.def_property_readonly("height", [](ObjectDetector::Box& self){return self.bottom - self.top;})
		.def_property_readonly("cx", [](ObjectDetector::Box& self){return (self.left + self.right) / 2;})
		.def_property_readonly("cy", [](ObjectDetector::Box& self){return (self.top + self.bottom) / 2;})
		.def("__repr__", [](ObjectDetector::Box& obj){
			return iLogger::format(
				"<Box: left=%.2f, top=%.2f, right=%.2f, bottom=%.2f, class_label=%d, confidence=%.5f>",
				obj.left, obj.top, obj.right, obj.bottom, obj.class_label, obj.confidence
			);	
		});

    py::class_<shared_future<ObjectDetector::BoxArray>>(m, "SharedFutureObjectBoxArray")
		.def("get", &shared_future<ObjectDetector::BoxArray>::get);

    py::enum_<Yolo::Type>(m, "YoloType")
		.value("V5", Yolo::Type::V5)
		.value("V3", Yolo::Type::V3)
		.value("X", Yolo::Type::X);

	py::enum_<Yolo::NMSMethod>(m, "NMSMethod")
		.value("CPU",     Yolo::NMSMethod::CPU)
		.value("FastGPU", Yolo::NMSMethod::FastGPU);

    py::class_<YoloInfer>(m, "Yolo")
		.def(py::init<string, Yolo::Type, int, float, float, Yolo::NMSMethod, int, bool>(), 
			py::arg("engine"), 
			py::arg("type")                 = Yolo::Type::V5, 
			py::arg("device_id")            = 0, 
			py::arg("confidence_threshold") = 0.4f,
			py::arg("nms_threshold") = 0.5f,
			py::arg("nms_method")    = Yolo::NMSMethod::FastGPU,
			py::arg("max_objects")   = 1024,
			py::arg("use_multi_preprocess_stream") = false
		)
		.def_property_readonly("valid", &YoloInfer::valid, "Infer is valid")
		.def("commit", &YoloInfer::commit, py::arg("image"));

    m.def(
		"compileTRT", compileTRT,
		py::arg("max_batch_size"),
		py::arg("source"),
		py::arg("output"),
		py::arg("fp16")                         = false,
		py::arg("device_id")                    = 0,
		py::arg("max_workspace_size")           = 1ul << 28
	);
}