#ifndef KIWI_COMMON_BOX_HPP
#define KIWI_COMMON_BOX_HPP

#include <vector>
#include <string>

namespace kiwi{

    struct Box{
        float left, top, right, bottom, confidence;
        int class_label;
        std::string label_name;

        Box() = default;

        Box(float left, float top, float right, float bottom, float confidence, int class_label)
        :left(left), top(top), right(right), bottom(bottom), confidence(confidence), class_label(class_label){}
    };

    typedef std::vector<Box> BoxArray;


    struct Face{
        float left, top, right, bottom, confidence;
        float landmark[10];  // xy, xy, xy        

        Face() = default;

        Face(float left, float top, float right, float bottom, float confidence, int class_label)
        :left(left), top(top), right(right), bottom(bottom), confidence(confidence){}
    };

    typedef std::vector<Face> FaceArray;
};


#endif // KIWI_COMMON_BOX_HPP