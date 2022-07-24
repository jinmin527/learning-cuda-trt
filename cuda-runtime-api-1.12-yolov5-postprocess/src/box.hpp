#ifndef BOX_HPP
#define BOX_HPP

struct Box{
    float left, top, right, bottom, confidence;
    int label;

    Box() = default;
    Box(float left, float top, float right, float bottom, float confidence, int label):
    left(left), top(top), right(right), bottom(bottom), confidence(confidence), label(label){}
};

#endif // BOX_HPP