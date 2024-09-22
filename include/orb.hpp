#ifndef ORB_HPP
#define ORB_HPP

#include <opencv2/opencv.hpp>
#include <vector>

using DescType = std::vector<uint32_t>;  // Descriptor type

class ORB {
 public:
  ORB();
  ~ORB() = default;

  std::vector<cv::KeyPoint> findFASTKeypoints(const cv::Mat &img,
                                              int threshold = 40) const;
  std::vector<DescType> computeBRIEFdescriptors(
      const cv::Mat &img, std::vector<cv::KeyPoint> &keypoints) const;
};

#endif  // ORB_HPP