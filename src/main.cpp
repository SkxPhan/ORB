#include <chrono>
#include <iostream>

#include "orb.hpp"

int main(int argc, char** argv) {
  std::string first_file = "../images/1.png";
  std::string second_file = "../images/2.png";

  cv::Mat first_image = cv::imread(first_file, cv::IMREAD_GRAYSCALE);
  cv::Mat second_image = cv::imread(second_file, cv::IMREAD_GRAYSCALE);
  assert(first_image.data != nullptr && second_image.data != nullptr);

  auto orb = ORB();

  auto keypoints1 = orb.findFASTKeypoints(first_image);
  auto keypoints2 = orb.findFASTKeypoints(second_image);

  std::cout << "Number of keypoints in image 1: " << keypoints1.size() << '\n';

  auto descriptors1 = orb.computeBRIEFdescriptors(first_image, keypoints1);
  auto descriptors2 = orb.computeBRIEFdescriptors(second_image, keypoints2);

  auto nbr_descriptors = 0;
  for (const auto& descriptor : descriptors1) {
    if (!descriptor.empty()) ++nbr_descriptors;
  }

  std::cout << "Number of descriptors: " << nbr_descriptors << '\n';

  std::cout << "Displaying keypoints...\n";
  cv::Mat outimg;
  drawKeypoints(first_image, keypoints1, outimg, cv::Scalar::all(-1),
                cv::DrawMatchesFlags::DEFAULT);
  cv::imshow("Keypoints", outimg);
  cv::waitKey(0);
}