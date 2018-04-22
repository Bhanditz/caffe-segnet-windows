#ifndef CAFFE_UTIL_IO_H_
#define CAFFE_UTIL_IO_H_

// https://github.com/Rprop/caffe-segnet-windows
#if !defined(WIN32) && !defined(_WINDOWS)
# include <unistd.h>
#else
# undef  NOMINMAX // conflicts with std::max
# define NOMINMAX
# include <windows.h>
# include <io.h>
# include <direct.h>
#endif
#include <string>

#include "google/protobuf/message.h"
#include "hdf5.h"
#include "hdf5_hl.h"

#include "caffe/blob.hpp"
#include "caffe/common.hpp"
#include "caffe/proto/caffe.pb.h"

#define HDF5_NUM_DIMS 4

namespace caffe {

using ::google::protobuf::Message;

inline void MakeTempFilename(string* temp_filename) {
  temp_filename->clear();
#if !defined(WIN32) && !defined(_WINDOWS)
  *temp_filename = "/tmp/caffe_test.XXXXXX";
  char* temp_filename_cstr = new char[temp_filename->size() + 1];
  // NOLINT_NEXT_LINE(runtime/printf)
  strcpy(temp_filename_cstr, temp_filename->c_str());
  int fd = mkstemp(temp_filename_cstr);
  CHECK_GE(fd, 0) << "Failed to open a temporary file at: " << *temp_filename;
  close(fd);
  *temp_filename = temp_filename_cstr;
  delete[] temp_filename_cstr;
#else
  char temp_dir[MAX_PATH + 1];
  GetTempPathA(sizeof(temp_dir), temp_dir);
  strcat_s(temp_dir, sizeof(temp_dir), "caffe_test.XXXXXX");
  _mktemp_s(temp_dir, sizeof(temp_dir));
  *temp_filename = temp_dir;
#endif
}

inline void MakeTempDir(string* temp_dirname) {
  temp_dirname->clear();
#if !defined(WIN32) && !defined(_WINDOWS)
  *temp_dirname = "/tmp/caffe_test.XXXXXX";
  char* temp_dirname_cstr = new char[temp_dirname->size() + 1];
  // NOLINT_NEXT_LINE(runtime/printf)
  strcpy(temp_dirname_cstr, temp_dirname->c_str());
  char* mkdtemp_result = mkdtemp(temp_dirname_cstr);
  CHECK(mkdtemp_result != NULL)
      << "Failed to create a temporary directory at: " << *temp_dirname;
  *temp_dirname = temp_dirname_cstr;
  delete[] temp_dirname_cstr;
#else
  char temp_dir[MAX_PATH + 1];
  GetTempPathA(sizeof(temp_dir), temp_dir);
  strcat_s(temp_dir, sizeof(temp_dir), "caffe_test.XXXXXX");
  _mktemp_s(temp_dir, sizeof(temp_dir));
  _mkdir(temp_dir);
  *temp_dirname = temp_dir;
#endif
}

CAFFE_API_ bool ReadProtoFromTextFile(const char* filename, Message* proto);

inline bool ReadProtoFromTextFile(const string& filename, Message* proto) {
  return ReadProtoFromTextFile(filename.c_str(), proto);
}

inline void ReadProtoFromTextFileOrDie(const char* filename, Message* proto) {
  CHECK(ReadProtoFromTextFile(filename, proto));
}

inline void ReadProtoFromTextFileOrDie(const string& filename, Message* proto) {
  ReadProtoFromTextFileOrDie(filename.c_str(), proto);
}

void WriteProtoToTextFile(const Message& proto, const char* filename);
inline void WriteProtoToTextFile(const Message& proto, const string& filename) {
  WriteProtoToTextFile(proto, filename.c_str());
}

CAFFE_API_ bool ReadProtoFromBinaryFile(const char* filename, Message* proto);

inline bool ReadProtoFromBinaryFile(const string& filename, Message* proto) {
  return ReadProtoFromBinaryFile(filename.c_str(), proto);
}

inline void ReadProtoFromBinaryFileOrDie(const char* filename, Message* proto) {
  CHECK(ReadProtoFromBinaryFile(filename, proto));
}

inline void ReadProtoFromBinaryFileOrDie(const string& filename,
                                         Message* proto) {
  ReadProtoFromBinaryFileOrDie(filename.c_str(), proto);
}


CAFFE_API_ void WriteProtoToBinaryFile(const Message& proto, const char* filename);
inline void WriteProtoToBinaryFile(
    const Message& proto, const string& filename) {
  WriteProtoToBinaryFile(proto, filename.c_str());
}

CAFFE_API_ bool ReadFileToDatum(const string& filename, const int label, Datum* datum);

inline bool ReadFileToDatum(const string& filename, Datum* datum) {
  return ReadFileToDatum(filename, -1, datum);
}

CAFFE_API_ bool ReadImageToDatum(const string& filename, const int label,
    const int height, const int width, const bool is_color,
    const std::string & encoding, Datum* datum);

inline bool ReadImageToDatum(const string& filename, const int label,
    const int height, const int width, const bool is_color, Datum* datum) {
  return ReadImageToDatum(filename, label, height, width, is_color,
                          "", datum);
}

inline bool ReadImageToDatum(const string& filename, const int label,
    const int height, const int width, Datum* datum) {
  return ReadImageToDatum(filename, label, height, width, true, datum);
}

inline bool ReadImageToDatum(const string& filename, const int label,
    const bool is_color, Datum* datum) {
  return ReadImageToDatum(filename, label, 0, 0, is_color, datum);
}

inline bool ReadImageToDatum(const string& filename, const int label,
    Datum* datum) {
  return ReadImageToDatum(filename, label, 0, 0, true, datum);
}

inline bool ReadImageToDatum(const string& filename, const int label,
    const std::string & encoding, Datum* datum) {
  return ReadImageToDatum(filename, label, 0, 0, true, encoding, datum);
}

CAFFE_API_ bool DecodeDatumNative(Datum* datum);
CAFFE_API_ bool DecodeDatum(Datum* datum, bool is_color);

CAFFE_API_ cv::Mat ReadImageToCVMat(const string& filename,
    const int height, const int width, const bool is_color,
    const bool nearest_neighbour_interp);

CAFFE_API_ cv::Mat ReadImageToCVMat(const string& filename,
    const int height, const int width, const bool is_color);

CAFFE_API_ cv::Mat ReadImageToCVMat(const string& filename,
    const int height, const int width);

CAFFE_API_ cv::Mat ReadImageToCVMat(const string& filename,
    const bool is_color);

CAFFE_API_ cv::Mat ReadImageToCVMat(const string& filename);

CAFFE_API_ cv::Mat DecodeDatumToCVMatNative(const Datum& datum);
CAFFE_API_ cv::Mat DecodeDatumToCVMat(const Datum& datum, bool is_color);

CAFFE_API_ void CVMatToDatum(const cv::Mat& cv_img, Datum* datum);

template <typename Dtype>
void hdf5_load_nd_dataset_helper(
    hid_t file_id, const char* dataset_name_, int min_dim, int max_dim,
    Blob<Dtype>* blob);

template <typename Dtype>
void hdf5_load_nd_dataset(
    hid_t file_id, const char* dataset_name_, int min_dim, int max_dim,
    Blob<Dtype>* blob);

template <typename Dtype>
void hdf5_save_nd_dataset(
    const hid_t file_id, const string& dataset_name, const Blob<Dtype>& blob);

}  // namespace caffe

#endif   // CAFFE_UTIL_IO_H_
