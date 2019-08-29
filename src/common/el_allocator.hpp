#pragma once

#include <stdio.h>
#include <sys/file.h>
#include <errno.h>
#include <stdint.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <string.h>
#include <unistd.h>
#include <semaphore.h>
#include "el_utils.hpp"

namespace euler {

class process_singleton_t {
public:
  process_singleton_t(const char *key) : key_(key) {
    fd_ = open(key, O_CREAT | O_RDWR, 0644);
    int rc = flock(fd_, LOCK_EX); // blocking
    if (rc) {
      printf("Euler: process_singleton: flock failed: key=%s\n", key);
    } else {
      printf("Euler: process_singleton: flock success: key=%s\n", key);
    }
  }

  ~process_singleton_t() {
    flock(fd_, LOCK_UN);
    close(fd_);
  }

private:
  const char *key_;
  int fd_;
};

#define SETUP_DONE_MASK (0xAABBCCDD)

class shared_workspace_mgr_t {
public:
  struct shared_workspace_header_t {
    uint32_t setup_done_;
    size_t size_;
  };

  shared_workspace_mgr_t(size_t size, const char *key) {
    if (key != nullptr && strlen(key) < sizeof(key_)) {
      sprintf(key_, "%s", key);
    }

    fd_ = shm_open(key_, O_RDWR | O_CREAT, 0644);
    if (fd_ == -1) {
      el_error("shm_open failed");
    }
    size_ = size;
    size_total_ = alignup(sizeof(shared_workspace_header_t), 64) + size;

    struct stat fdst;
    if (fstat(fd_, &fdst) ||
        (fdst.st_size != 0 && fdst.st_size != size_total_)) {
      el_error("Euler: shared workspace fstat error or size does not match");
    }

    if (ftruncate(fd_, size_total_)) {
      el_error("Euler: ftruncate failed");
    }
    workspace_ptr_ =
      mmap(0, size_total_, PROT_READ | PROT_WRITE, MAP_SHARED, fd_, 0);
  }

  bool is_setup_done() {
    if (workspace_ptr_ != nullptr) {
      shared_workspace_header_t *hdr =
        (shared_workspace_header_t *)workspace_ptr_;
      if (hdr->size_ != 0 && size_ != hdr->size_) {
        el_error("Euler: shared workspace size does not match");
      }
      return hdr->setup_done_ == SETUP_DONE_MASK;
    }
    return false;
  }

  void set_setup_done() {
    if (workspace_ptr_ != nullptr) {
      shared_workspace_header_t *hdr =
        (shared_workspace_header_t *)workspace_ptr_;
      hdr->setup_done_ = SETUP_DONE_MASK;
      hdr->size_ = size_;
    }
  }

  ~shared_workspace_mgr_t() {
    if (workspace_ptr_ != nullptr) {
      munmap(workspace_ptr_, size_);
      workspace_ptr_ = nullptr;
    }
    if (fd_ != -1) {
      close(fd_);
      fd_ = -1;
      shm_unlink(key_);
    }
  }

  void *get() {
    return (void *)alignup((size_t)workspace_ptr_
                           + sizeof(shared_workspace_header_t), 64);
  }

private:
  int fd_;
  char key_[256];
  size_t size_;
  size_t size_total_;
  void *workspace_ptr_;
};

// TODO: to-be-replaced with user provided buffer
// TODO: per-thread global buffer
struct galloc {
  static void *&get() {
    /*thread_local*/ static void *ptr_;
    return ptr_;
  }

  static size_t &sz() {
    /*thread_local*/ static size_t sz_;
    return sz_;
  }

  static size_t &ref_cnt() {
    /*thread_local*/ static size_t ref_cnt_;
    return ref_cnt_;
  }

  static void *acquire(size_t size)
  {
    auto &sz_ = sz();
    auto &ptr_ = get();
    size_t sz = ALIGNUP(size, 64);
    if (sz > sz_) {
      if (ptr_) ::free(ptr_);
      MEMALIGN64(&ptr_, sz);
      sz_ = sz;
    }
    ++ref_cnt();
    return ptr_;
  }

  static void release() {
    auto &sz_ = sz();
    auto &ptr_ = get();
    auto &cnt_ = ref_cnt();
    if (--cnt_ == 0 && ptr_ != nullptr) {
      ::free(ptr_);
      ptr_ = nullptr;
      sz_ = 0;
    }
  }
};

}
