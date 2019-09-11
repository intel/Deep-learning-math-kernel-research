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

// TODO: to-be-replaced with user provided buffer
struct galloc {
  static void *&get() {
    thread_local static void *ptr_;
    return ptr_;
  }

  static size_t &sz() {
    thread_local static size_t sz_;
    return sz_;
  }

  static size_t &ref_cnt() {
    thread_local static size_t ref_cnt_;
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

#define WS_BLOCK_SIZE (64 * 1024 * 1024)
struct walloc {
  static void *&get() {
    thread_local static void *ptr_ = nullptr;
    return ptr_;
  }

  static size_t &sz() {
    thread_local static size_t sz_ = 0;
    return sz_;
  }

  static size_t &ref_cnt() {
    thread_local static size_t ref_cnt_ = 0;
    return ref_cnt_;
  }

  static void *acquire(size_t size)
  {
    auto &sz_ = sz();
    auto &ptr_ = get();
    auto &cnt_ = ref_cnt();
    if (ptr_ == nullptr) {
      MEMALIGN64(&ptr_, WS_BLOCK_SIZE);
    }
    auto sz = ALIGNUP(size, 64);
    auto old_sz = sz_;
    auto new_sz = sz_ + sz;
    if (new_sz < WS_BLOCK_SIZE) {
      sz_ = new_sz;
      ++cnt_;
      return (char *)ptr_ + old_sz;
    } else {
      void *p = nullptr;
      MEMALIGN64(&p, sz);
      return p;
    }
  }

  static void release(void *ptr) {
    auto &sz_ = sz();
    auto &ptr_ = get();
    auto &cnt_ = ref_cnt();
    if (ptr >= ptr_ && ptr < ((char *)ptr_ + WS_BLOCK_SIZE)) {
      if (--cnt_ == 0 && ptr_ != nullptr) {
        ::free(ptr_);
        ptr_ = nullptr;
        sz_ = 0;
      }
    } else {
      ::free(ptr);
      ptr = nullptr;
    }
  }
};

#define SETUP_DONE_MASK (0xAABBCCDD)
struct shwalloc {
  struct shwhdr_t {
    uint32_t setup_done_;
    size_t size_;
  };
  static const int hdr_size = ALIGNUP(sizeof(shwhdr_t), 64);;

  static void *&get() {
    static void *ptr_ = nullptr;
    return ptr_;
  }

  static size_t &sz() {
    static size_t sz_ = 0;
    return sz_;
  }

  static size_t &ref_cnt() {
    static size_t ref_cnt_ = 0;
    return ref_cnt_;
  }

  static int &fd() {
    static int fd_ = -1;
    return fd_;
  }

  static void *acquire(size_t size, const char *key)
  {
    auto &sz_ = sz();
    auto &ptr_ = get();
    auto &cnt_ = ref_cnt();
    auto &fd_ = fd();

    if (ptr_ == nullptr) {
      fd_ = shm_open(key, O_RDWR | O_CREAT, 0644);
      if (fd_ == -1) {
        el_error("shm_open failed");
      }

      struct stat fdst;
      if (fstat(fd_, &fdst) ||
          (fdst.st_size != 0 && fdst.st_size != WS_BLOCK_SIZE)) {
        el_error("Euler: shared workspace fstat error or size does not match");
      }

      if (ftruncate(fd_, WS_BLOCK_SIZE)) {
        el_error("Euler: ftruncate failed");
      }
      ptr_ = mmap(0, WS_BLOCK_SIZE, PROT_READ | PROT_WRITE, MAP_SHARED, fd_, 0);
    }
    auto sz = alignup(size + hdr_size, 64);
    auto old_sz = sz_;
    auto new_sz = sz_ + sz;
    if (new_sz < WS_BLOCK_SIZE) {
      sz_ = new_sz;
      ++cnt_;
      shwhdr_t *hdr = (shwhdr_t *)((char *)ptr_ + old_sz);
      hdr->size_ = size;
      hdr->setup_done_ = false;
      return (char *)ptr_ + old_sz + hdr_size;
    } else {
      void *p = nullptr;
      MEMALIGN64(&p, sz);
      return p;
    }
  }

  static bool in_range(void *ptr) {
    auto &ptr_ = get();
    return (ptr != nullptr
            && ptr >= ptr_
            && ptr < ((char *)ptr_ + WS_BLOCK_SIZE));
  }

  static void release(void *ptr, const char *key) {
    auto &sz_ = sz();
    auto &ptr_ = get();
    auto &cnt_ = ref_cnt();
    auto &fd_ = fd();
    if (in_range(ptr)) {
      if (--cnt_ == 0 && ptr_ != nullptr) {
        if (ptr_ != nullptr) {
          munmap(ptr_, WS_BLOCK_SIZE);
          ptr_ = nullptr;
        }
        if (fd_ != -1) {
          close(fd_);
          fd_ = -1;
          shm_unlink(key);
        }
        sz_ = 0;
      }
    } else {
      ::free(ptr);
      ptr = nullptr;
    }
  }

  static bool is_setup_done(void *ptr) {
    if (in_range(ptr)) {
      shwhdr_t *hdr = (shwhdr_t *)((char *)ptr - hdr_size);
      if (hdr->size_ != 0)
        return hdr->setup_done_ == SETUP_DONE_MASK;
    }
    return false;
  }

  static void set_setup_done(void *ptr) {
    if (in_range(ptr)) {
      shwhdr_t *hdr = (shwhdr_t *)((char *)ptr - hdr_size);
      hdr->setup_done_ = SETUP_DONE_MASK;
    }
  }

};

}
