#pragma once

#include <queue>
#include <thread>
#include <mutex>
#include <condition_variable>

namespace euler {

struct elx_conv_t;

class elx_stream {
public:
  elx_stream();
  ~elx_stream();
  void submit(elx_conv_t *ep);
  void wait(elx_conv_t *ep);
  int run();

private:
  elx_stream& operator=(const elx_stream&) = delete;
  elx_stream(const elx_stream&) = delete;

  std::queue<elx_conv_t *> _stream;
  mutable std::mutex _mutex;
  std::condition_variable _cond;
  std::thread *_threadx;
};

extern elx_stream global_stream;

}
