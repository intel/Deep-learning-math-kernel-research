#include <unistd.h>
#include <sys/syscall.h>
#include <sys/types.h>
#include "elx_stream.hpp"
#include "elx_conv.hpp"
#include "el_init.hpp"

#define gettid() syscall(SYS_gettid)

namespace euler {

elx_stream global_stream;

int set_cpu_affinity() {
  // TODO
  return 0;
}

elx_stream::elx_stream() {
  _threadx = new std::thread([&]{
    set_cpu_affinity();
    // executor thread
    while (true) {
      run();
    }
  });
  _threadx->detach();
}

elx_stream::~elx_stream() {
  delete _threadx;
}

void elx_stream::submit(elx_conv_t *xc) {
  // user thread
  if (xc->stream_sync)
    xc->mu.lock();
  std::unique_lock<std::mutex> mlock(_mutex);
  _stream.push(xc);
  mlock.unlock();
  _cond.notify_one();
}

int elx_stream::run() {
  std::unique_lock<std::mutex> mlock(_mutex);
  while(_stream.empty()) {
    _cond.wait(mlock);
  }
  euler::elx_conv_t *xc = _stream.front();
  _stream.pop();
  mlock.unlock();

  if (xc != nullptr) {
    if (xc->on_destroy()) {
      xc->teardown();
    } else {
      if (ego.verbose) {
        xc->execute_verbose(
            xc->output_ptr, xc->input_ptr, xc->weights_ptr, xc->bias_ptr);
      } else {
        xc->execute(
            xc->output_ptr, xc->input_ptr, xc->weights_ptr, xc->bias_ptr);
      }
    }
    if (xc->stream_sync) {
      xc->mu.unlock();
    }
  }
  return 0;
}

void elx_stream::wait(elx_conv_t *xc) {
  // user thread
  std::lock_guard<std::mutex> mlock(xc->mu);
}

}
