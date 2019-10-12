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

void elx_stream::submit(elx_conv_t *ex) {
  // user thread
  if (ex->ep.stream_sync)
    ex->mu_.lock();
  std::unique_lock<std::mutex> mlock(_mutex);
  _stream.push(ex);
  mlock.unlock();
  _cond.notify_one();
}

int elx_stream::run() {
  std::unique_lock<std::mutex> mlock(_mutex);
  while(_stream.empty()) {
    _cond.wait(mlock);
  }
  euler::elx_conv_t *ex = _stream.front();
  _stream.pop();
  mlock.unlock();

  if (ex != nullptr) {
    if (ex->on_destroy()) {
      ex->teardown();
    } else {
      if (ego.verbose) {
        ex->execute_verbose(ex->output_, ex->input_, ex->weights_, ex->bias_);
      } else {
        ex->execute(ex->output_, ex->input_, ex->weights_, ex->bias_);
      }
    }
    if (ex->ep.stream_sync) {
      ex->mu_.unlock();
    }
  }
  return 0;
}

void elx_stream::wait(elx_conv_t *ex) {
  // user thread
  std::lock_guard<std::mutex> mlock(ex->mu_);
}

}
