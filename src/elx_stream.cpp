#include <unistd.h>
#include <sys/syscall.h>
#include <sys/types.h>
#include "elx_stream.hpp"
#include "elx_conv.hpp"
#include "el_init.hpp"

#define gettid() syscall(SYS_gettid)

namespace euler {

elx_stream global_stream;


struct elx_eol_t : public elx_conv_t {
public:
  elx_eol_t(eld_conv_t &dc) : elx_conv_t(dc) {
    this->eager_mode = false;
    this->stream_sync = true;
    this->shared_workspace_enabled = false;
    this->exit_thread = true;
  }

  virtual void execute(void *output, void *input, void *weights, void *bias) {}
  virtual ~elx_eol_t() {}

private:
  virtual void set_workspace_buffers(void *base) {}
  virtual void set_scratch_buffers(void *base) {}
};

int set_cpu_affinity() {
  // TODO
  return 0;
}

elx_stream::elx_stream() {
  _threadx = new std::thread([&]{
    set_cpu_affinity();
    // executor thread
    while (run());
  });
  _threadx->detach();
}

elx_stream::~elx_stream() {

  eld_conv_t dc;
  elx_eol_t eol(dc);

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

  int ret = 1;
  if (xc != nullptr) {
    if (xc->on_destroy()) {
      if (xc->exit_thread)
        ret = 0;
      else
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
  return ret;
}

void elx_stream::wait(elx_conv_t *xc) {
  // user thread
  std::lock_guard<std::mutex> mlock(xc->mu);
}

}
