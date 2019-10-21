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
    ep.eager_mode = false;
    ep.stream_sync = true;
    ep.shared_workspace_enabled = false;

    on_destroy_ = ELX_EVENT_EXIT; // signal exit
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

  eld_conv_t dummy;
  elx_eol_t eol(dummy);

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

  int ret = 1;
  if (ex != nullptr) {
    int event = ex->on_destroy();
    if (event != ELX_EVENT_NORMAL) {
      if (event == ELX_EVENT_TEARDOWN)
        ex->teardown();
      else if (event == ELX_EVENT_EXIT)
        ret = 0;
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
  return ret;
}

void elx_stream::wait(elx_conv_t *ex) {
  // user thread
  std::lock_guard<std::mutex> mlock(ex->mu_);
}

}
