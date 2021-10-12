#include <stdio.h>
#include <string.h>
#include <iostream>
#include <time.h>
#include <cstdlib>
#include <gsl/gsl_randist.h>
#include <gsl/gsl_randist.h>
#include <vector>
#include <poll.h>
#include <socket.h>


using namespace std;

int main(int argc, char** argv) {

  vector<double> array;
  const gsl_rng_type * T;
  gsl_rng * r;
  srand(time(NULL));

  unsigned int seed = rand();
  gsl_rng_env_setup();

  T = gsl_rng_default;
  r = gsl_rng_alloc (T);
  gsl_rng_set (r, seed);

  int k = 10;
  int N = 10;
  double ppp[k]; // Probability array

  for(int i = 0; i < k; i++) {
    ppp[i] = 0.1;
  }

  unsigned int mult_op[k];
  gsl_ran_multinomial(r, k, N, ppp, mult_op);

  for (size_t i = 0; i < k; i++)
  {
    cout << mult_op[i] << " " << endl;
  }






  // 初始化pfds
  int nfds;
  struct pollfd *pfds = (pollfd*)calloc(nfds, sizeof(struct pollfd));
  for (int j = 0; j < nfds; j++) {
    pfds[j].fd = socket(AF_INET, SOCK_STREAM, 0);
    pfds[j].events = POLLIN;
  }
  // 循环调用poll
  while (num_open_fds > 0) {
    int ready = poll(pfds, nfds, -1);
    // 遍历pfds处理事件
    for (int j = 0; j < nfds; j++) {
      char buf[10];
      if (pfds[j].revents & POLLIN) {
        ssize_t s = read(pfds[j].fd, buf, sizeof(buf));
      } else {   /* POLLERR | POLLHUP */
        close(pfds[j].fd);
        num_open_fds--;
      }
    }
  }

}


struct pollfd {
int   fd;         /* file descriptor */
short events;     /* requested events */
short revents;    /* returned events */
};












