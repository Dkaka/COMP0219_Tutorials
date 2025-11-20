#include <iostream>
#include <fcntl.h>
#include <unistd.h>
#include <termios.h>
#include <cstring>
#include <time.h>
#include <sched.h>
#include <cmath>

void enable_realtime() {
    struct sched_param param;
    param.sched_priority = 70;

    if (sched_setscheduler(0, SCHED_FIFO, &param) < 0) {
        perror("sched_setscheduler");
    }
}

static inline void add_ms(struct timespec &t, long ms) {
    t.tv_nsec += ms * 1000000LL;
    while (t.tv_nsec >= 1000000000LL) {
        t.tv_nsec -= 1000000000LL;
        t.tv_sec++;
    }
}

static inline void sleep_until(const struct timespec &ts) {
    clock_nanosleep(CLOCK_MONOTONIC, TIMER_ABSTIME, &ts, nullptr);
}

int main() {
    enable_realtime();

    const char* port = "/dev/ttyAMA0";
    int serial_fd = open(port, O_RDWR | O_NOCTTY | O_NDELAY);

    if (serial_fd == -1) {
        std::cerr << "Failed to open UART\n";
        return 1;
    }

    // UART config
    struct termios options;
    tcgetattr(serial_fd, &options);
    cfsetispeed(&options, B115200);
    cfsetospeed(&options, B115200);
    cfmakeraw(&options);
    options.c_cflag |= CREAD | CLOCAL;
    tcsetattr(serial_fd, TCSANOW, &options);

    std::cout << "Outputting sine wave: freq=0.33 Hz, range 0–8, step=50ms\n\n";

    // Sine parameters
    const double freq = 0.33;          // Hz
    const double two_pi = 2 * M_PI;
    const double amplitude = 4.0;      // half-range
    const double offset = 4.0;         // center so output is 0..8

    // Real-time clock start
    struct timespec next;
    clock_gettime(CLOCK_MONOTONIC, &next);
    add_ms(next, 50);

    // Use monotonic time to find the actual time in seconds
    struct timespec start;
    clock_gettime(CLOCK_MONOTONIC, &start);

    while (true) {
        // Current time relative to start
        struct timespec now;
        clock_gettime(CLOCK_MONOTONIC, &now);

        double t_sec =
            (now.tv_sec - start.tv_sec) +
            (now.tv_nsec - start.tv_nsec) / 1e9;

        // Compute sine wave
        double y = amplitude * std::sin(two_pi * freq * t_sec) + offset;

        // Format and transmit
        char buffer[32];
        snprintf(buffer, sizeof(buffer), "%.4f\r\n", y);

        write(serial_fd, buffer, strlen(buffer));
        tcdrain(serial_fd);

        std::cout << "Sent: " << buffer;

        // Wait for next tick
        sleep_until(next);
        add_ms(next, 50);
    }

    close(serial_fd);
    return 0;
}
