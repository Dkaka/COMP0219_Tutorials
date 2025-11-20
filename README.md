# COMP0219 Tutorial Repository 

This repository provides exampels code accomodating the tutorial examples.

For each week, there is a folder containing the corresponding example with code that you could try during the tutorial.

### Important Considerations

#### Workflow
1. Obtain video footage from a convenient camera or dataset at the lab sessions
2. Apply computer vision techniques to achieve the intended outcome at tutorial sessions (This can be done on your laptop or the Pi)
3. Try your code from 2 at the lab session on the Pi with the given camera.
#### Key Differences to Keep in Mind

**Video Format Differences:**
- There will likely be format/resolution/frame rate differences between your source video and the video captured by the Pi
- Plan accordingly when testing your code

**Performance Differences:**
- There may be differences in performance or availability of hardware (e.g., hardware encoder) between your development environment and the Pi
- You might see performance differences between tutorial results and actual results on the Pi, especially when running in real-time

**Optimization Strategies:**
Consider these approaches to ensure your project runs in real-time:
- Use lower resolution
- Reduce frame rate
- Process only one color channel
- Modify hardware design


### Environment Setup

1.  **Create a Python Virtual Environment:**

    It is recommended to use a virtual environment to manage the project's dependencies. The following command will create a virtual environment named `comp0219` at your machine (pi or laptop)

    ```bash
    python3 -m venv comp0219 --system-site-packages
    ```

2.  **Activate the Virtual Environment:**

    Before installing packages and running the scripts, you need to activate the virtual environment:

    ```bash
    source comp0219/bin/activate
    ```

