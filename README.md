### Download model
  ```bash
  chmod 755 install_model.sh
  ./install_model.sh
  ```
### How to run the code 

* C++:

  * using cmake :
    ```bash
    mkdir build
    cd build
    cmake ../
    make
    ./yolo_single_frame
    ```

  * using pkg-config:

    ```bash
    g++ yolo_single_frame.cpp -o yolo_single_frame `pkg-config --cflags --libs opencv4`
    ./yolo_Single_frame
    ```

##### Reference By

Want to become an expert in AI? [AI Courses by OpenCV](https://opencv.org/courses/) is a great place to start. 

<a href="https://opencv.org/courses/">
<p align="center"> 
<img src="https://www.learnopencv.com/wp-content/uploads/2020/04/AI-Courses-By-OpenCV-Github.png">
</p>
</a>
