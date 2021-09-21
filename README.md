



<!-- PROJECT LOGO -->
<br />
<p align="center">

  <h1 align="center">Image-Processing-CUDA</h1>

  <p align="center">
    Python implementations of popular image processing algorithms, modified to run on GPUs.
    <br />
        
  </p>
</p> 




<!-- ABOUT THE PROJECT -->
## About The Project

This project implements low level image processing techniques such as converting an image to grayscale, edge detecting and reducing noise. These algorithms are commonly used in the preprocessing stages of higher level techniques using machine learning. 
The algorithms have been modified to run on **Nvidia GPUs**, optimally using the 100s of parallel cores available. 

#### Image processing  
Is is a method to perform some operations on an image, in order to get an enhanced image or to extract some useful information from it. It is a type of signal processing in which input is an image and output may be image or characteristics/features associated with that image.

#### CUDA 
CUDA is a parallel computing platform and programming model that makes using a GPU for general purpose computing simple and elegant. For python, cuda is used along with Numba - Numba—a Python compiler from Anaconda that can compile Python code for execution on CUDA capable GPUs


### Built With


* [Cuda Toolkit](https://developer.nvidia.com/cuda-toolkit)
* [Python](https://www.python.org/downloads/release/python-3811/)




<!-- GETTING STARTED -->
## Getting Started


To get a local copy up and running follow these simple example steps.

### Prerequisites

This project requires anaconda to be installed. The following extra packages are required
* cudatoolkit
  ```sh
  conda install cudatoolkit
  ```
* numba
  ```sh
  conda install numba
  ```
* pillow
  ```sh
  conda install pillow
  ```
* matplotlib
  ```sh
  conda install matplotlib
  ```

### Installation
The foll
1. Clone the repo
   ```sh
   git clone https://github.com/MehulSharma1/Image-processing-CUDA.git
   ```
2. Setup the conda environment with the required packages
   ```sh
   npm install
   ```




<!-- USAGE EXAMPLES -->
## Usage

The different alogorithms are implemented in separate notebooks. The filters for gaussian and edge detection can be modified to adjust the intensity of the filters.

To use custom images for image processing, follow these steps:
1. Add your images to the /images folder and update the PATH variable in the notebooks

2. Run the notebook


<!-- ROADMAP -->
## Roadmap

Add Fourier transform.

<!-- CONTRIBUTING -->
## Contributing

Contributions are what make the open source community such an amazing place to learn, inspire, and create. Any contributions you make are **greatly appreciated**.

1. Fork the Project
2. Create your Feature Branch (`git checkout -b feature/AmazingFeature`)
3. Commit your Changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the Branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request



<!-- LICENSE -->
## License

Distributed under the MIT License. See `LICENSE` for more information.



<!-- CONTACT -->
## Contact

Mehul Sharma -  mehul.sharma213@gmail.com
Project Link: [https://github.com/MehulSharma1/Image-processing-CUDA](https://github.com/MehulSharma1/Image-processing-CUDAt)
