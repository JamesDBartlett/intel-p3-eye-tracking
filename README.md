# Computer Pointer Controller

Computer-Human Interface Peripheral Signal Manipulation via AI Retina Tracking (CHIPSMART)

## Project Set Up and Installation
Requires:
- One or both of the following hardware:
  - Intel® CPU
  - Intel® Neural Compute Stick 2
- All of the following software:
  - [Ubuntu 18.04.4 LTS (Bionic Beaver)](https://releases.ubuntu.com/18.04/)
  - [Python 3.7.6](https://docs.python.org/release/3.7.6/)
  - [Intel® Distribution of OpenVINO Toolkit 2020.3](https://software.intel.com/content/www/us/en/develop/articles/openvino-2020-3-lts-relnotes.html) [(LTS)](https://software.intel.com/content/www/us/en/develop/articles/openvino-long-term-support-release.html)
  - [This repository](https://github.com/JamesDBartlett/intel-p3-eye-tracking), cloned to a local directory

## Demo
To run this project, simply execute the script titled `demo.sh` in the root directory of this repository:  
`./demo.sh`

## Documentation
This readme file

## Benchmarks
The Intel® hardware I benchmarked with:
- [Intel® Neural Compute Stick 2](https://ark.intel.com/content/www/us/en/ark/products/140109/intel-neural-compute-stick-2.html)
- [Intel® Core™ i3-3110M Processor (Dual-Core, 3M Cache, 2.40 GHz)](https://ark.intel.com/content/www/us/en/ark/products/65700/intel-core-i3-3110m-processor-3m-cache-2-40-ghz.html)

## Results
  
Model Loading Times (ms) 
- Face Detection: 1117.04  
- Facial Landmark Detection: 443.23  
- Gaze Estimation: 326.16  
- Head Pose Estimation: 145.82  
- Total Loading Time (All Models): 1117.99  

Model Inference Times (ms)
- Face Detection: 102.60  
- Facial Landmark Detection: 1.55  
- Gaze Estimation: 9.10  
- Head Pose Estimation: 7.14  
