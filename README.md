# FrenetTenth

GPU implementation of the Frenet Path Planner algorithm.


## Dependencies
- [Eigen/Dense](https://eigen.tuxfamily.org/dox/GettingStarted.html)
- [matplotlib-cpp](https://github.com/lava/matplotlib-cpp)
- [nlohmann/json](https://github.com/nlohmann/json)

## Usage

### Install dependencies
```
sudo apt-get install build-essential cmake libeigen3-dev python3-matplotlib python3-numpy

```

```
cd include
git clone https://github.com/nlohmann/json.git

```

### compile

```
mkdir build
cd build
cmake ..
make
cd ..
```

### run

`./measuring_experiments_<cpu|gpu>_<double|float|half>


## Citation
The code is part of the following paper. If you use this code in accademic work, please cite us.
```
@article{MUZZINI2024GPUFrenet,
  title = {GPU implementation of the Frenet Path Planner for embedded autonomous systems: A case study in the F1tenth scenario},
  journal = {Journal of Systems Architecture},
  volume = {154},
  pages = {103239},
  year = {2024},
  issn = {1383-7621},
  doi = {https://doi.org/10.1016/j.sysarc.2024.103239},
  url = {https://www.sciencedirect.com/science/article/pii/S1383762124001760},
  author = {Filippo Muzzini and Nicola Capodieci and Federico Ramanzin and Paolo Burgio},
  keywords = {Planning, Autonomous vehicle, Parallel, GPU, Racing},
}

```

## Authors
* **Filippo Muzzini** - [fmuzzini](https://github.com/fmuzzini)
* **Federico Ramanzin** - [FedeRama](https://github.com/FedeRama)

## Project Managers
* **Nicola Capodieci** - [ncapodieci](https://git.hipert.unimore.it/ncapodieci)
* **Paolo Burgio** - [pburgio](https://github.com/pburgio)

## License
**Apache 2.0** - [License](https://opensource.org/licenses/Apache-2.0)

## AD Project
This repository is part of the autonomous driving project of the University of Modena and Reggio Emilia, [read more](https://hipert.github.io/ad_site/).
