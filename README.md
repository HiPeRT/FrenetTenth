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
The code is part of the following paper, which will be presented at the [EMSOFT23](https://esweek.org/) conference, on late September 2023. It will be released afterwards.
```
@article{muzzini2023frenet,
  author={Muzzini, Filippo and Capodieci, Nicola and Ramanzin, Federico and Burgio, Paolo},
  journal={IEEE Embedded Systems Letters}, 
  title={Optimized Local Path Planner Implementation for GPU-Accelerated Embedded Systems}, 
  year={2023},
  volume={},
  number={},
  pages={1-1},
  doi={10.1109/LES.2023.3298733}
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
