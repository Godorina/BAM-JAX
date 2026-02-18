# BAM-JAX LAMMPS Deployment

BAM-JAX (RACE) 모델을 LAMMPS에서 실행하기 위한 가이드입니다.

## Directory Structure

모든 빌드는 `bam/lammps/` 디렉토리 아래에서 이루어집니다.

```
bam/lammps/                          ← 작업 디렉토리
├── scripts/
│   ├── build_lammps.sh              ← 범용 빌드 스크립트
│   └── build_lammps_local.sh        ← 로컬 빌드 스크립트
├── chemtrain/                       ← Step 0에서 clone
│   └── chemtrain-deploy/
│       ├── lib/libconnector.so      ← Step 1 산출물
│       └── build/chemtrain_deployplugin.so  ← Step 2 산출물
├── lammps/                          ← Step 0에서 clone
│   └── build/lmp                    ← Step 3 산출물
├── export_model.py
├── exporter.py
└── ...
```

---

## Quick Build

자동화 스크립트로 전체 빌드:
```
$ cd bam/lammps/scripts
$ bash build_lammps_local.sh
```

개별 단계 건너뛰기:
```
$ bash build_lammps_local.sh --skip-connector          # Step 1 생략
$ bash build_lammps_local.sh --skip-connector --skip-lammps  # Step 2만 실행
$ bash build_lammps_local.sh --cpu-only                 # CPU 전용 빌드
```

아래는 각 단계를 수동으로 실행하는 방법입니다.

---

## Installation for GPU

**의존성 설치 (Ubuntu/Debian)**
```
$ sudo apt-get install cmake build-essential libopenmpi-dev libprotobuf-dev protobuf-compiler
```

**의존성 설치 (RHEL/CentOS)**
```
$ sudo yum install cmake gcc-c++ openmpi-devel protobuf-devel protobuf-compiler
```

**Step 0: 저장소 clone**
```
$ cd bam/lammps
$ git clone https://github.com/tummfm/chemtrain.git
$ git clone -b stable_2Aug2023_update3 --depth 1 https://github.com/lammps/lammps.git
```

**Step 1: chemtrain-deploy connector 빌드 (Bazel)**

`libconnector.so`와 PJRT 플러그인을 빌드합니다. JAX가 설치된 Python 3.11 환경이 필요합니다.
```
$ cd bam/lammps/chemtrain/chemtrain-deploy
$ python build.py
$ python build.py --load_gpu_pjrt_plugin
```

빌드 완료 후 `lib/` 디렉토리에 다음 파일이 생성됩니다:
- `lib/libconnector.so` — XLA/PJRT connector 라이브러리
- `lib/pjrt_plugin.xla_cuda12.so` — CUDA GPU용 PJRT 플러그인

> **참고**: `build.py`는 내부적으로 Bazel을 사용합니다. 첫 빌드 시 XLA 의존성 컴파일로 20-30분 소요됩니다.

**Step 2: LAMMPS 플러그인 빌드 (CMake)**

`libconnector.so`를 링크하여 LAMMPS pair_style 플러그인을 빌드합니다.
```
$ cd bam/lammps/chemtrain/chemtrain-deploy
$ mkdir build && cd build
$ cmake ../lammps_plugin \
    -DLAMMPS_HEADER_DIR=../../lammps/src
$ make -j $(nproc)
```

빌드 완료 후 `build/chemtrain_deployplugin.so`가 생성됩니다.

> **주의**: Step 1에서 생성된 `lib/libconnector.so`가 없으면 링크 에러가 발생합니다.

**Step 3: LAMMPS 빌드 (PLUGIN 패키지)**

MPI 환경 설정 후 빌드합니다. Intel MPI 예시:
```
$ source /opt/intel/oneapi/mpi/latest/env/vars.sh

$ cd bam/lammps/lammps
$ mkdir build && cd build
$ cmake ../cmake \
    -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_CXX_COMPILER=mpicxx \
    -DCMAKE_C_COMPILER=mpicc \
    -DPKG_PLUGIN=on \
    -DBUILD_MPI=on
$ make -j $(nproc)
```

> **주의**: `-DCMAKE_CXX_COMPILER=mpicxx`를 생략하면 MPI 없이 빌드되어 멀티 GPU 실행이 불가합니다.

**환경 변수 설정**

`bam/lammps/` 의 절대 경로를 `BAM_LAMMPS_DIR`로 설정합니다.
```
$ vi ~/.bashrc
```
```
# BAM-JAX LAMMPS
BAM_LAMMPS_DIR=/absolute/path/to/bam/lammps

export PATH=$BAM_LAMMPS_DIR/lammps/build:$PATH
export LAMMPS_PLUGIN_PATH=$BAM_LAMMPS_DIR/chemtrain/chemtrain-deploy/build
export JCN_PJRT_PATH=$BAM_LAMMPS_DIR/chemtrain/chemtrain-deploy/lib
```
```
$ source ~/.bashrc
```

**설치 확인**
```
$ which lmp
>>> .../bam/lammps/lammps/build/lmp

$ lmp -h | grep PLUGIN
>>> ... PLUGIN ...

$ ldd $(which lmp) | grep mpi
>>> libmpi.so.12 => ...
```

---

## Installation for CPU

의존성 설치는 GPU와 동일합니다.

**Step 0: 저장소 clone**
```
$ cd bam/lammps
$ git clone https://github.com/tummfm/chemtrain.git
$ git clone -b stable_2Aug2023_update3 --depth 1 https://github.com/lammps/lammps.git
```

**Step 1: chemtrain-deploy connector 빌드 (Bazel)**
```
$ cd bam/lammps/chemtrain/chemtrain-deploy
$ python build.py
$ python build.py --load_cpu_pjrt_plugin
```

**Step 2: LAMMPS 플러그인 빌드 (CMake)**
```
$ cd bam/lammps/chemtrain/chemtrain-deploy
$ mkdir build && cd build
$ cmake ../lammps_plugin \
    -DLAMMPS_HEADER_DIR=../../lammps/src
$ make -j $(nproc)
```

**Step 3: LAMMPS 빌드 (PLUGIN 패키지)**
```
$ source /opt/intel/oneapi/mpi/latest/env/vars.sh

$ cd bam/lammps/lammps
$ mkdir build && cd build
$ cmake ../cmake \
    -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_CXX_COMPILER=mpicxx \
    -DCMAKE_C_COMPILER=mpicc \
    -DPKG_PLUGIN=on \
    -DBUILD_MPI=on
$ make -j $(nproc)
```

환경 변수 설정 및 확인은 GPU와 동일합니다.

---

## Model Export

학습된 체크포인트를 LAMMPS용 MLIR protobuf로 변환합니다.

```
$ python -m bam.lammps.export_model \
    --ckpt ckpt_best.pkl \
    --config input.json \
    --output model-lammps.ptb
```

`model-lammps.ptb` 파일이 생성되면 준비 완료입니다.

---

## Quick Start

**구조 파일 준비** (ASE로 변환)
```python
from ase.io import read, write
atoms = read("structure.xyz")
write("structure.data", atoms, format="lammps-data")
```

**LAMMPS 입력 파일 작성** (`input.lammps`)
```
units           metal
atom_style      atomic
boundary        p p p

read_data       structure.data

pair_style      chemtrain_deploy cuda12 0.95
pair_coeff      * * model-lammps.ptb 1.1 1.5

comm_modify     cutoff 7.0

neighbor        1.0 bin
neigh_modify    every 1 delay 0 check yes

timestep        0.001

thermo          100
thermo_style    custom step temp pe ke etotal press vol

velocity        all create 300.0 12345 dist gaussian
fix             1 all nve
run             10000
```

**실행**
```
$ lmp -in input.lammps
```

---

## (Optional) Docker / Apptainer

직접 빌드 대신 컨테이너를 사용할 수 있습니다.

```
$ docker build -t bam-lammps:latest -f docker/Dockerfile .
$ docker run --gpus all -v $(pwd):/work bam-lammps:latest -in /work/input.lammps
```

HPC 클러스터 (Apptainer):
```
$ apptainer build bam-lammps.sif docker-daemon://bam-lammps:latest
$ apptainer exec --nv bam-lammps.sif lmp -in input.lammps
```

---

## Notes

- **`pair_style`**: `chemtrain_deploy cuda12 0.95` — `cuda12`는 CUDA 백엔드, `0.95`는 GPU 메모리 비율.
- **`pair_coeff`**: `* * model-lammps.ptb 1.1 1.5` — `1.1`과 `1.5`는 chemtrain_deploy 내부 파라미터.
- **`comm_modify cutoff`**: 최소 7.0 (model cutoff 6.0 + neighbor skin 1.0). chemtrain_deploy가 이 값 미만이면 에러를 발생시킵니다.
- **atom type = 원자번호**: LAMMPS data 파일에서 atom type을 원자번호로 설정해야 합니다 (H=1, C=6, N=7, O=8 등). chemtrain_deploy 내부에서 `species = type - 1`로 변환합니다.
- **`LAMMPS_PLUGIN_PATH`**: chemtrain_deploy 플러그인(`chemtrain_deployplugin.so`)이 있는 디렉토리를 지정해야 합니다.
- **`JCN_PJRT_PATH`**: PJRT 라이브러리 경로. chemtrain-deploy의 `lib/` 디렉토리.
- **units metal**: BAM 모델은 eV, Angstrom, ps 단위계를 사용합니다.
