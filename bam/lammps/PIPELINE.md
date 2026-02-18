# BAM-JAX LAMMPS Pipeline

전체 파이프라인: 빌드 → 모델 변환 → LAMMPS 실행

---

## 1. Build Pipeline

```
[Step 0] git clone
    │
    ├── tummfm/chemtrain ──→ bam/lammps/chemtrain/
    │                            └── chemtrain-deploy/
    │                                 ├── connector/    (C++ 소스)
    │                                 ├── lammps_plugin/ (C++ 소스)
    │                                 └── build.py
    │
    └── lammps/lammps ─────→ bam/lammps/lammps/
                                  └── src/             (헤더 파일)


[Step 1] python build.py (Bazel, ~25분)
    │
    │  connector/*.cpp ──Bazel──→ lib/libconnector.so (214MB)
    │  JAX 패키지 ──복사──→ lib/pjrt_plugin.xla_cuda12.so (260MB)
    │
    ▼

[Step 2] cmake ../lammps_plugin (CMake)
    │
    │  lammps_plugin/*.cpp
    │       + libconnector.so (링크)
    │       + lammps/src/*.h  (헤더)
    │       ──CMake──→ build/chemtrain_deployplugin.so (158KB)
    │
    ▼

[Step 3] cmake ../cmake (CMake)
    │
    │  lammps 전체 소스
    │       + PKG_PLUGIN=on
    │       + BUILD_MPI=on
    │       ──CMake──→ lammps/build/lmp (7.7MB)
    │
    ▼

[빌드 완료]
```

---

## 2. Model Export Pipeline

```
[학습]
    │
    ▼
ckpt_best.pkl (429M params, RACE 모델)
    │
    │  python -m bam.lammps.export_model
    │       --ckpt ckpt_best.pkl
    │       --config input.json
    │       --output model-lammps.ptb
    │
    │  내부 과정:
    │  ┌─────────────────────────────────────────┐
    │  │ 1. pkl 로드 (EMA params 우선)            │
    │  │ 2. RACE 모델 생성 (periodic=False)       │
    │  │ 3. jax.export → MLIR/StableHLO 컴파일   │
    │  │ 4. protobuf 직렬화 (.ptb)                │
    │  └─────────────────────────────────────────┘
    │
    ▼
model-lammps.ptb (MLIR 컴파일된 GPU 커널)
```

---

## 3. Runtime Pipeline

```
lmp -in input.lammps
    │
    │  환경변수 참조:
    │  LAMMPS_PLUGIN_PATH → chemtrain-deploy/build/
    │  JCN_PJRT_PATH      → chemtrain-deploy/lib/
    │
    ▼
┌─── LAMMPS (lmp) ───────────────────────────────────────┐
│                                                         │
│  pair_style chemtrain_deploy cuda12 0.95                │
│       │                                                 │
│       ▼                                                 │
│  chemtrain_deployplugin.so  (동적 로딩)                  │
│       │                                                 │
│       ▼                                                 │
│  libconnector.so  (XLA 런타임)                           │
│       │                                                 │
│       ├──→ pjrt_plugin.xla_cuda12.so  (GPU 백엔드)      │
│       │                                                 │
│       └──→ model-lammps.ptb  (pair_coeff에서 로드)       │
│            │                                            │
│            ▼                                            │
│       MLIR → GPU 커널 컴파일 + 실행                      │
│                                                         │
│  ┌──────────────────────────────────┐                   │
│  │  매 timestep:                    │                   │
│  │  positions ──→ connector         │                   │
│  │                  │               │                   │
│  │                  ▼               │                   │
│  │              GPU 연산             │                   │
│  │              (energy, forces)    │                   │
│  │                  │               │                   │
│  │                  ▼               │                   │
│  │  forces ←── connector            │                   │
│  │     │                            │                   │
│  │     ▼                            │                   │
│  │  MD 적분 (위치/속도 업데이트)      │                   │
│  └──────────────────────────────────┘                   │
│                                                         │
└─────────────────────────────────────────────────────────┘
    │
    ▼
출력: dump 파일, thermo 로그
```

---

## 4. File Dependency Summary

```
ckpt_best.pkl ──export──→ model-lammps.ptb ─┐
                                             │
connector/*.cpp ──Bazel──→ libconnector.so ──┤
                                             │
JAX ──복사──→ pjrt_plugin.xla_cuda12.so ─────┤
                                             │
lammps_plugin/*.cpp ──CMake──→ plugin.so ────┤
                                             │
lammps 소스 ──CMake──→ lmp ──────────────────┤
                                             │
                               ┌─────────────┘
                               ▼
                     lmp -in input.lammps
                          (MD 시뮬레이션)
```

---

## 5. Build Artifacts

| Step | 도구 | 산출물 | 위치 | 크기 |
|------|------|--------|------|------|
| 1 | Bazel | `libconnector.so` | `chemtrain-deploy/lib/` | 214MB |
| 1 | 복사 | `pjrt_plugin.xla_cuda12.so` | `chemtrain-deploy/lib/` | 260MB |
| 2 | CMake | `chemtrain_deployplugin.so` | `chemtrain-deploy/build/` | 158KB |
| 3 | CMake | `lmp` | `lammps/build/` | 7.7MB |
| export | Python | `model-lammps.ptb` | 사용자 지정 | - |

---

## 6. Environment Variables

```bash
# LAMMPS 실행 파일 경로
export PATH=<bam/lammps>/lammps/build:$PATH

# LAMMPS가 플러그인을 찾는 경로
export LAMMPS_PLUGIN_PATH=<bam/lammps>/chemtrain/chemtrain-deploy/build

# libconnector가 PJRT 플러그인을 찾는 경로
export JCN_PJRT_PATH=<bam/lammps>/chemtrain/chemtrain-deploy/lib
```
