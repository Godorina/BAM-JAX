# BAM-JAX LAMMPS deployment container
#
# Build:
#   docker build -t bam-lammps:latest -f deploy/docker/Dockerfile .
#
# Run:
#   docker run --gpus all -v $(pwd):/work bam-lammps:latest lmp -in /work/input.lammps
#
# Convert to Apptainer (HPC):
#   apptainer build bam-lammps.sif docker-daemon://bam-lammps:latest
#   apptainer exec --nv bam-lammps.sif lmp -in input.lammps

FROM nvidia/cuda:12.4.0-devel-ubuntu22.04 AS builder

ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get install -y \
    cmake git python3 python3-pip \
    libopenmpi-dev openmpi-bin \
    libprotobuf-dev protobuf-compiler \
    && rm -rf /var/lib/apt/lists/*

# Install JAX (for Python export step, no version constraint)
# chemtrain not needed: local bam/lammps/deploy/ module replaces it
RUN pip3 install --no-cache-dir \
    "jax[cuda12]" \
    "flax" \
    "e3nn-jax>=0.20.8" \
    "jraph>=0.0.6"

# Build chemtrain-deploy + LAMMPS
COPY deploy/scripts/build_lammps.sh /tmp/build_lammps.sh
RUN bash /tmp/build_lammps.sh

# --- Runtime stage ---
FROM nvidia/cuda:12.4.0-runtime-ubuntu22.04

RUN apt-get update && apt-get install -y \
    libopenmpi3 openmpi-bin \
    libprotobuf23 \
    && rm -rf /var/lib/apt/lists/*

COPY --from=builder /opt/bam-lammps/ /opt/bam-lammps/
ENV PATH="/opt/bam-lammps/bin:${PATH}"
ENV LD_LIBRARY_PATH="/opt/bam-lammps/lib:${LD_LIBRARY_PATH}"

WORKDIR /work
ENTRYPOINT ["lmp"]
