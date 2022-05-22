# M1GPU Benchmark

## environment

```shell
macOS Monterey
version 12.2.1
MacBook Air (M1, 2020)
chip: Apple M1
memory: 8GB
```

## vectorAdd

With Metal, GPU memory can be allocated and used in three ways: Private, Shared, and Managed. In this benchmark, I used Private, which is the fastest method, but there was no difference in speed when running Shared or Managed. 68GB/s is fast for CPU memory, but it is slow for GPU memory (the memory band of the V100 is 800 GB/s), so it is not suitable for HPC use.

```shell
shared to private: copy_size: 1.02 GB, bandwidth: 22.16 GB/ s
lendth: 128000000, rw_size: 1.54 GB, bandwidth: 56.93 GB/ s, duration: 0.027 s
lendth: 128000000, rw_size: 1.54 GB, bandwidth: 2.80 GB/ s, duration: 0.549 s
private to shared: copy_size: 0.51 GB, bandwidth: 28.98 GB/ s
```
