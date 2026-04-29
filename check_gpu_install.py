"""Sanity-check the SGCRL GPU runtime inside the active Python environment."""

import importlib
import os
import sys


def _version(module_name):
  module = importlib.import_module(module_name)
  return getattr(module, "__version__", "<unknown>")


def main():
  print("Python:", sys.version.replace("\n", " "), flush=True)
  print("Executable:", sys.executable, flush=True)
  print("CONDA_PREFIX:", os.environ.get("CONDA_PREFIX"), flush=True)
  print("CUDA_VISIBLE_DEVICES:", os.environ.get("CUDA_VISIBLE_DEVICES"), flush=True)
  print("JAX_PLATFORM_NAME:", os.environ.get("JAX_PLATFORM_NAME"), flush=True)
  print("JAX_PLATFORMS:", os.environ.get("JAX_PLATFORMS"), flush=True)
  print("LD_LIBRARY_PATH:", os.environ.get("LD_LIBRARY_PATH"), flush=True)

  import jax
  import jax.numpy as jnp
  import jaxlib

  print("jax:", jax.__version__, flush=True)
  print("jaxlib:", jaxlib.__version__, flush=True)
  for module_name in (
      "optax",
      "haiku",
      "acme",
      "tensorflow",
      "tensorflow_probability",
  ):
    try:
      print(f"{module_name}: {_version(module_name)}", flush=True)
    except Exception as exc:  # pylint: disable=broad-except
      print(f"{module_name}: import failed: {exc!r}", flush=True)

  backend = jax.default_backend()
  devices = jax.devices()
  print("JAX default backend:", backend, flush=True)
  print("JAX devices:", devices, flush=True)
  if backend not in ("gpu", "cuda") and not any(d.platform == "gpu" for d in devices):
    raise RuntimeError(
        "JAX does not see a GPU. This usually means CPU-only jaxlib, an empty "
        "CUDA_VISIBLE_DEVICES, or incompatible CUDA/cuDNN libraries.")

  x = jnp.ones((2048, 2048), dtype=jnp.float32)
  y = (x @ x).block_until_ready()
  device_attr = getattr(y, "device", None)
  device = device_attr() if callable(device_attr) else device_attr
  print("JAX matmul device:", device, flush=True)
  print("GPU sanity check passed.", flush=True)


if __name__ == "__main__":
  main()
