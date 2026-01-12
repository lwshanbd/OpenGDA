#!/bin/bash
echo "=== AMD/GPU Environment Variables ==="
env | grep -iE "rocm|hip|hsa|gpu|amd|cuda" | sort

echo ""
echo "=== HSA Runtime Info ==="
ls -la /dev/kfd /dev/dri/render* 2>&1 | head -10

echo ""
echo "=== Process capabilities ==="
cat /proc/self/status | grep -iE "cap|uid|gid"

echo ""
echo "=== ROCm visible devices ==="
echo "ROCR_VISIBLE_DEVICES=${ROCR_VISIBLE_DEVICES:-not set}"
echo "HIP_VISIBLE_DEVICES=${HIP_VISIBLE_DEVICES:-not set}"
echo "GPU_DEVICE_ORDINAL=${GPU_DEVICE_ORDINAL:-not set}"
