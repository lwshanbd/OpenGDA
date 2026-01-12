#!/bin/bash
# test_barrier.sh - 重复测试barrier-verify 10次

ITERATIONS=10000
WINDOW_SIZE=32
NODES=4
REPEAT=10

echo "=========================================="
echo "Barrier Verification Stress Test"
echo "  Iterations per run: $ITERATIONS"
echo "  Window size: $WINDOW_SIZE"
echo "  Nodes: $NODES"
echo "  Repeat: $REPEAT times"
echo "=========================================="
echo ""

PASSED=0
FAILED=0

for i in $(seq 1 $REPEAT); do
    echo -n "Run $i/$REPEAT: "

    # 运行测试，捕获输出
    OUTPUT=$(srun -N $NODES -n $NODES --ntasks-per-node=1 ./barrier-verify $ITERATIONS $WINDOW_SIZE 2>&1)

    # 检查所有rank是否都PASS
    PASS_COUNT=$(echo "$OUTPUT" | grep -c "Status: PASS")
    ERROR_COUNT=$(echo "$OUTPUT" | grep -c "Status: FAIL")

    if [ "$PASS_COUNT" -eq "$NODES" ] && [ "$ERROR_COUNT" -eq 0 ]; then
        echo "OK (all $NODES ranks passed)"
        PASSED=$((PASSED + 1))
    else
        echo "FAILED!"
        echo "$OUTPUT"
        FAILED=$((FAILED + 1))
    fi
done

echo ""
echo "=========================================="
echo "Summary: $PASSED/$REPEAT passed, $FAILED failed"
echo "=========================================="

if [ "$FAILED" -eq 0 ]; then
    echo "All tests PASSED!"
    exit 0
else
    echo "Some tests FAILED!"
    exit 1
fi
