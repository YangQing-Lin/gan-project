#!/bin/bash

# 备份训练成果脚本
# 用法: ./backup.sh [版本标签]
# 示例: ./backup.sh v1

VERSION=${1:-"v1"}
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
BACKUP_DIR="models/${VERSION}_${TIMESTAMP}"

mkdir -p "$BACKUP_DIR"

# 备份模型文件
cp gen.pth "$BACKUP_DIR/gen.pth" 2>/dev/null && echo "✓ gen.pth"
cp disc.pth "$BACKUP_DIR/disc.pth" 2>/dev/null && echo "✓ disc.pth"
cp loss_curve.png "$BACKUP_DIR/loss_curve.png" 2>/dev/null && echo "✓ loss_curve.png"

echo ""
echo "备份完成: $BACKUP_DIR"
ls -la "$BACKUP_DIR"
