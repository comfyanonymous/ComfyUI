
usage() {
    cat << EOF
用法: $0 <mapping_file> <source_dir> <dest_dir> [--move|--copy] [--dry-run]

參數:
  mapping_file  - 映射檔案 (格式: subdir model_name url)
  source_dir    - 來源目錄
  dest_dir      - 目標目錄
  --move        - 移動檔案 (預設)
  --copy        - 複製檔案
  --dry-run     - 僅顯示操作,不實際執行

範例:
  $0 models.txt /mnt/1T/Download /home/user/Comfyui/models --move
  $0 models.txt /mnt/1T/Download /home/user/Comfyui/models --copy --dry-run
EOF
    exit 1
}

# 檢查參數
if [ $# -lt 3 ]; then
    usage
fi

MAPPING_FILE="$1"
SOURCE_DIR="$2"
DEST_DIR="$3"
ACTION="move"
DRY_RUN=false

# 解析選項
shift 3
while [ $# -gt 0 ]; do
    case "$1" in
        --move)
            ACTION="move"
            ;;
        --copy)
            ACTION="copy"
            ;;
        --dry-run)
            DRY_RUN=true
            ;;
        *)
            echo "未知選項: $1"
            usage
            ;;
    esac
    shift
done

# 檢查檔案和目錄
if [ ! -f "$MAPPING_FILE" ]; then
    echo "錯誤: 映射檔案 '$MAPPING_FILE' 不存在"
    exit 1
fi

if [ ! -d "$SOURCE_DIR" ]; then
    echo "錯誤: 來源目錄 '$SOURCE_DIR' 不存在"
    exit 1
fi

# 統計
FOUND=0
NOTFOUND=0
DOWNLOADED=0
FAILED=0

echo "=========================================="
echo "映射檔案: $MAPPING_FILE"
echo "來源目錄: $SOURCE_DIR"
echo "目標目錄: $DEST_DIR"
echo "動作: $ACTION"
echo "Dry-run: $DRY_RUN"
echo "=========================================="
echo ""

# 讀取映射檔案並處理
while IFS= read -r line; do
    # 跳過空行和註解
    [[ -z "$line" || "$line" =~ ^[[:space:]]*# ]] && continue
    
    # 解析行並移除前後空格
    line=$(echo "$line" | sed 's/^[[:space:]]*//;s/[[:space:]]*$//')
    
    # 跳過空行
    [[ -z "$line" ]] && continue
    
    # 解析: subdir / model_name url
    # 先分割第一個部分 (subdir / model_name)
    if [[ "$line" =~ ^([^[:space:]]+)[[:space:]]*/[[:space:]]*([^[:space:]]+)[[:space:]]+(.+)$ ]]; then
        subdir="${BASH_REMATCH[1]}"
        model_name="${BASH_REMATCH[2]}"
        url="${BASH_REMATCH[3]}"
    else
        echo "警告: 無法解析行: $line"
        continue
    fi
    
    # 移除多餘斜線
    subdir="${subdir%/}"
    
    echo "處理: $model_name"
    echo "  子目錄: $subdir"
    
    # 在來源目錄遞迴搜尋檔案
    SOURCE_FILE=$(find "$SOURCE_DIR" -type f -name "$model_name" 2>/dev/null | head -n 1)
    
    DEST_SUBDIR="$DEST_DIR/$subdir"
    DEST_FILE="$DEST_SUBDIR/$model_name"
    
    if [ -n "$SOURCE_FILE" ]; then
        echo "  找到: $SOURCE_FILE"
        
        if $DRY_RUN; then
            echo "  [DRY-RUN] 將 $ACTION 到: $DEST_FILE"
        else
            # 建立目標子目錄
            mkdir -p "$DEST_SUBDIR"
            
            if [ "$ACTION" = "move" ]; then
                mv -v "$SOURCE_FILE" "$DEST_FILE"
                echo "  已移動到: $DEST_FILE"
            else
                cp -v "$SOURCE_FILE" "$DEST_FILE"
                echo "  已複製到: $DEST_FILE"
            fi
        fi
        ((FOUND++))
        
    elif [ -n "$url" ]; then
        echo "  未找到,嘗試下載"
        echo "  URL: $url"
        
        if $DRY_RUN; then
            echo "  [DRY-RUN] 將下載到: $DEST_FILE"
            ((NOTFOUND++))
        else
            mkdir -p "$DEST_SUBDIR"
            
            if wget -q --show-progress -O "$DEST_FILE" "$url"; then
                echo "  已下載到: $DEST_FILE"
                ((DOWNLOADED++))
            else
                echo "  下載失敗"
                rm -f "$DEST_FILE"
                ((FAILED++))
            fi
        fi
        
    else
        echo "  未找到且無下載連結"
        ((NOTFOUND++))
    fi
    
    echo ""
    
done < "$MAPPING_FILE"

echo "=========================================="
echo "處理完成"
echo "已處理 (${ACTION}): $FOUND"
echo "已下載: $DOWNLOADED"
echo "未找到: $NOTFOUND"
echo "失敗: $FAILED"
echo "=========================================="
