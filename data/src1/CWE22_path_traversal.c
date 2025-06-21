// 不安全：直接拼接路径
void read_file_unsafe(const char* filename) {
    char filepath[256];
    snprintf(filepath, sizeof(filepath), "data/%s", filename);
    printf("[不安全] 尝试读取文件: %s\n", filepath);
    // 执行文件读取...
} 