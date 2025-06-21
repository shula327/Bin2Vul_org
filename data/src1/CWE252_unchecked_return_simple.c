// 未检查返回值 (CWE-252)
void write_to_log_unsafe(const char* filename, const char* message) {
    // 未检查返回值
    FILE* log_file = fopen(filename, "w");
    
    // 即使文件打开失败，也尝试写入
    fprintf(log_file, "日志: %s\n", message);
    
    // 如果log_file为NULL，这里将导致段错误
    fclose(log_file);
} 