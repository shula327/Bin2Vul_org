// 不安全：固定大小缓冲区没有边界检查
void process_input_unsafe(const char* input) {
    char buffer[64];
    strcpy(buffer, input);  // 如果input超过64字节，会导致缓冲区溢出
} 