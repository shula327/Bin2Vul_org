// 栈缓冲区溢出 (CWE-121)
void copy_data_unsafe(const char* input) {
    // 在栈上分配固定大小的缓冲区
    char buffer[8];
    
    // 不安全：不检查输入长度是否超过缓冲区
    strcpy(buffer, input);  // 如果input超过8个字符，会发生栈溢出
    
    // 此时程序可能已经崩溃，或者栈上的重要数据被破坏
} 