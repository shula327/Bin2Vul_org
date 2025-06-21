// 内存泄漏 (CWE-401)
char* process_data_unsafe(const char* input) {
    // 分配内存
    char* buffer = (char*)malloc(strlen(input) + 1);
    
    if (buffer == NULL) {
        return NULL;
    }
    
    // 复制和处理数据
    strcpy(buffer, input);
    
    // 分配一个新缓冲区用于返回结果
    char* result = (char*)malloc(strlen(buffer) + 10);
    
    if (result == NULL) {
        // 内存泄漏！遇到错误但没有释放先前分配的buffer
        return NULL;
    }
    
    // 组合结果
    sprintf(result, "处理: %s", buffer);
    
    // 内存泄漏！buffer未释放就返回了
    return result;
} 