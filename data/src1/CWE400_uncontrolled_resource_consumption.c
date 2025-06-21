// 不安全：无限制的资源分配
void process_request_unsafe(const char* input, size_t size) {
    void* buffer = malloc(size);  // 无大小限制分配内存
    // 处理数据...
    free(buffer);
} 