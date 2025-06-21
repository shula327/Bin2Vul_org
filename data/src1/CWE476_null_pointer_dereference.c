// 不安全：没有检查空指针
void process_data_unsafe(char* data) {
    printf("数据长度: %zu\n", strlen(data));  // 如果data为NULL，会触发空指针解引用
} 