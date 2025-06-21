// 不安全：可控制长度的缓冲区访问
void copy_data_unsafe(char* dest, char* src, int length) {
    // 没有检查dest缓冲区大小，直接复制指定长度的数据
    for(int i = 0; i < length; i++) {
        dest[i] = src[i];
    }
} 