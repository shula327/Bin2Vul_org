// 不安全：没有边界检查的数组写入
void write_data_unsafe(int* buffer, size_t index, int value) {
    buffer[index] = value;  // 没有检查index是否超出buffer边界
} 