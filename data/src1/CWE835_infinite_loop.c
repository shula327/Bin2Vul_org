// 不安全：可能的无限循环
void process_data_unsafe(const char* data) {
    int i = 0;
    while (data[i] != '\0') {
        if (data[i] == '\n') {
            continue;  // 没有增加i，可能导致无限循环
        }
        process_char(data[i]);
        i++;
    }
} 
 