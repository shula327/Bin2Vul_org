
void CWE761_free_not_at_start_of_buffer() {
    char *data = (char *)malloc(100);
    if (data == 0) return;
    // 漏洞：释放非起始地址
    free(data + 5);
} 