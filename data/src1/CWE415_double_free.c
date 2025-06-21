
void CWE415_double_free() {
    char *data = (char *)malloc(100);
    if (data == 0) return;
    free(data);
    // 漏洞：重复释放同一指针
    free(data);
} 