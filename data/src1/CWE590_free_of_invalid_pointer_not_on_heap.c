void CWE590_free_of_invalid_pointer_not_on_heap() {
    char str[20] = "test string";
    // 漏洞：释放栈上分配的内存
    free(str);
} 