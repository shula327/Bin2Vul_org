
void CWE680_integer_overflow_to_buffer_overflow(int count) {
    // 漏洞：count过大导致malloc分配空间溢出
    int *arr = (int *)malloc(count * sizeof(int));
    if (arr == 0) return;
    for (int i = 0; i < count; i++) {
        arr[i] = i;
    }
    printf("arr[0]=%d\n", arr[0]);
    free(arr);
} 