void CWE464_addition_of_data_structure_sentinel(char *input) {
    char arr[5] = {'A', 'B', 'C', 'D', 'E'};
    // 漏洞：错误地将\0插入数组中
    arr[2] = input[0];
} 