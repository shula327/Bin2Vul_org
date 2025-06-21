void CWE457_use_of_uninitialized_variable() {
    int x;
    // 漏洞：未初始化直接使用
    int y = x + 1;
} 