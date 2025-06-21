// 整数下溢 (CWE-191)
unsigned int subtract_unsafe(unsigned int a, unsigned int b) {
    // 直接进行减法，没有检查结果是否会下溢
    unsigned int result = a - b;
    
    // 如果a < b，会发生整数下溢，结果是一个很大的数
    return result;
} 