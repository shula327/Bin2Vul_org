// 除零错误 (CWE-369)
double divide_unsafe(int numerator, int denominator) {
    // 不安全：直接进行除法，没有检查denominator是否为零
    return (double)numerator / denominator;
    
    // 如果denominator为0，将导致程序崩溃或未定义行为
} 