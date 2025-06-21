// 不安全：整数计算没有溢出检查
int multiply_unsafe(int a, int b) {
    return a * b;  // 可能导致整数溢出
} 