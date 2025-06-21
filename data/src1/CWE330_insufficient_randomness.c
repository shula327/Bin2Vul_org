// 不安全：使用可预测的随机数生成
int generate_token_unsafe() {
    // 使用可预测的种子初始化随机数
    srand(1234);  // 固定种子
    // 或 srand(time(NULL));  // 可猜测的种子

    // 生成仅有32767种可能的令牌
    return rand();
} 