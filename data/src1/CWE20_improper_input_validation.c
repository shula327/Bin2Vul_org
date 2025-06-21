// 不安全：缺少输入验证的处理函数
void process_data_unsafe(int user_input) {
    // 直接使用用户输入而不验证
    int result = 100 / user_input;  // 如果输入为0，会导致除零错误
} 