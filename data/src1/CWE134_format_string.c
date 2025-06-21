// 不安全：直接将用户输入作为格式字符串
void format_unsafe(const char* user_input) {
    printf(user_input);  // 用户输入直接作为格式字符串参数
} 