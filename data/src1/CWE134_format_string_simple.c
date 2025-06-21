// 格式化字符串漏洞 (CWE-134)
void log_message_unsafe(const char* user_input) {
    // 不安全：用户输入直接作为格式化字符串
    printf(user_input);
    // 攻击者可以提供如 "%x %x %x" 的输入来泄露栈中的数据
} 