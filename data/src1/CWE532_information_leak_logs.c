// 不安全：在日志中记录敏感信息
void log_auth_attempt_unsafe(const char* username, const char* password) {
    // 直接在日志中记录密码
    printf("LOG: 尝试登录 - 用户名: %s, 密码: %s\n", username, password);
} 