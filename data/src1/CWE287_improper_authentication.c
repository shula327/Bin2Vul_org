// 不安全：简单的硬编码密码验证
int authenticate_unsafe(const char* username, const char* password) {
    if (strcmp(username, "admin") == 0 && strcmp(password, "admin123") == 0) {
        return 1; // 验证成功
    }
    return 0; // 验证失败
} 