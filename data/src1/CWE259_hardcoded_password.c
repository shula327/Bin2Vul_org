// 不安全：硬编码密码
int authenticate_administrator_unsafe(const char* password) {
    // 直接在代码中硬编码管理员密码
    if (strcmp(password, "super_secure_admin123") == 0) {
        return 1; // 认证成功
    }
    return 0; // 认证失败
} 