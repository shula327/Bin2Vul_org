// 不安全：使用弱哈希方法存储密码
char* hash_password_unsafe(const char* password) {
    // 使用简单的MD5哈希，没有加盐
    return md5(password);
    
    // 或使用已知不安全的哈希
    // return sha1(password);
} 