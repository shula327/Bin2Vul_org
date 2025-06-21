// 不安全：使用弱加密算法
EncryptionResult encrypt_weak(const char* plaintext) {
    printf("[不安全] 使用弱加密算法加密: %s\n", plaintext);
    // 模拟弱加密
    EncryptionResult result = { .data = strdup(plaintext), .length = strlen(plaintext) };
    return result;
} 