// 不安全：使用弱伪随机数生成器
int generate_encryption_key_unsafe(unsigned char* key, size_t len) {
    // 使用不安全的随机数源
    srand(time(NULL));
    
    // 生成低熵的密钥
    for (size_t i = 0; i < len; i++) {
        // rand()只有15位熵，对于加密应用远远不够
        key[i] = rand() & 0xFF;
    }
    
    return 0;
} 