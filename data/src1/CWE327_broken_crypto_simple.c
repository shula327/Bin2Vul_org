// 使用破解或有风险的加密算法 (CWE-327)
void xor_encrypt_unsafe(const char* input, char* output, const char* key) {
    int input_len = strlen(input);
    int key_len = strlen(key);
    
    for (int i = 0; i < input_len; i++) {
        // 使用简单的XOR和循环密钥进行加密 - 非常不安全
        output[i] = input[i] ^ key[i % key_len];
    }
    
    output[input_len] = '\0';
} 