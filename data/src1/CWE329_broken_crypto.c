#include <stdio.h>
#include <stdlib.h>
#include <time.h>

// 不安全：使用不安全的随机数生成
void generate_key_unsafe(unsigned char* key, size_t key_length) {
    srand(time(NULL));
    for (size_t i = 0; i < key_length; i++) {
        key[i] = rand() % 256;
    }
}