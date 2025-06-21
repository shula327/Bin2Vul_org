#include <stdio.h>
#include <string.h>

void CWE194_unexpected_sign_extension(short len) {
    char src[100] = "test";
    char dest[50] = {0};
    // 漏洞：len为负时会被解释为大正数
    if (len < 100) {
        memcpy(dest, src, len);
        dest[49] = '\0';
    }
} 