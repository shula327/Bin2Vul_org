// 不安全：未验证的URL重定向
void redirect_user_unsafe(const char* url) {
    // 直接使用用户提供的URL进行重定向，没有验证
    printf("HTTP/1.1 302 Found\r\nLocation: %s\r\n\r\n", url);
} 