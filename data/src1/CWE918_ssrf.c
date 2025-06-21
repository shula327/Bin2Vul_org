// 不安全：直接使用用户输入作为URL
void fetch_url_unsafe(const char* url) {
    // 直接使用用户提供的URL进行请求
    http_request(url);
}