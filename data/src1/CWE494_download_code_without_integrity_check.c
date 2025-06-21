// 不安全：下载并执行未验证完整性的代码
void update_application_unsafe(const char* update_url) {
    // 下载更新包
    char* update_package = download_file(update_url);
    
    // 直接安装和执行，没有验证签名或校验和
    install_package(update_package);
    
    // 没有验证源，攻击者可能提供恶意更新
} 