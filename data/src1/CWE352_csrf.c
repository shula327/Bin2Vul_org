// 不安全：处理敏感操作没有CSRF保护
void process_transfer_unsafe(const char* to_account, float amount) {
    // 直接执行转账，没有验证请求来源（没有CSRF令牌）
    transfer_money(current_user, to_account, amount);
    printf("转账已完成\n");
} 