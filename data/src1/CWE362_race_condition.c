// 不安全：存在竞态条件的共享资源访问
void withdraw_unsafe(Account* account, double amount) {
    // 读取当前余额
    double balance = account->balance;
    
    // 检查余额是否足够（这里可能发生竞态条件）
    if (balance >= amount) {
        // 模拟处理延迟
        sleep(1);
        
        // 更新余额（如果另一个线程在此期间更改了余额，可能导致超额提取）
        account->balance = balance - amount;
        printf("提取成功，当前余额: %.2f\n", account->balance);
    } else {
        printf("余额不足\n");
    }
} 