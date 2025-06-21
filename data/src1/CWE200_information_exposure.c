// 不安全：暴露敏感信息的错误处理
void process_payment_unsafe(const char* card_number, const char* cvv) {
    FILE* log = fopen("transaction.log", "a");
    
    // 尝试处理
    int result = process_credit_card(card_number, cvv);
    
    if (result != SUCCESS) {
        // 记录详细错误，包括敏感信息
        fprintf(log, "支付失败，卡号：%s，CVV：%s，错误：%d\n", 
                card_number, cvv, result);
        printf("支付处理失败，请查看日志了解详情\n");
    }
    
    fclose(log);
} 