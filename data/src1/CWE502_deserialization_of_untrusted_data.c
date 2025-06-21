// 不安全：直接反序列化用户输入
void deserialize_unsafe(const char* input) {
    // 直接反序列化用户输入而不验证格式或内容
    deserialize_data(input);
} 